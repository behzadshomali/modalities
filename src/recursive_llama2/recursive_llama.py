import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaPreTrainedModel, 
    LlamaRMSNorm,
    # 1. Import the necessary Llama components
    LlamaRotaryEmbedding, 
    LlamaAttention, 
    LlamaMLP 
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.cache_utils import DynamicCache, Cache
from transformers.masking_utils import create_causal_mask
from typing import List, Optional, Tuple, Union, Unpack


class BlockRecursiveModule(nn.Module):
    def __init__(self, layer_block, num_recursions, sample_random_recursion, track_diagnostics):
        super().__init__()
        self.layer_block = layer_block
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics
        if track_diagnostics:
            self.gradient_history = []
            self.cosine_similarities = []
            self.step_count = 0

    def compute_cosine_similarity(self, tensor1, tensor2):
        t1 = tensor1.reshape(-1, tensor1.shape[-1])
        t2 = tensor2.reshape(-1, tensor2.shape[-1])
        cos_sim = F.cosine_similarity(t1, t2, dim=-1)
        return cos_sim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        if 'cache_position' in kwargs:
            kwargs.pop('cache_position')

        if self.sample_random_recursion and self.training:
            num_recursions = random.randint(1, self.num_recursions)
        else:
            num_recursions = self.num_recursions

        if self.track_diagnostics and self.training:
            prev_hidden_states = hidden_states.detach().clone()

        # all_present_key_values = past_key_values  # initial cache
        for iteration in range(num_recursions):
            is_final = (iteration == num_recursions - 1)

            all_present_key_values = past_key_values if is_final else None

            for layer in self.layer_block:
                print(f"XXX Input to iteration {iteration}:")
                # let every layer update the *same* cache object
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=all_present_key_values,
                    use_cache=is_final,          # write only in last iteration
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )

        # return exactly what a normal decoder layer would return
        if use_cache:
            return hidden_states, all_present_key_values
        else:
            return hidden_states

    def _make_grad_hook(self, layer_idx, iteration_idx, is_recurrent):
        def grad_hook(grad):
            # record grad info
            self.gradient_history.append({
                'layer_name': layer_name,
                'is_recurrent': is_recurrent,
                'iteration': iteration_idx,
                'step': self.step_count,
                'grad_norm': grad.norm().item(),
                'grad_mean': grad.mean().item(),
                'grad_std': grad.std().item(),
                'grad_max': grad.abs().max().item()
            })
            # increment step counter
            self.step_count += 1
            return grad
        return grad_hook

    def get_diagnostics(self):
        return {
            'cosine_similarities': self.cosine_similarities,
            'gradient_history': self.gradient_history,
        }
        


class RecursiveLlamaConfig(LlamaConfig):
    model_type = "recursive-llama"

    def __init__(
        self,
        model_name=None,                 # Used ONLY for initial creation
        recursion_start_layer=None,
        recursion_end_layer=None,
        num_recursions=None,
        original_num_hidden_layers=None, # Default to None
        sample_random_recursion=False,
        track_diagnostics=False,
        **kwargs,
    ):

        # Case 1: Initial creation from a base model
        if model_name:
            # Load the base config to get its parameters
            base_config = LlamaConfig.from_pretrained(model_name)
            base_config_dict = base_config.to_dict()

            # Get original layers from the base config
            original_layers = base_config.num_hidden_layers
            
            # Populate kwargs with properties from the base config
            # (e.g., vocab_size, hidden_size, etc.)
            # base_config_dict.pop("num_hidden_layers", None)
            base_config_dict.pop("model_type", None) # We're setting our own
            kwargs.update(base_config_dict)

            # Calculate the new number of hidden layers
            num_layers_before = recursion_start_layer
            num_layers_after = original_layers - (recursion_end_layer + 1)
            new_num_hidden_layers = num_layers_before + 1 + num_layers_after
            # kwargs["num_hidden_layers"] = new_num_hidden_layers
            
            # Pass the *calculated* new_num_hidden_layers to the parent
            super().__init__(**kwargs)
            
            # Store the original layer count
            self.original_num_hidden_layers = original_layers

        # Case 2: Loading from config.json (model_name is None)
        else:
            # All properties, including the correct (modified) `num_hidden_layers`,
            # are already in `kwargs`, loaded from config.json.
            # We just pass them all to the parent.
            super().__init__(**kwargs)
            
            # `original_num_hidden_layers` will be passed from kwargs
            # (or be None if it wasn't saved, though it should be).
            self.original_num_hidden_layers = original_num_hidden_layers

        # Finally, set all custom attributes.
        # This runs in both cases, ensuring the object has the correct
        # values (either from user args or from the loaded config.json).
        self.recursion_start_layer = recursion_start_layer
        self.recursion_end_layer = recursion_end_layer
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics



class RecursiveLlamaModel(LlamaModel):
    config_class = RecursiveLlamaConfig

    def __init__(self, config: RecursiveLlamaConfig):
        super(LlamaModel, self).__init__(config)

        # Manually copy LlamaModel's __init__ logic (including missing parts)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        self.rotary_emb = LlamaRotaryEmbedding(config=config) 

        self.layers = nn.ModuleList()
        start = config.recursion_start_layer
        end = config.recursion_end_layer

        # 1. Add layers *before* the block
        for idx in range(start):
            self.layers.append(LlamaDecoderLayer(config, layer_idx=idx))
        
        # 2. Create and add the *recursive block*
        layer_block = nn.ModuleList()
        for idx in range(start, end + 1):
            # idx = start
            layer_block.append(LlamaDecoderLayer(config, layer_idx=idx))
            


        
        self.layers.append(BlockRecursiveModule(
            layer_block=layer_block,
            num_recursions=config.num_recursions,
            sample_random_recursion=config.sample_random_recursion,
            track_diagnostics=config.track_diagnostics
        ))
        
        # 3. Add layers *after* the block
        next_non_recurrent_layer_idx = start + 1
        for idx in range(end + 1, config.original_num_hidden_layers):
            # idx = next_non_recurrent_layer_idx
            # print(idx)
            self.layers.append(LlamaDecoderLayer(config, layer_idx=idx))
            next_non_recurrent_layer_idx += 1

        
        """
        File "/raid/s3/opengptx/behzad_shomali/miniforge3/envs/lighteval_env/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py", line 252, in forward
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/raid/s3/opengptx/behzad_shomali/miniforge3/envs/lighteval_env/lib/python3.11/site-packages/transformers/cache_utils.py", line 776, in update
    keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
                   ~~~~~~~~~~~^^^^^^^^^^^
IndexError: list index out of range
        """

        # missing_layers_num = end - start
        # print("Missing Layers num:", missing_layers_num)
        # for _ in range(missing_layers_num):
        #     self.layers.append(nn.Identity())

        print("Number of final layers:", len(self.layers))
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if isinstance(decoder_layer, BlockRecursiveModule):
                pass
            
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



class RecursiveLlamaForCausalLM(LlamaForCausalLM):

    config_class = RecursiveLlamaConfig

    def __init__(self, config: RecursiveLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = RecursiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()


# AutoConfig.register("recursive-llama", RecursiveLlamaConfig)
# AutoModelForCausalLM.register(RecursiveLlamaConfig, RecursiveLlamaForCausalLM)