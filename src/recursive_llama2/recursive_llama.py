import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers.utils.deprecation import deprecate_kwarg
from transformers import AutoTokenizer, GradientCheckpointingLayer, LlamaConfig, LlamaModel, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaPreTrainedModel, 
    LlamaRMSNorm,
    LlamaRotaryEmbedding, 
    LlamaAttention, 
    LlamaMLP 
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.cache_utils import DynamicCache, Cache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from typing import Any, List, Optional, Tuple, Union, Unpack
from trl import setup_chat_format


class CustomDynamicLayer(DynamicLayer):
    """
    Same as DynamicLayer but with a custom `update` method.
    """

    should_update_cache: bool = True
    current_iteration: int = 0
    keys_per_iteration: List[torch.Tensor] = []
    values_per_iteration: List[torch.Tensor] = []

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
    
        self.keys_per_iteration = [torch.tensor([], dtype=self.dtype, device=self.device) for _ in range(10)]  # max 10 recursions
        self.values_per_iteration = [torch.tensor([], dtype=self.dtype, device=self.device) for _ in range(10)]
        self.is_initialized = True
    

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update (on-demand) the key and value caches in-place, and return the necessary keys and value states.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if key_states.shape == torch.Size([1, 32, 1, 64]):
            pass
        if self.should_update_cache:
            self.keys_per_iteration[self.current_iteration] = torch.cat(
                [self.keys_per_iteration[self.current_iteration], key_states], dim=-2
            )
            self.values_per_iteration[self.current_iteration] = torch.cat(
                [self.values_per_iteration[self.current_iteration], value_states], dim=-2
            )
            self.keys = self.keys_per_iteration[self.current_iteration]
            self.values = self.values_per_iteration[self.current_iteration]

            return self.keys, self.values
        else:
            # If not updating the cache, just return the new states
            # return torch.cat([self.keys_per_iteration[self.current_iteration], key_states], dim=-2), torch.cat([self.values_per_iteration[self.current_iteration], value_states], dim=-2)
            k = torch.cat([self.keys_per_iteration[self.current_iteration], key_states], dim=-2)
            v = torch.cat([self.values_per_iteration[self.current_iteration], value_states], dim=-2)
            return k, v


class BlockRecursiveModule(nn.Module):
    def __init__(self, config, layer_block, num_recursions, sample_random_recursion, track_diagnostics, neft=False, neft_alpha=None):
        super().__init__()
        self.config = config # to store the modified num_recursions later
        self.layer_block = layer_block
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics
        self.neft = neft
        self.neft_alpha = neft_alpha
        self.is_cache_class_overwritten = [False] * len(layer_block)
        if track_diagnostics:
            self.gradient_history = []
            self.cosine_similarities = []
            self.step_count = 0

    def compute_cosine_similarity(self, tensor1, tensor2):
        t1 = tensor1.reshape(-1, tensor1.shape[-1])
        t2 = tensor2.reshape(-1, tensor2.shape[-1])
        cos_sim = F.cosine_similarity(t1, t2, dim=-1)
        return cos_sim

    def _overwrite_cache_class(self, layer, past_key_values, new_layer_idx):
        original_layer_idx = layer.self_attn.layer_idx
        if not isinstance(past_key_values.layers[original_layer_idx], CustomDynamicLayer):
            # Replace with CustomDynamicLayer
            original_cache = past_key_values.layers[original_layer_idx]
            if original_cache != (None, None):
                raise ValueError(f"Expected None cache at layer {original_layer_idx} before overwriting, got non-None.")
            new_cache = CustomDynamicLayer()
            past_key_values.layers[original_layer_idx] = new_cache
            print(f"Overwritten cache class for layer {original_layer_idx} to CustomDynamicLayer.")
            self.is_cache_class_overwritten[new_layer_idx] = True

    def _set_current_iteration(self, layer, past_key_values, iteration_idx: int):
        original_layer_idx = layer.self_attn.layer_idx
        if isinstance(past_key_values.layers[original_layer_idx], CustomDynamicLayer):
            past_key_values.layers[original_layer_idx].current_iteration = iteration_idx
        else:
            raise ValueError(f"Cache for layer {original_layer_idx} is not a CustomDynamicLayer.")
        
    def set_num_recursions(self, num_recursions: int):
        self.num_recursions = num_recursions
        self.config.num_recursions = num_recursions

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
            for i, layer in enumerate(self.layer_block):
                if past_key_values is not None:
                    original_layer_idx = layer.self_attn.layer_idx
                    if not isinstance(past_key_values.layers[original_layer_idx], CustomDynamicLayer):
                        if past_key_values[original_layer_idx] == (None, None):
                            self._overwrite_cache_class(layer, past_key_values, i)
                        else:
                            raise ValueError(f"Expected None cache at layer {original_layer_idx} before overwriting, got non-None.")
                
                if past_key_values is not None:
                    self._set_current_iteration(layer, past_key_values, iteration_idx=iteration)

                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )

                if self.neft and self.training:
                    alpha = self.neft_alpha
                    L = hidden_states.shape[-2]
                    d = hidden_states.shape[-1]
                    noise = torch.rand_like(hidden_states) * 2 -1 # range: [-1,1]
                    scaled_noise = noise * alpha / ((L*d)**0.5)
                    hidden_states = hidden_states + scaled_noise

                if self.track_diagnostics and self.training:
                    layer_name = f"recursive_block_layer_{i}"
                    grad_hook = self._make_grad_hook(layer_name, iteration, is_recurrent=True)
                    hidden_states.register_hook(grad_hook)

            if self.track_diagnostics and self.training:
                current_activation = hidden_states.detach().clone()
                cos_sim = self.compute_cosine_similarity(prev_hidden_states, current_activation)
                self.cosine_similarities.append({
                    'iteration': iteration,
                    'mean': cos_sim.mean().item(),
                    'min': cos_sim.min().item(),
                    'max': cos_sim.max().item(),
                    'std': cos_sim.std().item(),
                })
                prev_hidden_states = current_activation


        return hidden_states

    def _make_grad_hook(self, layer_name, iteration_idx, is_recurrent):
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
        neft=False,
        neft_alpha=None,
        gradually_increase_recursions=False,
        reset_optimizer=False,
        increase_steps=None,
        recurrent_blocks_have_residual=True,
        **kwargs,
    ):

        # Case 1: Initial creation from a base model
        if model_name:
            # Load the base config to get its parameters
            base_config = LlamaConfig.from_pretrained(model_name)
            base_config_dict = base_config.to_dict()

            # Get original layers from the base config
            original_layers = base_config.num_hidden_layers
            
            base_config_dict.pop("model_type", None) # We're setting our own
            kwargs.update(base_config_dict)
            
            super().__init__(**kwargs)
            
            # Store the original layer count
            self.original_num_hidden_layers = original_layers

        # Case 2: Loading from config.json (model_name is None)
        else:
            # All properties are already in `kwargs`, loaded from config.json.
            # We just pass them all to the parent.
            super().__init__(**kwargs)
            
            # `original_num_hidden_layers` will be passed from kwargs
            # (or be None if it wasn't saved, though it should be).
            self.original_num_hidden_layers = original_num_hidden_layers

        # Finally, set all custom attributes.
        # This runs in both cases (initial creation and loading), 
        # ensuring the object has the correct values (either 
        # from user args or from the loaded config.json).
        self.recursion_start_layer = recursion_start_layer
        self.recursion_end_layer = recursion_end_layer
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics
        self.neft = neft
        self.neft_alpha = neft_alpha
        self.recurrent_blocks_have_residual = recurrent_blocks_have_residual
        self.gradually_increase_recursions = gradually_increase_recursions
        self.reset_optimizer = reset_optimizer
        self.increase_steps = increase_steps if increase_steps is not None else []




class RecursiveLlamaModel(LlamaModel):
    config_class = RecursiveLlamaConfig

    def __init__(self, config: RecursiveLlamaConfig):
        super(LlamaModel, self).__init__(config)

        # Manually copy LlamaModel's __init__ logic
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
            if config.recurrent_blocks_have_residual:
                layer_block.append(LlamaDecoderLayer(config, layer_idx=idx))
            else:
                print("Adding layer without residual:", idx)
                layer_block.append(LlamaDecoderLayerWOResidual(config, layer_idx=idx))

        self.layers.append(BlockRecursiveModule(
            config,
            layer_block=layer_block,
            num_recursions=config.num_recursions,
            sample_random_recursion=config.sample_random_recursion,
            track_diagnostics=config.track_diagnostics,
            neft=config.neft,
            neft_alpha=config.neft_alpha
        ))
        
        # 3. Add layers *after* the block
        next_non_recurrent_layer_idx = start + 1
        for idx in range(end + 1, config.original_num_hidden_layers):
            self.layers.append(LlamaDecoderLayer(config, layer_idx=idx))
            next_non_recurrent_layer_idx += 1

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

    def __init__(self, config: RecursiveLlamaConfig, use_bf16=True):
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = RecursiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        try:
            self.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
        
            self.model.norm.load_state_dict(base_model.model.norm.state_dict())
            self.lm_head.load_state_dict(base_model.lm_head.state_dict())


            RECURSION_START = config.recursion_start_layer
            RECURSION_END = config.recursion_end_layer

            # Layers BEFORE the block
            self.model.layers[:RECURSION_START].load_state_dict(
                base_model.model.layers[:RECURSION_START].state_dict()
            )

            # Layers INTO the block
            # model.model.layers[RECURSION_START] is our BlockRecursiveModule
            self.model.layers[RECURSION_START].layer_block.load_state_dict(
                base_model.model.layers[RECURSION_START : RECURSION_END + 1].state_dict()
            )

            # Layers AFTER the block
            # The new index is RECURSION_START + 1
            # The original index is RECURSION_END + 1
            self.model.layers[RECURSION_START + 1 :].load_state_dict(
                base_model.model.layers[RECURSION_END + 1 :].state_dict()
            )
        except Exception as e:
            print("Error loading state dict from base model:", e)
            print("Continuing with randomly initialized weights.")

        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
            self.lm_head = self.lm_head.to(torch.bfloat16)


        del base_model


        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class LlamaDecoderLayerWOResidual(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


AutoConfig.register("recursive-llama", RecursiveLlamaConfig)
AutoModelForCausalLM.register(RecursiveLlamaConfig, RecursiveLlamaForCausalLM)