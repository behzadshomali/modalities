# coding=utf-8
"""Pondering Model for HuggingFace Transformers"""

from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin


# Import base model components - adjust this import based on your base model
# For GPT2-style models:
from modalities.conversion.gpt2.modeling_gpt2 import GPT2ForCausalLM, GPT2Model
from modalities.conversion.pondering.configuration_pondering import PonderingModelConfig



__all__ = [
    "PonderingModelConfig",
    "PonderingModel",
    "PonderingModelForCausalLM",
    "PonderingPreTrainedModel",
]



import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM, GPT2Model
from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.conversion_model import _get_layer_norm_value, _map_attention_type
from modalities.models.model import SwiGLU

from typing import Annotated


class PonderingModelConfig(PretrainedConfig):
    """
    HuggingFace-compatible configuration for PonderingModel.
    
    Args:
        base_model_config: Config dict for the base model
        pondering_steps: Number of pondering iterations
        topk: Top-K sampling for efficiency (-1 for full vocab)
        softmax_temperature: Temperature for softmax in pondering
        apply_embed_scale: Whether to apply embedding scaling
        inverse_scale: Use inverse square root scaling
        grad_checkpointing: Enable gradient checkpointing
    """
    model_type = "pondering"
    is_composition = False
    
    def __init__(
        self,
        base_model_config: Optional[Dict] = None,
        base_model_type: str = "gpt2",
        pondering_steps: int = 3,
        topk: int = -1,
        softmax_temperature: float = 1.0,
        apply_embed_scale: bool = False,
        inverse_scale: bool = False,
        grad_checkpointing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.inner_model_config = base_model_config or {}
        self.inner_model_type = base_model_type
        self.pondering_steps = pondering_steps
        self.topk = topk
        self.softmax_temperature = softmax_temperature
        self.apply_embed_scale = apply_embed_scale
        self.inverse_scale = inverse_scale
        self.grad_checkpointing = grad_checkpointing


class PonderingModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    HuggingFace-compatible PonderingModel for causal language modeling.
    
    This model wraps a base transformer model and adds pondering capability,
    allowing the model to iteratively refine token embeddings before final prediction.
    """
    config_class = PonderingModelConfig

    def __init__(self, config: PonderingModelConfig):
        super().__init__(config)
        self.config = config
        
        # this model must not be named "base_model" as it 
        # would conflict with HF's internal naming conventions
        inner_model_config = config.inner_model_config
        
        ffn_norm_key = "ffn_norm" if "ffn_norm" in config else "ffn_norm_config"
        gpt_config = GPT2Config(
            vocab_size=inner_model_config["vocab_size"],
            hidden_size=inner_model_config["n_embd"],
            pad_token_id=None,
            num_hidden_layers=inner_model_config["n_layer"],
            num_key_value_heads=inner_model_config["n_head_kv"],
            num_attention_heads=inner_model_config["n_head_q"],
            intermediate_size=SwiGLU._get_hidden_dim(ffn_hidden=inner_model_config["ffn_hidden"]),
            attention_bias=inner_model_config["bias"],
            mlp_bias=inner_model_config["bias"],
            hidden_act="silu",
            layer_norm_eps=_get_layer_norm_value(inner_model_config[ffn_norm_key]["config"], "eps"),
            layer_norm_elementwise_affine=_get_layer_norm_value(inner_model_config[ffn_norm_key]["config"], "elementwise_affine"),
            layer_norm_bias=_get_layer_norm_value(inner_model_config[ffn_norm_key]["config"], "bias"),
            max_position_embeddings=inner_model_config["sequence_length"],
            rope_theta=inner_model_config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
            _attn_implementation=_map_attention_type(inner_model_config),
            output_attentions=False,
        )
        
        # self.inner_model = GPT2ForCausalLM(gpt_config).model
        self.inner_model = GPT2ForCausalLM(gpt_config)

        # Store pondering parameters
        self.pondering_steps = config.pondering_steps
        self.softmax_temperature = config.softmax_temperature
        self.apply_embed_scale = config.apply_embed_scale
        self.inverse_scale = config.inverse_scale
        self.topk = config.topk
        
        # Initialize embedding scale
        self.embed_scale = None
        
    def get_input_embeddings(self):
        """Get input embeddings from base model."""
        if hasattr(self.inner_model, 'get_input_embeddings'):
            return self.inner_model.get_input_embeddings()
        elif hasattr(self.inner_model, 'transformer'):
            # For GPT2-style models
            return self.inner_model.transformer.wte
        else:
            raise AttributeError("Cannot find input embeddings in base model")
    
    def set_input_embeddings(self, value):
        """Set input embeddings in base model."""
        if hasattr(self.inner_model, 'set_input_embeddings'):
            self.inner_model.set_input_embeddings(value)
        elif hasattr(self.inner_model, 'transformer'):
            self.inner_model.transformer.wte = value
        else:
            raise AttributeError("Cannot set input embeddings in base model")
    
    def get_output_embeddings(self):
        """Get output embeddings from base model."""
        if hasattr(self.inner_model, 'get_output_embeddings'):
            return self.inner_model.get_output_embeddings()
        return None
    
    def set_output_embeddings(self, value):
        """Set output embeddings in base model."""
        if hasattr(self.inner_model, 'set_output_embeddings'):
            self.inner_model.set_output_embeddings(value)
    
    def _pondering_step(
        self, 
        input_ids: torch.LongTensor,
        embedding: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single pondering iteration.
        
        Args:
            embedding: Current token embeddings [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated embeddings after pondering step
        """
        # Forward pass through base model with current embeddings
        outputs = self.inner_model(
            inputs=input_ids,
            inputs_embeds=embedding,
            # attention_mask=attention_mask,
            # return_dict=True
        )
        logits = outputs['logits']
        
        # Get embedding weights
        embed_weight = self.get_input_embeddings().weight
        
        if self.topk > 0:
            # Top-K optimization for efficiency
            top_k_logits, top_k_indices = torch.topk(
                logits, 
                k=self.topk, 
                dim=-1
            )  # [batch_size, seq_len, K]
            
            top_k_probs = torch.softmax(
                top_k_logits / self.softmax_temperature, 
                dim=-1
            )  # [batch_size, seq_len, K]
            
            # Gather embeddings for top-K tokens
            top_k_embeds = embed_weight[top_k_indices]  # [batch_size, seq_len, K, embed_dim]
            
            # Weighted sum of top-K embeddings
            interpolated_embeds = torch.einsum(
                'bsk,bske->bse', 
                top_k_probs, 
                top_k_embeds * self.embed_scale
            )  # [batch_size, seq_len, embed_dim]
        else:
            # Full vocabulary (original implementation)
            probs = torch.softmax(logits / self.softmax_temperature, dim=-1)
            interpolated_embeds = torch.matmul(
                probs, 
                embed_weight * self.embed_scale
            )
        
        return embedding + interpolated_embeds
    
    def forward(
        self,
        input_ids: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with pondering.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            inputs_embeds: Pre-computed embeddings (optional)
            labels: Labels for language modeling loss
            return_dict: Whether to return ModelOutput object
            
        Returns:
            CausalLMOutputWithPast with logits and optional loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get initial embeddings
        if inputs_embeds is None:
            if type(input_ids) is dict:
                inputs_embeds = self.get_input_embeddings()(input_ids['input_ids'])
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Initialize embedding scale
        if self.apply_embed_scale:
            embed_dim = inputs_embeds.shape[-1]
            if self.inverse_scale:
                self.embed_scale = 1.0 / torch.sqrt(
                    torch.tensor(embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                )
            else:
                self.embed_scale = torch.sqrt(
                    torch.tensor(embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                )
        else:
            self.embed_scale = torch.tensor(1.0, device=inputs_embeds.device)
        
        # Pondering iterations
        for _ in range(self.pondering_steps):
            if self.config.grad_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                inputs_embeds = checkpoint(
                    self._pondering_step,
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                inputs_embeds = self._pondering_step(input_ids, inputs_embeds, attention_mask)
        
        # Final forward pass through base model
        outputs: BaseModelOutputWithPast = self.inner_model(
            inputs=input_ids,
            inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            # labels=labels,
            # return_dict=return_dict,
            **kwargs
        )
        logits = outputs['logits']

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )