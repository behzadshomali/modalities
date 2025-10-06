import os
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Tuple, Annotated
from torch import Tensor
from dataclasses import dataclass
from modalities.models.model import ActivationType, NNModel, SwiGLU
from torch.utils.checkpoint import checkpoint


class PonderingModelConfig(BaseModel):
    """Configuration for registering with Modalities."""
    base_model: nn.Module
    pondering_steps: Annotated[int, Field(3, strict=True, ge=0)]
    topk: Annotated[int, Field(-1, strict=True)]
    softmax_temperature: float = 1.0
    apply_embed_scale: bool = False
    inverse_scale: bool = False
    grad_checkpointing: bool = True
    seed: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}


class PonderingModelForCausalLM(NNModel):
    """
    PonderingLM integrated with Modalities Model interface.
    This class can be registered as a Modalities component.
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        pondering_steps: int,
        seed: int = None,
        softmax_temperature: float = 1.0,
        apply_embed_scale: bool = False,
        inverse_scale: bool = False,
        grad_checkpointing: bool = True,
        topk: int = -1
    ):
        weight_decay_groups = {
            "linear": [".attn", ".mlp", ".lm_head.weight"],
            "embedding": [".wte", ".wpe"],
            "layernorm": [".attention_norm", ".ffn_norm", ".lm_head_norm"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        
        
        # Wrap with pondering
        self.model = PonderingModelWrapper(
            base_model,
            pondering_steps=pondering_steps,
            softmax_temperature=softmax_temperature,
            apply_embed_scale=apply_embed_scale,
            inverse_scale=inverse_scale,
            grad_checkpointing=grad_checkpointing,
            topk=topk
        )
        
    # def _init_base_model(self, base_config: Dict[str, Any]) -> nn.Module:
    #     """Initialize base model from config."""
    #     # In real implementation, this would load the model
    #     # based on the Modalities configuration
    #     from transformers import AutoModelForCausalLM
    #     model = AutoModelForCausalLM.from_pretrained(base_config["model_name"])
        
    #     return model
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the pondering model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for base model
            
        Returns:
            Model outputs including logits and optional hidden states
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
class PonderingModelWrapper(nn.Module):
    """
    Wrapper that adds pondering capability to existing models.
    Compatible with Modalities framework.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        pondering_steps: int = 3,
        softmax_temperature: float = 1.0,
        apply_embed_scale: bool = False,
        inverse_scale: bool = False,
        grad_checkpointing: bool = True,
        topk: int = -1
    ):
        super().__init__()
        self.base_model = base_model
        self.pondering_steps = pondering_steps
        self.softmax_temperature = softmax_temperature
        self.apply_embed_scale = apply_embed_scale       
        self.inverse_scale = inverse_scale 
        self.grad_checkpointing = grad_checkpointing
        self.topk = topk


        # # vocab_size x embedding_dim
        # # The module's name MUST be named 'wte' to be compatible with weight decay grouping
        # if hasattr(self.base_model, 'get_input_embeddings'):
        #     self.wte = self.base_model.get_input_embeddings().weight
        # else:
        #     # Fallback for GPT2LLM style models
        #     self.wte = self.base_model.transformer['wte'].weight
        
    def get_base_model_embeddings(self):
        if hasattr(self.base_model, 'get_input_embeddings'):
            return self.base_model.get_input_embeddings().weight
        else:
            # Fallback for GPT2LLM style models
            return self.base_model.transformer['wte'].weight


    def forward(
        self,
        input_ids: dict,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward pass with pondering.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for base model
            
        Returns:
            Model outputs including logits and optional hidden states
        """
        
        def pondering_step(embedding):
            logits = self.base_model(
                inputs=input_ids,
                inputs_embeds=embedding, 
                **kwargs
            )["logits"]
            

            if self.topk > 0:
                # Top-K optimization
                top_k_logits, top_k_indices = torch.topk(
                    logits, 
                    k=self.topk, 
                    dim=-1
                )  # [batch_size, seq_len, K]
                
                top_k_probs = torch.softmax(top_k_logits / self.softmax_temperature, dim=-1)  # [batch_size, seq_len, K]
                
                # Gather embeddings for top-K tokens
                embeddings = self.get_base_model_embeddings()  # [vocab_size, embed_dim]
                top_k_embeds = embeddings[top_k_indices]  # [batch_size, seq_len, K, embed_dim]
                
                # Weighted sum of top-K embeddings
                interpolated_embeds = torch.einsum(
                    'bsk,bske->bse', 
                    top_k_probs, 
                    top_k_embeds * self.embed_scale
                )  # [batch_size, seq_len, embed_dim]
            else:
                # Full vocabulary (original implementation)
                probs = torch.softmax(logits / self.softmax_temperature, dim=-1)
                interpolated_embeds = torch.matmul(probs, self.get_base_model_embeddings() * self.embed_scale)
            
            return embedding + interpolated_embeds
        

        input_embedding = self.get_base_model_embeddings()[input_ids["input_ids"]]

        if self.apply_embed_scale is not None:
            if self.inverse_scale:
                self.embed_scale = 1 / torch.sqrt(torch.tensor(input_embedding.shape[-1], dtype=input_embedding.dtype))
            else:
                self.embed_scale = torch.sqrt(torch.tensor(input_embedding.shape[-1], dtype=input_embedding.dtype))
        else:
            self.embed_scale = torch.tensor(1.0)
        
        for _ in range(self.pondering_steps):
            if self.grad_checkpointing:
                input_embedding = checkpoint(pondering_step, input_embedding, use_reentrant=False)
            else:
                input_embedding = pondering_step(input_embedding)


        # Continue with base model forward pass
        outputs = self.base_model(
            inputs=input_ids,
            inputs_embeds=input_embedding,
            **kwargs
        )
        
        return outputs