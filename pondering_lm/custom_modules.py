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
        seed: int = None
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
            pondering_steps
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
        embed_scale: float = 1.0,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.pondering_steps = pondering_steps
        self.softmax_temperature = softmax_temperature
        self.embed_scale = embed_scale
        self.grad_checkpointing = grad_checkpointing

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
            
            probs = torch.softmax(logits / self.softmax_temperature, dim=-1)
            interpolated_embeds = torch.matmul(probs, self.get_base_model_embeddings() * self.embed_scale)
            return embedding + interpolated_embeds
        

        input_embedding = self.get_base_model_embeddings()[input_ids["input_ids"]]
        for _ in range(self.pondering_steps):
            if self.grad_checkpointing:
                input_embedding = checkpoint(pondering_step, input_embedding, use_reentrant=False)
            else:
                logits = self.base_model(
                    inputs=input_ids,
                    inputs_embeds=input_embedding, 
                    # attention_mask=attention_mask,
                    **kwargs
                )["logits"] # [batch_size, seq_len, vocab_size]
                
                probs = torch.softmax(logits / self.softmax_temperature, dim=-1)
                interpolated_embeds = torch.matmul(probs, self.get_base_model_embeddings() * self.embed_scale) # [batch_size, seq_len, embedding_dim]
                input_embedding.add_(interpolated_embeds)


        # Continue with base model forward pass
        outputs = self.base_model(
            inputs=input_ids,
            inputs_embeds=input_embedding,
            # attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs