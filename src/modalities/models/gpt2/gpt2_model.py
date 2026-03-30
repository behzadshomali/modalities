import logging
import math
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Callable, List, Mapping, Optional, Any, overload, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from pydantic import BaseModel, Field, model_validator, validator

from modalities.config.lookup_enum import LookupEnum
from modalities.config.utils import convert_base_model_config_to_dict
from modalities.models.components.layer_norms import (
    LayerNormConfig,
    PytorchRMSLayerNormConfig,
    RMSLayerNorm,
    RMSLayerNormConfig,
)
from modalities.models.model import ActivationType, NNModel, SwiGLU
from modalities.util import parse_enum_by_name

try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func = None

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class LayerNorms(LookupEnum):
    """
    Enum lookup class for LayerNorms.

    Attributes:
        RMSNorm: RMSLayerNorm class.
        LayerNorm: nn.LayerNorm class.
        PyTorchRMSNorm: nn.RMSNorm class.
    """

    rms_norm = RMSLayerNorm
    layer_norm = nn.LayerNorm
    pytorch_rms_norm = nn.RMSNorm


class LayerNormWrapperConfig(BaseModel):
    norm_type: LayerNorms
    config: PytorchRMSLayerNormConfig | RMSLayerNormConfig | LayerNormConfig


class PositionTypes(str, Enum):
    """
    Enum class representing different position types.

    Attributes:
        ABSOLUTE (str): Represents the absolute position type.
        NOPE (str): Represents the nope (no postional emebddigns) position type.
    """

    ABSOLUTE = "ABSOLUTE"
    NOPE = "NOPE"

class BlockTypes(str, Enum):
    """
    Enum class representing different block types.

    Attributes:
        STANDARD (str): Represents the standard GPT2 block type.
        GROUP_RECURSIVE_MTP (str): Represents the recursive GPT2 block type used for MTP.
        GROUP_RECURSIVE (str): Represents the group recursive GPT2 block type.
    """

    STANDARD = "STANDARD"
    RECURSIVE = "RECURSIVE"
    GROUP_RECURSIVE_MTP = "GROUP_RECURSIVE_MTP"
    GROUP_RECURSIVE = "GROUP_RECURSIVE"
    COMBINED_REPRESENTATION = "COMBINED_REPRESENTATION"
    HALT = "HALT"


class QueryKeyValueTransform(nn.Module):
    """Query Key Value Transform base class."""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for transforming queries/keys/values.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the output tensors.
        """
        raise NotImplementedError


class IdentityTransform(QueryKeyValueTransform):
    """IdentityTransform class which does not apply any transform."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the IdentityTransform which does not apply any transform.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The tensors q, k, and v.
        """
        return q, k, v


class RotaryTransform(QueryKeyValueTransform):
    """
    RotaryTransform class which implements rotary positional embeddings.

    Source: https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
            We added the corresponding code here, becauase there is a conflict with "@torch.jit.script" used in the
            XFormers implementation and removed in this implementation.#
    """

    def __init__(self, n_embd: int, n_head: int, seq_length_dim: int = -2, base_freq: int = 10000):
        """
        Initializes the RotaryTransform object.

        Args:
            n_embd (int): The size of the embedding dimension.
            n_head (int): The number of attention heads.
            seq_length_dim (int, optional): The dimension along which the sequence length is defined. Defaults to -2.
            base_freq (int): Base frequency for RoPE. Defaults to 10000.
        """
        super().__init__()
        # this also holds when using TP, since n_embd is the total embedding size and
        # n_head is the number of heads globally
        self.dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        self.base_freq = base_freq

        self.reset_parameters()

    def reset_parameters(self):
        # If previously initialized on or moved to a device, reuse that device.
        # Otherwise, use the default device of the current environment.
        device = self.inv_freq.device if hasattr(self, "inv_freq") else None
        inv_freq = 1.0 / (
            self.base_freq ** (torch.arange(0, self.dim_model, 2, device=device).float() / self.dim_model)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        """
        Rearange tentor elements.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update_cos_sin_tables(self, x):
        # Update the cosine and sine tables.
        seq_len = x.shape[self.seq_length_dim]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_length_dim], device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x, cos, sin):
        """
        Applies rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Cosine values for rotary positional embedding.
            sin (torch.Tensor): Sine values for rotary positional embedding.

        Returns:
            torch.Tensor: Tensor after applying rotary positional embedding.
        """
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[:, :, : x.shape[self.seq_length_dim], :]
        sin = sin[:, :, : x.shape[self.seq_length_dim], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RotaryTransform module.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Tuple containing the modified query tensor, key tensor, and value tensor.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        
        q = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return q, k, v


class QueryKeyValueTransformType(Enum):
    """
    Enum class representing different types of query-key-value transform.

    Attributes:
        IdentityTransform: Represents the identity transform.
        RotaryTransform: Represents the rotary transform.
    """

    IdentityTransform = IdentityTransform
    RotaryTransform = RotaryTransform


class AttentionImplementation(str, Enum):
    """
    Enum class representing different implementations of attention.

    Attributes:
        MANUAL (str): Manual attention implementation.
        PYTORCH_FLASH (str): PyTorch's flash attention implementation.
        DAO_FLASH (str): DAO's flash attention implementation.
    """

    MANUAL = "manual"
    PYTORCH_FLASH = "pytorch_flash"
    DAO_FLASH = "dao_flash"


class AttentionConfig(BaseModel):
    """
    Configuration class for attention mechanism.

    Attributes:
        qkv_transforms (list[QueryKeyValueTransformConfig]): List of configurations for query-key-value transforms.
    """

    class QueryKeyValueTransformConfig(BaseModel):
        """
        Configuration class for QueryKeyValueTransform.

        Attributes:
            type_hint (QueryKeyValueTransformType): The type hint for the transform.
            config (RotaryTransformConfig | IdentityTransformConfig): The configuration for the transform.
        """

        class IdentityTransformConfig(BaseModel):
            """IdentityTransformConfig class."""

            pass

        class RotaryTransformConfig(BaseModel):
            """
            Configuration class for RotaryTransform.

            Attributes:
                n_embd (int): Number of embeddings.
                n_head (int): Number of attention heads.
                seq_length_dim (int): Dimension of the sequence length.
                base_freq (int): Base frequency for RoPE.

            """

            n_embd: Annotated[int, Field(strict=True, ge=0)]
            n_head: Annotated[int, Field(strict=True, ge=0)]
            seq_length_dim: Annotated[int, Field(strict=True)]
            base_freq: Annotated[int, Field(strict=True, ge=10000)]

        @validator("type_hint", pre=True, always=True)
        def parse_sharding_strategy_by_name(cls, name):
            """
            Parses a QueryKeyValueTransform by its name.

            Args:
                name (str): The name of the sharding strategy.

            Returns:
                QueryKeyValueTransformType: The parsed sharding strategy.

            """
            return parse_enum_by_name(name=name, enum_type=QueryKeyValueTransformType)

        type_hint: QueryKeyValueTransformType
        config: RotaryTransformConfig | IdentityTransformConfig

    qkv_transforms: list[QueryKeyValueTransformConfig]
    qk_norm_config: Optional[LayerNormWrapperConfig] = None


class GPT2LLMConfig(BaseModel):
    """
    Configuration class for GPT2LLM model.

    Args:
        sample_key (str): The key for the samples.
        prediction_key (str): The key for the predictions.
        use_meta_device (bool, optional): Whether to use meta device. Defaults to False.
        poe_type (PositionTypes): The type of position encoding.
        sequence_length (int): The length of the sequence.
        vocab_size (int): The size of the vocabulary.
        n_layer (int): The number of layers.
        n_head_q (int): The number of attention heads for queries.
        n_head_kv (int): The number of attention heads for keys and values.
        n_embd (int): The embedding size.
        ffn_hidden (int): The hidden size of the feed-forward network.
        dropout (float): The dropout rate.
        bias (bool): Whether to use bias in Linears.
        attention_config (AttentionConfig): The attention configuration.
        attention_implementation (AttentionImplementation): The attention implementation.
        activation_type (ActivationType): The activation type.
        attention_norm_config (LayerNormWrapperConfig): Config for normalization of the attention.
        ffn_norm_config (LayerNormWrapperConfig): Config for normalization of the feed-forward network.
        lm_head_norm_config (LayerNormWrapperConfig): Config for normalization of the language model head.
        use_weight_tying (bool): Whether to use weight tying.
        seed: Optional[int] = None: The random seed for reproducibility.
        enforce_swiglu_hidden_dim_multiple_of (int): If specified, enforces the hidden dimension
            in the SwiGLU layer to be a multiple of this value. Note that this is only relevant if the
            activation_type is SwiGLU. Defaults to 256.
    """

    sample_key: str
    prediction_key: str
    use_meta_device: Optional[bool] = False
    poe_type: PositionTypes
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears like GPT-2. False: a bit better and faster
    attention_config: AttentionConfig
    attention_implementation: AttentionImplementation
    activation_type: ActivationType
    attention_norm_config: LayerNormWrapperConfig
    ffn_norm_config: LayerNormWrapperConfig
    lm_head_norm_config: LayerNormWrapperConfig
    use_weight_tying: bool
    recurrent_blocks_indices: Optional[Union[list[int], list[list[int]]]] = []
    recurrent_blocks_max_recurrences: Optional[Union[int, list[int]]] = 0
    use_recurrence_embedding: Optional[bool] = False
    recurrence_embedding_base_freq: Optional[float] = 10000.0
    seed: Optional[int] = None
    enforce_swiglu_hidden_dim_multiple_of: int = 256
    use_LNS: bool = False
    track_recurrence_embd_similarity: bool = False
    return_each_recurrence_output: bool = False
    return_each_recurrence_logits_entropy: bool = False
    separate_lm_head_norm: bool = False
    use_last_iteration_output_as_final: bool = True
    use_combined_representation: bool = False
    halt_threshold: Optional[float] = None
    gates_bias: Optional[List[float]] = None
    do_shifted_input: bool = True
    future_masking_prob: float = 0.0
    use_activation_checkpointing: bool = False
    aggregation_type: str = "WS"  # WS: weighted_sum, ATT: self-attention, ATT_CROSS: cross-attention, ATT_CROSS_EMBD: cross-attention with pre-MTP query
    use_per_iter_norms: bool = False  # When True, each recurrence iteration gets its own prev_iter_embd_norm inside GroupRecursiveGPT2MTPBlock
    use_latent_autoregressive: bool = False  # When True, step r>0 uses combined_output from step r-1 as input_embd instead of tokens_repres

    @model_validator(mode="after")
    def check_divisibility(self) -> "GPT2LLMConfig":
        """
        Check if the value of n_head_q is divisible by n_head_kv.

        Raises:
            ValueError: If n_head_q is not divisible by n_head_kv.

        Returns:
            GPT2LLMConfig: The current instance of GPT2LLMConfig.
        """
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError("n_head_q must be divisible by n_head_kv")
        return self

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        """
        Validates the sizes of the GPT2 model parameters.

        Returns:
            GPT2LLMConfig: The current instance of GPT2LLMConfig object.

        Raises:
            ValueError: If any of the parameters (ffn_hidden, vocab_size, n_embd) is not divisible by 128.
        """
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                # See https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


class CausalSelfAttention(nn.Module):
    """Causal Self Attention class."""

    def __init__(
        self,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        attention_config: AttentionConfig,
        attention_impl: AttentionImplementation,
        bias: bool,
        dropout: float,
    ):
        """
        Initializes the CausalSelfAttention object.

        Args:
            n_head_q (int): Number of attention heads for queries.
            n_head_kv (int): Number of attention heads for keys and values.
            n_embd (int): Size of the embedding dimension.
            attention_config (AttentionConfig): The attention configuration.
            attention_impl (AttentionImplementation): The attention implementation.
            bias (bool): Whether to include bias in linear layers.
            dropout (float): Dropout rate.

        Returns:
            None
        """
        super().__init__()
        assert n_embd % n_head_q == 0, "`n_embd needs` to be divisible by `n_head_q`."
        assert n_head_q % n_head_kv == 0, "`n_head_q needs` to be divisible by `n_head_kv`."

        self.n_rep = n_head_q // n_head_kv
        self.attention_impl = attention_impl

        # query, key, value projections (separate)
        self.q_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )
        self.k_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )
        self.v_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )

        # regularization
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv

        self.n_embd = n_embd
        # TODO: we might want different values for attention_dropout and linear_dropout
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(self.dropout)

        # TODO: inject QKVTransforms from outside
        self.qkv_transforms = nn.ModuleList(
            transform_config.type_hint.value(
                **convert_base_model_config_to_dict(transform_config.config)
            )  # TODO refactor, still uses the legacy type_hint
            for transform_config in attention_config.qkv_transforms
        )

        # QK Norm - helpful for models >1B to stabilize training
        # Baseline logits w/o qk norm: (Q @ K^T) / sqrt(d_h)
        # with geometric form of dot product: (||q_i|| * ||k_j|| * cos(θ_ij)) / sqrt(d_h)
        # so if the model wants to increase the distance between logits
        # it needs to scale q or k OR adjust the angle between them
        # qk norm forces the model to mostly adjust the angle between q and k which stabilizes training
        if attention_config.qk_norm_config is not None:
            self.q_norm = attention_config.qk_norm_config.norm_type.value(
                **dict(attention_config.qk_norm_config.config)
            )
            self.k_norm = attention_config.qk_norm_config.norm_type.value(
                **dict(attention_config.qk_norm_config.config)
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies projections to the input tensor to get queries, keys, and values.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the query, key, and value tensors.
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        return self.q_attn(x), self.k_attn(x), self.v_attn(x)

    @staticmethod
    def execute_qkv_transforms(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qkv_transforms: nn.ModuleList, n_head_q: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a series of transformations to the query, key, and value tensors.

        Args:
            q (torch.Tensor): The query tensors.
            k (torch.Tensor): The key tensors
            v (torch.Tensor): The value tensors.
            qkv_transforms (nn.ModuleList): A list of transformation modules to be applied to q, k, and v.
            n_head_q (int): The number of heads for the query tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the transformed query, key, and value tensors.
        """
        batch_size, sequence_length, embedding_dim = q.size()
        # hidden dimension of single head
        # Note, that number of heads does not change the overall parameters of the networks
        # to scale up the network we either have to increase the embedding_dim or the number of layers
        n_head_dim = embedding_dim // n_head_q

        q = q.view(batch_size, sequence_length, n_head_q, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_q, T, hd)
        k = k.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)
        v = v.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)

        for transform in qkv_transforms:
            q, k, v = transform(q, k, v)

        return q, k, v

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat the key-value tensor along the second dimension.

        Args:
            x (torch.Tensor): The input tensor of shape (B, nh_kv, T, hs).
            n_rep (int): The number of times to repeat the tensor along the second dimension.

        Returns:
            torch.Tensor: The repeated tensor of shape (B, nh_kv * n_rep, T, hs).

        Note:
            Source code adopted from
            https://github.com/facebookresearch/llama/blob/9a001c7a0987afd7b8de94e538916eff8950a73a/llama/model.py#L164
            Adapted ordered dimensions and namings: bs=B, n_kv_heads=nh_kv, slen=T, head_dim=hs
        """
        B, nh_kv, T, hs = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    @classmethod
    def repeat_kv_heads(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Repeats the key-value (k, v) heads if the number of query (q) heads is different.

        Args:
            cls (class): The class object.
            q (torch.Tensor): The query tensor of shape (B, nh_q, T, hs).
            k (torch.Tensor): The key tensor of shape (B, nh_kv, T, hs).
            v (torch.Tensor): The value tensor of shape (B, nh_kv, T, hs).

        Returns:
            tuple: A tuple containing the repeated key tensor (k) and the repeated value tensor (v).
        """
        # repeat k/v heads if self.n_rep > 1
        n_head_q = q.shape[1]
        n_head_kv = k.shape[1]
        if n_head_q != n_head_kv:
            n_rep = n_head_q // n_head_kv
            k = cls._repeat_kv(k, n_rep)  # (B, nh_q, T, hs)
            v = cls._repeat_kv(v, n_rep)  # (B, nh_q, T, hs)
        return k, v

    @classmethod
    def execute_attention(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: float,
        attention_impl: AttentionImplementation,
    ) -> torch.Tensor:
        """
        Executes attention mechanism based on the specified implementation.

        Args:
            cls (object): The class object.
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            dropout (float): The dropout rate.
            attention_impl (AttentionImplementation): The attention implementation to use.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            NotImplementedError: If the specified attention implementation is not supported.
        """
        if attention_impl == AttentionImplementation.MANUAL:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = manual_scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.PYTORCH_FLASH:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.DAO_FLASH:
            # Due to the lack of GPUs in github actions and the requirement of those in the flash-attn library,
            # we have to check if the library is installed and raise an error if not.
            # Note, that the library is not required for the CPU-only tests.
            if flash_attn_func is None:
                raise NotImplementedError("ERROR! Dao Flash Attention is not installed.")
            # the next three lines are only needed for flash-attn from Daio Lab
            q = q.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
            k = k.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            v = v.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            y = flash_attn_func(
                q, k, v, dropout_p=dropout, causal=True, softmax_scale=None, window_size=(-1, -1)
            )  # (B, T, nh_q, hd)
        else:
            raise NotImplementedError(f"Attention implementation {attention_impl} not supported")
        return y  # (B, T, nh_q, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CausalSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embd), representing the output projection.
        """
        B, T, _ = x.size()  # batch size (B), sequence length (T), embedding dimensionality (self.n_embd)
        q, k, v = self.projection(x)  # q: (B, T, n_embd), k: (B, T, n_embd // n_rep), v: (B, T, n_embd // n_rep)

        # q: (B, nh_q, T, hd), k: (B, nh_kv, T, hd), v: (B, nh_kv, T, hd)
        q, k, v = CausalSelfAttention.execute_qkv_transforms(q, k, v, self.qkv_transforms, self.n_head_q)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        y = CausalSelfAttention.execute_attention(q, k, v, self.dropout, self.attention_impl)  # (B, T, nh_q, hd)
        y = y.reshape(B, T, -1)  # (B, T, n_embd), re-assemble all head outputs side by side
        return self.resid_dropout(self.c_proj(y))  # (B, T, n_embd), output projection


class TransformerMLP(nn.Module):
    """TransformerMLP class."""

    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        """
        Initializes the TransformerMLP class.

        Args:
            n_embd (int): The size of the input embedding.
            ffn_hidden (int): The size of the hidden layer in the feed-forward network.
            bias (bool): Whether to include bias terms in the linear layers.
            dropout (float): The dropout probability.

        Returns:
            None
        """
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=n_embd,
            out_features=ffn_hidden,  # best practice: 4 * n_embd,
            bias=bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=ffn_hidden,
            out_features=n_embd,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerMLP module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SinusoidalRecurrenceEmbedding(nn.Module):
    def __init__(self, n_embd, base_freq=10000.0):
        super().__init__()
        MAX_ALLOWED_RECURRENCE = 32 # use a large value to cover all possible recurrences

        # Create the table of embeddings
        pe = torch.zeros(MAX_ALLOWED_RECURRENCE + 1, n_embd)
        position = torch.arange(0, MAX_ALLOWED_RECURRENCE + 1, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * 
                             (-math.log(base_freq) / n_embd))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer → NOT a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, recurrence_idx):
        """
        recurrence_idx: tensor of shape [batch, ...] with integer recurrence values
        """
        return self.pe[recurrence_idx]

class GPT2Block(nn.Module):
    """GPT2Block class."""

    def __init__(
        self,
        n_embd: int,
        bias: bool,
        n_head_q: int,
        n_head_kv: int,
        activation_type: ActivationType,
        attention_impl: AttentionImplementation,
        attention_config: AttentionConfig,
        dropout: float,
        ffn_hidden: int,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        enforce_swiglu_hidden_dim_multiple_of: int,
        lns_getter: Callable[[], nn.Module],
        increment_fn: Callable[[], None],
    ):
        """
        Initializes the GPT2Block.

        Args:
            n_embd (int): The embedding dimension.
            bias (bool): Whether to include bias in the model.
            n_head_q (int): The number of attention heads for queries.
            n_head_kv (int): The number of attention heads for keys and values.
            activation_type (ActivationType): The type of activation function to use.
            attention_impl (AttentionImplementation): The implementation of attention mechanism.
            attention_config (AttentionConfig): The configuration for attention mechanism.
            dropout (float): The dropout rate.
            ffn_hidden (int): The size of the hidden layer in the feed-forward network.
            attention_norm (nn.Module): The normalization layer for attention.
            ffn_norm (nn.Module): The normalization layer for feed-forward network.
            enforce_swiglu_hidden_dim_multiple_of (int): Enforces the
                hidden dimension in the SwiGLU layer to be a multiple of this value. Note that this
                is only relevant if the activation_type is SwiGLU. Defaults to None.
        """
        super().__init__()
        self.block_type = BlockTypes.STANDARD
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self._check_ffn_hidden_dim(n_embd=n_embd, ffn_hidden=ffn_hidden)
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_config=attention_config,
            attention_impl=attention_impl,
            bias=bias,
            dropout=dropout,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(
                n_embd=n_embd,
                ffn_hidden=ffn_hidden,
                bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
        else:
            raise NotImplementedError("unimplemented activation")
        
        self._lns_getter = lns_getter
        self._increment_fn = increment_fn

    def _check_ffn_hidden_dim(self, n_embd: int, ffn_hidden: int) -> None:
        expected_hidden_dim = 4 * n_embd

        if ffn_hidden != expected_hidden_dim:
            logger.warning(
                f"Expected `ffn_hidden` to be 4 * `n_embd` ({expected_hidden_dim}), "
                f"but got `n_embd = {n_embd}` and `ffn_hidden = {ffn_hidden}`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT2Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        x = self.attention_norm(x)

        scale_factor = self._lns_getter()
        x = scale_factor * x

        x = residual + self.attn(x)

        residual = x
        x = self.ffn_norm(x)

        scale_factor = self._lns_getter()
        x = scale_factor * x

        x = residual + self.mlp(x)
        self._increment_fn()
        return x  


class GroupRecursiveGPT2MTPBlock(nn.Module):
    """
    GroupRecursiveGPT2MTPBlock class.
    This class extends the functionality of a standard GPT2 block by packing multiple GPT2 blocks
    into a single recurrent block. 
    """
    def __init__(
        self,
        gpt2_blocks: list[GPT2Block],
        max_recurrence: int,
        n_embd: int,
        use_recurrence_embedding: bool = False,
        recurrence_embedding_base_freq: float = 10000.0,
        track_recurrence_embd_similarity: bool = False,
        return_each_recurrence_output: bool = False,
        use_combined_representation: bool = False,
        halt_threshold: Optional[float] = None,
        gates_bias: Optional[List[float]] = None,
        do_shifted_input: bool = True,
        future_masking_prob: float = 0.0,
        use_activation_checkpointing: bool = False,
        aggregation_type: str = "WS",  # WS: weighted_sum, ATT: self-attention, ATT_CROSS: cross-attention
        use_per_iter_norms: bool = False,  # When True, each recurrence iteration gets its own prev_iter_embd_norm
        use_latent_autoregressive: bool = False,  # When True, step r>0 uses combined_output from step r-1 as input_embd
    ):
        """
        Initializes the GroupRecursiveGPT2MTPBlock.

        Args:
            gpt2_blocks (list[GPT2Block]): List of GPT2Block instances to be packed into a single recurrent block.
            max_recurrence (int): The maximum number of recurrences.
            gates_bias (Optional[List[float]]): Optional list of biases for the gates. Defaults to None.
            do_shifted_input (bool): Whether to apply shifted input. Defaults to True.
            future_masking_prob (float): Probability of replacing each real shifted future token
                with its corresponding latent thought during training. Bridges the train-inference
                gap by teaching the model to work with latent surrogates. 0.0 = no masking (default),
                1.0 = always use latent thoughts (equivalent to inference behavior). Recommended:
                start at 0.0 and anneal up to ~0.5 during training.
            use_activation_checkpointing (bool): Whether to use gradient/activation checkpointing
                for each GPT2Block inside the recurrence loop, trading compute for memory. Defaults to False.
        Note:
            When using GroupRecursiveGPT2MTPBlock, the input tensor is feeded to the block max_recurrence (i.e. L) times. In other words,
            when max_recurrence=1, the GroupRecursiveGPT2MTPBlock behaves like a standard GPT2Block.
        """
        super().__init__()
        self.block_type = BlockTypes.GROUP_RECURSIVE_MTP
        self.gpt2_blocks = nn.ModuleList(gpt2_blocks)
        self.num_blocks = len(gpt2_blocks)
        self.max_recurrence = max_recurrence
        self.current_recurrence = 0
        self.use_recurrence_embedding = use_recurrence_embedding
        self.recurrence_embedding_base_freq = recurrence_embedding_base_freq
        if use_recurrence_embedding:
            self.recurrence_embd = SinusoidalRecurrenceEmbedding(n_embd, base_freq=recurrence_embedding_base_freq)
        
        self.track_recurrence_embd_similarity = track_recurrence_embd_similarity

        self.return_each_recurrence_output = return_each_recurrence_output # should be used only when iterating over the entire model
        self.use_per_iter_norms = use_per_iter_norms
        self.embd_norm = nn.LayerNorm(n_embd)
        norm_dim = n_embd + 1 if use_recurrence_embedding else n_embd
        if self.use_recurrence_embedding:
            self.proj = nn.Linear(n_embd*2 + 1, n_embd)
        else:
            self.proj = nn.Linear(n_embd*2, n_embd)
        if use_per_iter_norms:
            self.prev_iter_embd_norm = nn.ModuleList([nn.LayerNorm(norm_dim) for _ in range(max_recurrence)])
        else:
            self.prev_iter_embd_norm = nn.LayerNorm(norm_dim)

        self.use_combined_representation = use_combined_representation
        self.halt_threshold = halt_threshold
        self.aggregation_type = aggregation_type
        if self.use_combined_representation:
            self.combined_representation_block = CombinedRepresentationGPT2Block(
                n_embd=n_embd, 
                num_representations_max=self.max_recurrence,
                gates_bias=gates_bias,
                aggregation_type=aggregation_type
            )
            self.halt_block = HaltGPT2Block(n_embd=n_embd)
            
        # we will have at most "max_recurrence" latent thoughts which we want to learn
        self.latent_thoughts = nn.Parameter(torch.randn(max_recurrence-1, n_embd))
        self.do_shifted_input = do_shifted_input
        self.future_masking_prob = future_masking_prob
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_latent_autoregressive = use_latent_autoregressive

        # self._check_max_recurrence()

    def _recurrence_step(self, prev_iter_embd, input_embd, steps_done, step_idx: int = 0):
        """
        One recurrence step with or without gradient tracking.
        
        Args:
            prev_iter_embd: the output of the previous recurrence iteration
            input_embd: the input embedding to the current recurrence iteration (either the original input (x_{1:t}) or the output of the previous iteration based on use_latent_autoregressive)
        """
        if self.use_recurrence_embedding:
            normalized_steps_done = steps_done / self.max_recurrence
            steps_feature = normalized_steps_done.view(1, 1, 1).expand(
                prev_iter_embd.size(0),
                prev_iter_embd.size(1),
                1,
            )
            prev_iter_embd = torch.cat([steps_feature, prev_iter_embd], dim=-1).to(prev_iter_embd.dtype)

        if self.use_per_iter_norms:
            prev_iter_embd = self.prev_iter_embd_norm[step_idx](prev_iter_embd)
        else:
            prev_iter_embd = self.prev_iter_embd_norm(prev_iter_embd)
        
        if self.do_shifted_input:
            effective_steps = min(int(steps_done), input_embd.size(1))

            # embd length: seq_len - step_done
            embd = input_embd[:, effective_steps: , :] # batch, seq_len - steps_done, embd_dim
            # latent thought shape: steps_done, embd_dim --> batch, steps_done, embd_dim
            batch_size = embd.size(0)
            latent_thoughts = self.latent_thoughts[:effective_steps].unsqueeze(0).repeat(batch_size, 1, 1)

            # During training, randomly replace real shifted future tokens with a
            # learnable latent thought (the same surrogates the model sees at inference
            # for positions without real future context).  This teaches the model to
            # produce good representations regardless of whether it has access to the
            # actual future token or must rely on a latent thought.
            # At step r there are r available latent thoughts (indices 0..r-1),
            # so each masked position gets a randomly chosen one of those r surrogates.
            if self.training and self.future_masking_prob > 0.0 and steps_done > 0:
                num_real = embd.size(1)  # positions with real shifted tokens
                # Per-position Bernoulli mask: True → replace with a latent thought
                mask = torch.rand(batch_size, num_real, device=embd.device) < self.future_masking_prob
                # For each masked position, randomly pick one of the r available latent thoughts
                latent_indices = torch.randint(0, int(steps_done), (batch_size, num_real), device=embd.device)
                surrogates = self.latent_thoughts[latent_indices]  # (batch, num_real, n_embd)
                mask = mask.unsqueeze(-1)  # (batch, num_real, 1)
                embd = torch.where(mask, surrogates.to(embd.dtype), embd)

            embd = torch.cat([embd, latent_thoughts], dim=1).to(prev_iter_embd.dtype) # seq_len - steps_done + steps_done, embd_dim
            embd = self.embd_norm(embd)
        else:
            embd = self.embd_norm(input_embd)
        
        x = torch.cat([embd, prev_iter_embd], dim=-1).to(prev_iter_embd.dtype)
        x = self.proj(x)
        for block in self.gpt2_blocks:
            if self.use_activation_checkpointing and self.training:
                x = torch_checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        return x

    def forward(
        self, 
        x: torch.Tensor, 
        recurrence_step: Optional[int] = None, 
        recurrence_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the GroupRecursiveGPT2MTPBlock.

        Args:
            x (torch.Tensor): Input tensor.
            recurrence_step (optional, int): If provided, only this recurrence step is executed.
            recurrence_outputs (optional, list[torch.Tensor]): List of outputs from previous recurrence steps.

        Returns:
            torch.Tensor: Output tensor.
        """
        # ensure output type is the same as input type
        type_ = x.dtype
        
        if self.track_recurrence_embd_similarity:
            cosine_similarities = []
            mse_similarities = []
        
        if self.return_each_recurrence_output:
            all_recurrence_outputs = []

        self.current_recurrence = self.max_recurrence
        output = {}
        combined_output = None
        output['halt_signal'] = {r: 1 for r in range(self.current_recurrence)}
        ponder_regularization_losses = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        unhalted_prob = torch.ones(batch_size, seq_len, device=x.device)
        p_n_list = []
        pre_mtp_embd = x  # hidden state before any recurrence iteration; used as query in ATT_CROSS_EMBD
        for r in range(self.current_recurrence):
            x_before = x
            if self.use_latent_autoregressive and r > 0:
                current_input_embd = x_before * 0.01 + x_before.detach() * 0.99  # previous recurrence output (x_after_{r-1})
            else:
                current_input_embd = kwargs.get("tokens_repres")
            x_after = self._recurrence_step(
                prev_iter_embd=x,
                input_embd=current_input_embd,
                steps_done=torch.tensor(r, device=x.device),
                step_idx=r,
            )
            x = x_after

            if self.return_each_recurrence_output:
                all_recurrence_outputs.append(x)

            if self.track_recurrence_embd_similarity:
                # cosine similarity between x_before and x_after
                cosine_similarity = (nn.CosineSimilarity(dim=-1)(x_before.view(x_before.size(0), -1), x_after.view(x_after.size(0), -1)) + 1.0 ) / 2.0 # shift to [0, 1]
                cosine_similarities.append(cosine_similarity.mean())

                # mse 
                mse_similarity = nn.functional.mse_loss(x_before, x_after, reduction='mean')
                mse_similarities.append(mse_similarity)

            if self.use_combined_representation and r == self.current_recurrence - 1:  # for the time being, we don't train/use the halting block            
                output['recurrence_outputs'] = all_recurrence_outputs
                combined_output = self.combined_representation_block(
                    output,
                    combined_output,
                    pre_mtp_embd=pre_mtp_embd,
                )

                # # Halting probability (lambda_n)
                # lambda_n = self.halt_block(combined_output)

                # # # Force halt at last step
                # # if r == self.max_recurrence - 1:
                # #     lambda_n = torch.ones_like(lambda_n)

                # p_n = unhalted_prob * lambda_n
                
                # unhalted_prob = unhalted_prob * (1 - lambda_n)

                # p_n_list.append(p_n)

                # # break the loop if halting probability is above the threshold
                # # if not self.training and self.halt_threshold is not None:
                # #     if (lambda_n.mean() > self.halt_threshold or r == self.max_recurrence - 1):
                # #         output['halt_signal'][r] = lambda_n.mean().item()
                # #         break    

                
        if self.use_combined_representation:
            current_gates_normalized_tensor = torch.stack(
                self.combined_representation_block.current_gates_normalized, dim=1
            )  # [batch, steps, seq, dim]

            # Convert gate tensor to a halting distribution over steps by averaging over hidden dim.
            current_gates_normalized_tensor = current_gates_normalized_tensor.mean(dim=-1)  # [batch, steps, seq]
            current_gates_normalized_tensor = current_gates_normalized_tensor / (current_gates_normalized_tensor.sum(dim=1, keepdim=True) + 1e-8)

            b, current_steps, seq = current_gates_normalized_tensor.size()
            if current_steps < self.max_recurrence:
                pad = torch.zeros(
                    b,
                    self.max_recurrence - current_steps,
                    seq,
                    device=current_gates_normalized_tensor.device,
                    dtype=current_gates_normalized_tensor.dtype,
                )
                p_n_for_kl = torch.cat([current_gates_normalized_tensor, pad], dim=1)
            else:
                p_n_for_kl = current_gates_normalized_tensor[:, : self.max_recurrence, :]

            prior_dist = torch.ones(
                b,
                self.max_recurrence,
                seq,
                device=current_gates_normalized_tensor.device,
                dtype=current_gates_normalized_tensor.dtype,
            ) / self.max_recurrence

            # KL(P || Q) with P: model step distribution, Q: uniform prior over max recurrence.
            kl_loss = p_n_for_kl * (torch.log(p_n_for_kl + 1e-8) - torch.log(prior_dist + 1e-8))
            kl_loss = kl_loss.sum(dim=1).mean()  # sum over steps, mean over batch/seq
            ponder_regularization_losses.append(kl_loss)
                   
        
        if self.return_each_recurrence_output:
            output["recurrence_outputs"] = all_recurrence_outputs
        
        if self.track_recurrence_embd_similarity:
            output["cosine_similarity"] = torch.stack(cosine_similarities).mean()
            output["mse_similarity"] = torch.stack(mse_similarities).mean()

        if self.use_combined_representation:
            output["combined_output"] = combined_output.to(type_)
            output["pn_tensor"] = current_gates_normalized_tensor
            

        output["output"] = x.to(type_)
        
        # missing_halt_signals_cnt = self.current_recurrence - len(p_n_list)
        # for _ in range(missing_halt_signals_cnt):
        #     p_n_list.append(torch.ones(batch_size, seq_len, device=x.device))
        
        return output, ponder_regularization_losses, p_n_list

class CombinedRepresentationGPT2Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_representations_max: int,
        gates_bias: Optional[List[float]] = None,
        aggregation_type: str = "WS",  # WS: weighted_sum, ATT: self-attention, ATT_CROSS: cross-attention
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.num_representations = num_representations_max
        self.block_type = BlockTypes.COMBINED_REPRESENTATION
        self.aggregation_type = aggregation_type
        print(f"Using {aggregation_type} for combining representations in CombinedRepresentationGPT2Block")

        # --- WS-specific ---
        if aggregation_type == "WS":
            self.gate_layers = nn.ModuleList([
                nn.Linear(n_embd, n_embd) for _ in range(num_representations_max)
            ])
            if gates_bias is not None:
                assert len(gates_bias) == num_representations_max, \
                    "Length of gates_bias should be equal to num_representations_max"
                self.gate_bias = nn.Parameter(
                    torch.tensor(gates_bias).unsqueeze(1).repeat(1, n_embd)
                )
            else:
                self.gate_bias = nn.Parameter(torch.zeros(num_representations_max, n_embd))

        if aggregation_type in ("ATT", "ATT_CROSS", "ATT_CROSS_EMBD"):
            # --- ATT / ATT_CROSS / ATT_CROSS_EMBD shared ---
            self.attn = nn.MultiheadAttention(
                embed_dim=n_embd,
                num_heads=n_heads,
                batch_first=True,
            )

        if aggregation_type == "ATT_CROSS_EMBD":
            # Per-iteration LayerNorm applied to each ri before it enters as K/V.
            self.kv_norms = nn.ModuleList([
                nn.LayerNorm(n_embd) for _ in range(num_representations_max)
            ])

        # --- Shared gate storage (populated by both paths) ---
        self.current_gates = [torch.zeros(n_embd) for _ in range(num_representations_max)]
        self.current_gates_normalized = [torch.zeros(n_embd) for _ in range(num_representations_max)]

    def reset_current_gates(self):
        self.current_gates = [torch.zeros(self.n_embd) for _ in range(self.num_representations)]
        self.current_gates_normalized = [torch.zeros(self.n_embd) for _ in range(self.num_representations)]

    # ------------------------------------------------------------------
    # Aggregation paths
    # ------------------------------------------------------------------
    def _forward_ws(
        self,
        representations: list,
        pre_computed_combined_representation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        combined_h = torch.zeros_like(representations[0])

        for i, h in enumerate(representations):
            if (
                isinstance(pre_computed_combined_representation, (list, tuple))
                and i < len(pre_computed_combined_representation)
            ):
                continue
            gate = torch.sigmoid(self.gate_layers[i](h)) * torch.nn.functional.softplus(self.gate_bias[i])
            self.current_gates[i] = gate

        gates_stack = torch.stack(
            [self.current_gates[i] for i in range(len(representations))], dim=0
        )  # (num_repr, B, S, n_embd)
        gates_normalized = gates_stack / (gates_stack.sum(dim=0, keepdim=True) + 1e-8)
        self.current_gates_normalized = [gates_normalized[i] for i in range(len(representations))]

        for i, h in enumerate(representations):
            combined_h = combined_h + gates_normalized[i] * h

        return combined_h

    def _forward_att(self, representations: list) -> torch.Tensor:
        """Self-attention: all representations attend to each other; output taken from position 0."""
        if len(representations) == 1:
            return representations[0]

        B, S, D = representations[0].shape
        num_repr = len(representations)

        # (B, S, num_repr, D) → (B*S, num_repr, D)
        stacked = torch.stack(representations, dim=2)
        stacked_flat = stacked.view(B * S, num_repr, D)

        # attn_weights: (B*S, n_heads, num_repr, num_repr)
        out_flat, attn_weights = self.attn(
            stacked_flat, stacked_flat, stacked_flat,
            need_weights=True,
            average_attn_weights=False,
        )

        combined_h = out_flat[:, 0, :].view(B, S, D)          # (B, S, D)

        # Per-source weight: how much position-0 attended to each repr, averaged over heads.
        weights = attn_weights[:, :, 0, :].mean(dim=1).view(B, S, num_repr)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        self._store_gates(representations, weights, weights_norm, B, S, D, num_repr)
        return combined_h

    def _forward_att_cross(self, representations: list) -> torch.Tensor:
        """Cross-attention: repr-0 is the query; all representations are keys/values.

        This is more efficient (query length = 1) and semantically cleaner:
        repr-0 selectively pulls in information from deeper representations.
        """
        if len(representations) == 1:
            return representations[0]

        B, S, D = representations[0].shape
        num_repr = len(representations)

        # Query: repr-0 only → (B*S, 1, D)
        query_flat = representations[0].view(B * S, 1, D)

        # Keys/Values: all representations → (B*S, num_repr, D)
        stacked = torch.stack(representations, dim=2)          # (B, S, num_repr, D)
        kv_flat = stacked.view(B * S, num_repr, D)

        # attn_weights: (B*S, n_heads, 1, num_repr)
        out_flat, attn_weights = self.attn(
            query_flat, kv_flat, kv_flat,
            need_weights=True,
            average_attn_weights=False,
        )

        combined_h = out_flat[:, 0, :].view(B, S, D)          # (B, S, D)

        # Per-source weight: how much repr-0 attended to each repr, averaged over heads.
        weights = attn_weights[:, :, 0, :].mean(dim=1).view(B, S, num_repr)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        self._store_gates(representations, weights, weights_norm, B, S, D, num_repr)
        return combined_h

    def _store_gates(self, representations, weights, weights_norm, B, S, D, num_repr):
        """Populate current_gates / current_gates_normalized for downstream logging."""
        self.current_gates = [torch.zeros_like(representations[0]) for _ in range(self.num_representations)]
        self.current_gates_normalized = [torch.zeros_like(representations[0]) for _ in range(self.num_representations)]
        for i in range(num_repr):
            self.current_gates[i] = weights[:, :, i].unsqueeze(-1).expand(B, S, D)
            self.current_gates_normalized[i] = weights_norm[:, :, i].unsqueeze(-1).expand(B, S, D)

    @staticmethod
    def _sinusoidal_iter_pe(pos: int, d_model: int, device, dtype) -> torch.Tensor:
        """Fixed sinusoidal encoding for iteration index `pos`, shape (d_model,).

        Uses the standard sin/cos formula so no parameters are added:
            PE[2k]   = sin(pos / 10000^(2k/d_model))
            PE[2k+1] = cos(pos / 10000^(2k/d_model))
        """
        half = (d_model + 1) // 2  # ceil(d_model / 2)
        i = torch.arange(0, half, device=device, dtype=dtype)
        div = torch.exp(i * -(math.log(10000.0) / d_model))
        pe = torch.zeros(d_model, device=device, dtype=dtype)
        pe[0::2] = torch.sin(pos * div)           # indices 0, 2, 4, ... (half elements)
        pe[1::2] = torch.cos(pos * div)[:d_model // 2]  # indices 1, 3, 5, ... (floor(d/2) elements)
        return pe

    def _forward_att_cross_embd(self, representations: list, pre_mtp_embd: torch.Tensor) -> torch.Tensor:
        """Cross-attention: pre-MTP embedding is the query; per-iter-normed representations are K/V.

        Compared to ATT_CROSS (query=r1), the query here is the hidden state *before* any
        recurrence iteration, so the aggregation is fully decoupled from the recurrence outputs.
        Each ri is normalised by its own kv_norm and tagged with a sinusoidal iteration encoding
        (no extra parameters) so the attention can distinguish iteration depth.
        A residual from pre_mtp_embd is added to the output so the model can always fall back
        to the original representation if the recurrence outputs are not yet helpful.
        """
        B, S, D = representations[0].shape
        num_repr = len(representations)

        # Query: pre-MTP embedding → (B*S, 1, D)
        query_flat = pre_mtp_embd.view(B * S, 1, D)

        # Apply per-iteration norms + sinusoidal iteration encoding, then stack as K/V.
        # The PE distinguishes "iteration 0" from "iteration 1" etc. without learned params.
        normed = [
            self.kv_norms[i](representations[i])
            + self._sinusoidal_iter_pe(i, D, representations[i].device, representations[i].dtype)
            for i in range(num_repr)
        ]
        kv_flat = torch.stack(normed, dim=2).view(B * S, num_repr, D)  # (B*S, num_repr, D)

        # attn_weights: (B*S, n_heads, 1, num_repr)
        out_flat, attn_weights = self.attn(
            query_flat, kv_flat, kv_flat,
            need_weights=True,
            average_attn_weights=False,
        )

        combined_h = out_flat[:, 0, :].view(B, S, D)

        # Per-source weights averaged over heads → (B, S, num_repr)
        weights = attn_weights[:, :, 0, :].mean(dim=1).view(B, S, num_repr)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        self._store_gates(representations, weights, weights_norm, B, S, D, num_repr)
        return combined_h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Union[dict, torch.Tensor],
        pre_computed_combined_representation: Optional[torch.Tensor] = None,
        pre_mtp_embd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        representations = x["recurrence_outputs"] if isinstance(x, dict) else [x]

        if self.aggregation_type == "WS":
            return self._forward_ws(representations, pre_computed_combined_representation)
        elif self.aggregation_type == "ATT":
            return self._forward_att(representations)
        elif self.aggregation_type == "ATT_CROSS":
            return self._forward_att_cross(representations)
        elif self.aggregation_type == "ATT_CROSS_EMBD":
            assert pre_mtp_embd is not None, "pre_mtp_embd must be provided for ATT_CROSS_EMBD"
            return self._forward_att_cross_embd(representations, pre_mtp_embd)
        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type!r}. Choose 'WS', 'ATT', 'ATT_CROSS', or 'ATT_CROSS_EMBD'.")
    
class HaltGPT2Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
    ):
        super().__init__()
        self.block_type = BlockTypes.HALT
        
        # this layer outputs a scalar between 0 and 1 indicating whether to halt or not
        # for the given sequence
        self.halt_layer = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        halt_signal = torch.sigmoid(self.halt_layer(x)).squeeze(-1)  # shape: (batch_size, seq_len)
        return halt_signal


class GroupRecursiveGPT2Block(nn.Module):
    """
    GroupRecursiveGPT2Block class.
    This class extends the functionality of a standard GPT2 block by packing multiple GPT2 blocks
    into a single recurrent block. 
    """
    def __init__(
        self,
        gpt2_blocks: list[GPT2Block],
        max_recurrence: int,
        n_embd: int,
        use_recurrence_embedding: bool = False,
        recurrence_embedding_base_freq: float = 10000.0,
        track_recurrence_embd_similarity: bool = False,
        return_each_recurrence_output: bool = False,
    ):
        """
        Initializes the GroupRecursiveGPT2Block.

        Args:
            gpt2_blocks (list[GPT2Block]): List of GPT2Block instances to be packed into a single recurrent block.
            max_recurrence (int): The maximum number of recurrences.

        Note:
            When using GroupRecursiveGPT2Block, the input tensor is fed to the block max_recurrence (i.e. L) times. In other words,
            when max_recurrence=1, the GroupRecursiveGPT2Block behaves like a standard GPT2Block.
        """
        super().__init__()
        self.block_type = BlockTypes.GROUP_RECURSIVE
        self.gpt2_blocks = nn.ModuleList(gpt2_blocks)
        self.num_blocks = len(gpt2_blocks)
        self.max_recurrence = max_recurrence
        self.current_recurrence = 0
        self.use_recurrence_embedding = use_recurrence_embedding
        self.recurrence_embedding_base_freq = recurrence_embedding_base_freq
        if use_recurrence_embedding:
            self.recurrence_embd = SinusoidalRecurrenceEmbedding(n_embd, base_freq=recurrence_embedding_base_freq)
        
        self.track_recurrence_embd_similarity = track_recurrence_embd_similarity
        self.return_each_recurrence_output = return_each_recurrence_output


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the GroupRecursiveGPT2Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        def step(x, steps_done):
            """One recurrence step with or without gradient tracking."""
            if self.use_recurrence_embedding:
                recurrence_emb = self.recurrence_embd(steps_done)
                x = x + recurrence_emb

            for block in self.gpt2_blocks:
                x = block(x)
            
            return x
        
        # ensure output type is the same as input type
        type_ = x.dtype
        
        if self.track_recurrence_embd_similarity:
            cosine_similarities = []
            mse_similarities = []
        
        if self.return_each_recurrence_output:
            all_recurrence_outputs = []

        self.current_recurrence = self.max_recurrence
        for r in range(self.current_recurrence):
            x_before = x
            x_after = step(x, steps_done=torch.tensor(r, device=x.device))
            x = x_after

            if self.return_each_recurrence_output:
                all_recurrence_outputs.append(x)


            if self.track_recurrence_embd_similarity:
                # cosine similarity between x_before and x_after
                cosine_similarity = (nn.CosineSimilarity(dim=-1)(x_before.view(x_before.size(0), -1), x_after.view(x_after.size(0), -1)) + 1.0 ) / 2.0 # shift to [0, 1]
                cosine_similarities.append(cosine_similarity.mean())

                # mse 
                mse_similarity = nn.functional.mse_loss(x_before, x_after, reduction='mean')
                mse_similarities.append(mse_similarity)

        
        output = {}
        if self.return_each_recurrence_output:
            output["recurrence_outputs"] = all_recurrence_outputs
        
        if self.track_recurrence_embd_similarity:
            output["cosine_similarity"] = torch.stack(cosine_similarities).mean()
            output["mse_similarity"] = torch.stack(mse_similarities).mean()
            

        output["output"] = x.to(type_)

        return output


class GPT2LLM(NNModel):
    """GPT2LLM class."""

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        sequence_length: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        attention_implementation: AttentionImplementation,
        attention_config: AttentionConfig,
        attention_norm_config: LayerNormWrapperConfig,
        ffn_norm_config: LayerNormWrapperConfig,
        lm_head_norm_config: LayerNormWrapperConfig,
        use_weight_tying: bool,
        recurrent_blocks_indices: Union[list[int], list[list[int]]],
        recurrent_blocks_max_recurrences: Union[int, list[int]],
        use_recurrence_embedding: bool = False,
        recurrence_embedding_base_freq: float = 10000.0,
        seed: Optional[int] = None,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
        use_LNS: bool = False,
        track_recurrence_embd_similarity: bool = False,
        return_each_recurrence_output: bool = False,
        return_each_recurrence_logits_entropy: bool = False,
        separate_lm_head_norm: bool = False,
        use_last_iteration_output_as_final: bool = True,
        use_combined_representation: bool = False,
        halt_threshold: Optional[float] = 1.0,
        gates_bias: Optional[List[float]] = None,
        do_shifted_input: bool = True,
        future_masking_prob: float = 0.0,
        use_activation_checkpointing: bool = False,
        aggregation_type: str = "WS",  # WS: weighted_sum, ATT: self-attention, ATT_CROSS: cross-attention, ATT_CROSS_EMBD: cross-attention with pre-MTP query
        use_per_iter_norms: bool = False,  # When True, each recurrence iteration gets its own norm before the LM head
        use_latent_autoregressive: bool = False,  # When True, step r>0 uses combined_output from step r-1 as input_embd
    ):
        """
        Initializes the GPT2LLM object.

        Args:
            sample_key (str): The sample key.
            prediction_key (str): The prediction key.
            poe_type (PositionTypes): The position type.
            sequence_length (int): The sequence length.
            vocab_size (int): The vocabulary size.
            n_layer (int): The number of layers.
            n_head_q (int): The number of query heads.
            n_head_kv (int): The number of key-value heads.
            n_embd (int): The embedding dimension.
            ffn_hidden (int): The hidden dimension of the feed-forward network.
            dropout (float): The dropout rate.
            bias (bool): Whether to include bias in linear layers.
            activation_type (ActivationType): The activation type.
            attention_implementation (AttentionImplementation): The attention implementation.
            attention_config (AttentionConfig): The attention configuration.
            attention_norm_config (LayerNormWrapperConfig): Config for the attention normalization module.
            ffn_norm_config (LayerNormWrapperConfig): Config for the feed-forward network normalization module.
            lm_head_norm_config (LayerNormWrapperConfig): Config for the language model head normalization module.
            use_weight_tying (bool): Whether to use weight tying.
            recurrent_blocks_indices (list[int]): List of indices of the layers to be made recurrent.
            recurrent_blocks_max_recurrences (Union[int, list[int]]): Maximum number of recurrences for each recurrent block.
            seed (int, optional): The random seed. Defaults to None.
            enforce_swiglu_hidden_dim_multiple_of (int): Enforces
                the hidden dimension in the SwiGLU layer to be a multiple of this value.
                Note that this is only relevant if the activation_type is SwiGLU. Defaults to 256.
            do_shifted_input (bool): Whether to apply shifted input. Defaults to True.
        """
        weight_decay_groups = {
            "linear": [".attn", ".mlp", ".lm_head.weight", ".proj", ".gate_layers", ".halt_layer",],
            "embedding": [".wte", ".wpe", ".recurrence_embd", ".latent_thoughts", ".gate_bias",],
            "layernorm": [".attention_norm", ".ffn_norm", ".lm_head_norm", ".prev_iter_embd_norm", ".embd_norm", ".kv_norms"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.poe_type = poe_type
        self.recurrent_blocks_indices = recurrent_blocks_indices
        self.blocks_types = []
        self.use_recurrence_embedding = use_recurrence_embedding
        self.recurrence_embedding_base_freq = recurrence_embedding_base_freq
        self.recurrence_usage_stats = {}
        self.processed_layers_in_this_run = 0
        self.use_LNS = use_LNS
        self.track_recurrence_embd_similarity = track_recurrence_embd_similarity
        self.return_each_recurrence_output = return_each_recurrence_output
        self.return_each_recurrence_logits_entropy = return_each_recurrence_logits_entropy
        self.separate_lm_head_norm = separate_lm_head_norm
        self.use_last_iteration_output_as_final = use_last_iteration_output_as_final
        self.use_combined_representation = use_combined_representation
        self.halt_threshold = halt_threshold
        self.gates_bias = gates_bias
        self.do_shifted_input = do_shifted_input
        self.future_masking_prob = future_masking_prob
        self.use_activation_checkpointing = use_activation_checkpointing
        self.aggregation_type = aggregation_type
        self.use_per_iter_norms = use_per_iter_norms
        self.use_latent_autoregressive = use_latent_autoregressive

        if return_each_recurrence_logits_entropy:
            self.recurrence_logits_entropy_stats = {}

        if track_recurrence_embd_similarity:
            self.recurrence_embedding_cosine_similarity_stats = {}
            self.recurrence_embedding_mse_similarity_stats = {} 

        self.vocab_size = vocab_size


        assert vocab_size is not None
        assert sequence_length is not None

        if isinstance(recurrent_blocks_max_recurrences, int):
            self.max_recurrences = [recurrent_blocks_max_recurrences] * len(recurrent_blocks_indices)
        else:
            self.max_recurrences = recurrent_blocks_max_recurrences

        self._check_max_recurrences()

        if use_combined_representation:
            self.halt_value_stats = {}
            self.gate_stats = {f"gate_{i}": {} for i in range(max(self.max_recurrences))}
            self.gate_normalized_stats = {f"normalized_gate_{i}": {} for i in range(max(self.max_recurrences))}

        # TODO: dependency injection
        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=sequence_length, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            # Using a pre-trained layer, requires to define a separate FSDP unit for the frozen layer c.f.
            # https://github.com/huggingface/accelerate/issues/807
            # wpe = nn.Embedding.from_pretrained(torch.zeros(sequence_length, n_embd))
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention_config.qkv_transforms
        ]:
            raise ValueError('It is expected to use "RotaryTransform" together with "NOPE".')


        blocks_list = []
        recurrent_blocks_cnt = 0
        n = 0
        while n < n_layer:
            # determine block type
            block_type = self._determine_block_type(n)
            print(f"Building block {n} of type {block_type}")            
            if block_type == BlockTypes.STANDARD:
                num_blocks = 1
                block = GPT2Block(
                    n_embd=n_embd,
                    bias=bias,
                    n_head_q=n_head_q,
                    n_head_kv=n_head_kv,
                    activation_type=activation_type,
                    attention_impl=attention_implementation,
                    attention_config=attention_config,
                    dropout=dropout,
                    ffn_hidden=ffn_hidden,
                    # deepcopy did not work here! The weights were then automatically
                    # moved to a cuda device even when the deepcopied weights were on
                    # a meta device!
                    attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                    ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                    enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
                    lns_getter=self.get_LNS_factor,
                    increment_fn=self.increment_processed_layers_cnt,
                )
            elif block_type == BlockTypes.GROUP_RECURSIVE or block_type == BlockTypes.GROUP_RECURSIVE_MTP:
                max_recurrence = self._determine_max_recurrence(block_type, recurrent_blocks_cnt)
                num_blocks = self._determine_num_blocks_in_group(n)
                gpt2_blocks = []
                for i in range(num_blocks):
                    gpt2_block = GPT2Block(
                        n_embd=n_embd,
                        bias=bias,
                        n_head_q=n_head_q,
                        n_head_kv=n_head_kv,
                        activation_type=activation_type,
                        attention_impl=attention_implementation,
                        attention_config=attention_config,
                        dropout=dropout,
                        ffn_hidden=ffn_hidden,
                        attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                        ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                        enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
                        lns_getter=self.get_LNS_factor,
                        increment_fn=self.increment_processed_layers_cnt,
                    )
                    gpt2_blocks.append(gpt2_block)
                
                block_arguments = dict(
                    gpt2_blocks=gpt2_blocks,
                    max_recurrence=max_recurrence,
                    n_embd=n_embd,
                    use_recurrence_embedding=use_recurrence_embedding,
                    recurrence_embedding_base_freq=recurrence_embedding_base_freq,
                    track_recurrence_embd_similarity=track_recurrence_embd_similarity,
                    return_each_recurrence_output=return_each_recurrence_output
                )

                if block_type == BlockTypes.GROUP_RECURSIVE:
                    block = GroupRecursiveGPT2Block(**block_arguments)
                else:  # GROUP_RECURSIVE_MTP
                    block = GroupRecursiveGPT2MTPBlock(
                        use_combined_representation=use_combined_representation,
                        halt_threshold=halt_threshold,
                        gates_bias=gates_bias,
                        do_shifted_input=do_shifted_input,
                        future_masking_prob=future_masking_prob,
                        use_activation_checkpointing=use_activation_checkpointing,
                        aggregation_type=aggregation_type,
                        use_per_iter_norms=use_per_iter_norms,
                        use_latent_autoregressive=use_latent_autoregressive,
                        **block_arguments
                    )
            elif block_type in [BlockTypes.COMBINED_REPRESENTATION, BlockTypes.HALT]:
                pass
            else:
                raise ValueError(
                    f"Block type {block_type} not supported! "
                    f"Supported block types are: {list(BlockTypes)}"
                )
            blocks_list.append(block)
            self.blocks_types.append(block_type)
            self.recurrent_blocks_indices.append(len(blocks_list) - 1)
            n += num_blocks
            if block_type == BlockTypes.RECURSIVE or block_type == BlockTypes.GROUP_RECURSIVE or block_type == BlockTypes.GROUP_RECURSIVE_MTP:
                recurrent_blocks_cnt += 1

        assert n == n_layer, f"Expected {n_layer} blocks, but got {n}!"

        if self.separate_lm_head_norm:
            # for each iteration, we want to have a separate norm layer for the lm head
            lm_head_norms = []
            for _ in range(self.max_recurrences[-1]):
                lm_head_norms.append(
                    lm_head_norm_config.norm_type.value(**dict(lm_head_norm_config.config))
                )
            lm_head_norms = nn.ModuleList(lm_head_norms)
        else:
            lm_head_norms = lm_head_norm_config.norm_type.value(**dict(lm_head_norm_config.config))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=wpe,
                drop=nn.Dropout(dropout),
                h=nn.ModuleDict({str(layer_idx): blocks_list[layer_idx] for layer_idx in range(len(blocks_list))}),
                lm_head_norm=lm_head_norms,
                # NOTE: If we make the bias configurable, we must update the number of parameters calculation
                # in the test_initialization_fsdp1.py, accordingly.
                lm_head=nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False),
            )
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if use_weight_tying:
            self.transformer.wte.weight = (
                self.transformer.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

    def _check_max_recurrences(self) -> None:
        if self.blocks_types is not None:
            if BlockTypes.RECURSIVE in self.blocks_types:
                if self.max_recurrences is None:
                    raise ValueError(
                        "When using RecursiveGPT2Block, 'max_recurrences' must be provided."
                    )
                if len(self.max_recurrences) != len(self.recurrent_blocks_indices):
                    raise ValueError(
                        f"The length of max_recurrences ({len(self.max_recurrences)}) must be equal to the length of "
                        f"recurrent_blocks_indices ({len(self.recurrent_blocks_indices)})."
                    )
                for recurrence_block_index in self.recurrent_blocks_indices:
                    if recurrence_block_index >= self.n_layer or recurrence_block_index < 0:
                        raise ValueError(
                            f"All values in 'recurrent_blocks_indices' must be (including) between 0 and n_layer-1 ({self.n_layer-1})."
                        )
                    
    def _determine_block_type(self, layer_index: int) -> BlockTypes:
        if layer_index in self.recurrent_blocks_indices:
            return BlockTypes.RECURSIVE
        
        for group in self.recurrent_blocks_indices:
            if isinstance(group, list): 
                if layer_index == group[0]:
                    return BlockTypes.GROUP_RECURSIVE
                if -layer_index == group[0]:
                    return BlockTypes.GROUP_RECURSIVE_MTP
        
        return BlockTypes.STANDARD
    
    def _determine_num_blocks_in_group(self, layer_index: int) -> int:
        for group in self.recurrent_blocks_indices:
            # use abs() because group[0] can be negative for GROUP_RECURSIVE_MTP
            if isinstance(group, list) and layer_index == abs(group[0]):
                return len(group)
        raise ValueError(f"Layer index {layer_index} is not part of any group in recurrent_blocks_indices.")
        
    def _determine_max_recurrence(self, block_type: BlockTypes, recurrent_blocks_cnt: int) -> int:
        if block_type in [BlockTypes.RECURSIVE, BlockTypes.GROUP_RECURSIVE, BlockTypes.GROUP_RECURSIVE_MTP]:
            return self.max_recurrences[recurrent_blocks_cnt]
        else:
            return 0  # not used for standard blocks, but needed to create the block

    def increment_processed_layers_cnt(self):
        """Increments the count of processed layers in this run."""
        self.processed_layers_in_this_run += 1

    def get_processed_layers_cnt(self) -> int:
        """Returns the count of processed layers in this run."""
        return self.processed_layers_in_this_run

    def reset_processed_layers_cnt(self):
        """Resets the count of processed layers in this run to zero."""
        self.processed_layers_in_this_run = 0    

    def get_LNS_factor(self) -> float:
        """
        Returns the LNS factor based on the number of processed layers.
        """
        if not self.use_LNS:
            return 1.0 # no scaling
        # return 1.0 / math.sqrt(self.get_processed_layers_cnt() + 1)
        return  1.0 / (self.get_processed_layers_cnt() + 1)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Load the state dictionary into the model. Forwards to nn.Module.load_state_dict().
        Used for backward compatibility with the old state dict format.
        """
        if "lm_head.weight" in state_dict:
            state_dict["transformer.lm_head.weight"] = state_dict["lm_head.weight"]
            del state_dict["lm_head.weight"]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def record_recurrence_usage_stats(self, layer_idx, block):
        if int(layer_idx) not in self.recurrence_usage_stats:
            self.recurrence_usage_stats[int(layer_idx)] = []
        self.recurrence_usage_stats[int(layer_idx)].append(block.current_recurrence)

    def record_recurrence_embedding_similarity_stats(self, cosine_similarity, mse_similarity, layer_idx):
        if int(layer_idx) not in self.recurrence_embedding_cosine_similarity_stats:
            self.recurrence_embedding_cosine_similarity_stats[int(layer_idx)] = []
        self.recurrence_embedding_cosine_similarity_stats[int(layer_idx)].append(cosine_similarity.item())

        if int(layer_idx) not in self.recurrence_embedding_mse_similarity_stats:
            self.recurrence_embedding_mse_similarity_stats[int(layer_idx)] = []
        self.recurrence_embedding_mse_similarity_stats[int(layer_idx)].append(mse_similarity.item())

    def record_recurrence_logits_entropy_stats(self, iter_idx, entropy):
        if int(iter_idx) not in self.recurrence_logits_entropy_stats:
            self.recurrence_logits_entropy_stats[int(iter_idx)] = []
        
        self.recurrence_logits_entropy_stats[int(iter_idx)].append(entropy.item())

    def record_halt_signal_stats(self, halt_signals):
        if not self.training:
            return
        for iter_idx, halt_signal in enumerate(halt_signals):
            if int(iter_idx) not in self.halt_value_stats:
                self.halt_value_stats[int(iter_idx)] = []
            
            # Detach to prevent pinning the entire autograd graph in memory
            if isinstance(halt_signal, torch.Tensor):
                halt_signal = halt_signal.detach()
            self.halt_value_stats[int(iter_idx)].append(halt_signal)

    @staticmethod
    def _get_gate_mean_for_logging(gate_value: Any) -> torch.Tensor:
        """Return a scalar tensor mean for a gate value used in logging."""
        if gate_value is None:
            return torch.tensor(0.0)
        if isinstance(gate_value, torch.Tensor):
            return gate_value.float().mean()
        return torch.tensor(float(gate_value))

    def _normalize_gate_values_for_logging(
        self,
        gate_values: Union[dict[int, Any], list[Any]],
        max_iters: int,
    ) -> list[torch.Tensor]:
        """
        Normalize scalar gate means across iterations.

        This preserves ordering of logged gate means (if gate_i > gate_j, then norm_i > norm_j)
        for the same logging step.
        """
        gate_means = []
        for iter_idx in range(max_iters):
            value = None
            if isinstance(gate_values, dict) and iter_idx in gate_values:
                value = gate_values[iter_idx]
            elif isinstance(gate_values, list) and iter_idx < len(gate_values):
                value = gate_values[iter_idx]
            gate_means.append(self._get_gate_mean_for_logging(value))

        gates_tensor = torch.stack(gate_means)
        gates_tensor = gates_tensor / (gates_tensor.sum() + 1e-8)
        return [gates_tensor[i] for i in range(max_iters)]

    def record_gate_stats(self, gate_values_dict, layer_idx, max_iters):
        if not self.training:
            return
        
        for iter_idx in range(max_iters):
            gate_val = 0.0
            if isinstance(gate_values_dict, dict) and iter_idx in gate_values_dict:
                gate_val = self._get_gate_mean_for_logging(gate_values_dict[iter_idx]).item()
            elif isinstance(gate_values_dict, list) and iter_idx < len(gate_values_dict):
                gate_val = self._get_gate_mean_for_logging(gate_values_dict[iter_idx]).item()
            
            gate_key = f"gate_{iter_idx}"
            if gate_key not in self.gate_stats:
                 self.gate_stats[gate_key] = {}
            
            if int(layer_idx) not in self.gate_stats[gate_key]:
                self.gate_stats[gate_key][int(layer_idx)] = {}

            if int(iter_idx) not in self.gate_stats[gate_key][int(layer_idx)]:
                self.gate_stats[gate_key][int(layer_idx)][int(iter_idx)] = []

            self.gate_stats[gate_key][int(layer_idx)][int(iter_idx)].append(gate_val)
    
    def record_gate_normalized_stats(self, gate_values_normalized_dict, layer_idx, max_iters):
        if not self.training:
            return
        
        for iter_idx in range(max_iters):
            gate_val = 0.0
            if isinstance(gate_values_normalized_dict, dict) and iter_idx in gate_values_normalized_dict:
                gate_val = self._get_gate_mean_for_logging(gate_values_normalized_dict[iter_idx]).item()
            elif isinstance(gate_values_normalized_dict, list) and iter_idx < len(gate_values_normalized_dict):
                gate_val = self._get_gate_mean_for_logging(gate_values_normalized_dict[iter_idx]).item()
            
            gate_key = f"gate_{iter_idx}_normalized"
            if gate_key not in self.gate_normalized_stats:
                 self.gate_normalized_stats[gate_key] = {}
            
            if int(layer_idx) not in self.gate_normalized_stats[gate_key]:
                self.gate_normalized_stats[gate_key][int(layer_idx)] = {}
            
            if int(iter_idx) not in self.gate_normalized_stats[gate_key][int(layer_idx)]:
                self.gate_normalized_stats[gate_key][int(layer_idx)][int(iter_idx)] = []
            
            self.gate_normalized_stats[gate_key][int(layer_idx)][int(iter_idx)].append(gate_val)    
            
    
    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the GPT2LLM module.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.
                - sample_key (str): Key for the input tensor containing token ids.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing output tensors.
                - prediction_key (str): Key for the output tensor containing logits.
        """
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT2LLM module.

        Args:
            inputs (torch.Tensor): A tensor containing input token ids.

        Returns:
            torch.Tensor: A tensor containing output logits.
        """
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor, sampling_std: float = None) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass of the GPT2LLM module.

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): Input data.

        Returns:
            dict[str, torch.Tensor] | torch.Tensor: Model output.
        """
        if isinstance(inputs, dict):
            return {self.prediction_key: self.forward_impl(inputs[self.sample_key], sampling_std=sampling_std)}
        else:
            return self.forward_impl(inputs, sampling_std=sampling_std)

    def forward_impl(self, inputs: torch.Tensor, sampling_std=None) -> torch.Tensor:
        """
        Forward pass implementation of the GPT2LLM module.

        Args:
            inputs (torch.Tensor): A tensor containing input token ids.

        Returns:
            torch.Tensor: A tensor containing output logits.
        """
        if sampling_std is not None:
            pass
        device = inputs.device
        seq_len = inputs.size(1)
        assert seq_len <= self.sequence_length, f"Cannot forward sequence of length {seq_len}, the model's maximum "
        f"input sequence length is only {self.sequence_length}."

        # forward the GPT model itself
        h = (
            self.transformer.wte(inputs) if hasattr(self.transformer, "wte") else inputs
        )  # token embeddings of shape (b, seq_len, n_embd)

        if self.poe_type is PositionTypes.ABSOLUTE and hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)  # shape (seq_len)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (seq_len, n_embd)
            h = h + pos_emb

        # TODO: use drop out also without absolute position embedding?
        h = self.transformer.drop(h) if hasattr(self.transformer, "drop") else h

        tokens_repres = h # shape: (b, seq_len, n_embd)

        recurrence_cosine_similarities = []
        recurrence_mse_similarities = []
        ponder_regularization_losses = []
        combined_output = None
        for layer_idx in self.transformer.h:
            block_type = self.blocks_types[int(layer_idx)]
                
            if block_type in [BlockTypes.GROUP_RECURSIVE, BlockTypes.GROUP_RECURSIVE_MTP]:
                block: Union[GroupRecursiveGPT2Block, GroupRecursiveGPT2MTPBlock] = self.transformer.h[layer_idx]  # type: ignore

                output = {}
                if block_type == BlockTypes.GROUP_RECURSIVE_MTP:
                    seq_len = h.size(1)
                    output, ponder_regularization_losses, p_n_list = block(
                        h,
                        tokens_repres=tokens_repres
                    )

                    if self.use_combined_representation:
                        h = output["combined_output"]
                    else:
                        h = output["output"]

                    self.record_gate_stats(block.combined_representation_block.current_gates, layer_idx, block.max_recurrence)
                    
                    # For logging, normalize scalar gate means across iterations so trends remain comparable.
                    normalized_gates_for_logging = self._normalize_gate_values_for_logging(
                        block.combined_representation_block.current_gates,
                        block.max_recurrence,
                    )
                    self.record_gate_normalized_stats(normalized_gates_for_logging, layer_idx, block.max_recurrence)

                    self.record_halt_signal_stats(p_n_list)
                    # Detach gate references to avoid pinning autograd graph in memory
                    current_gates = [g.detach() if isinstance(g, torch.Tensor) else g for g in block.combined_representation_block.current_gates]
                    current_gates_normalized = [g.detach() if isinstance(g, torch.Tensor) else g for g in block.combined_representation_block.current_gates_normalized]
                else:
                    output = block(h, tokens_repres=tokens_repres)
                    h = output["output"]
                
                if self.track_recurrence_embd_similarity:
                    cosine_similarity = output["cosine_similarity"]
                    mse_similarity = output["mse_similarity"]
                    recurrence_cosine_similarities.append(cosine_similarity)
                    recurrence_mse_similarities.append(mse_similarity)
                if self.return_each_recurrence_output:
                    each_recurrence_outputs = output["recurrence_outputs"]
            else:
                before_h = h
                h = self.transformer.h[layer_idx](h)
                
                cosine_similarity = nn.functional.cosine_similarity(
                    before_h.view(before_h.size(0), -1), h.view(h.size(0), -1), dim=-1
                ).mean()
                with torch.no_grad():
                    mse_similarity = nn.functional.mse_loss(before_h, h, reduction='mean').mean()

            if block_type in [BlockTypes.GROUP_RECURSIVE, BlockTypes.GROUP_RECURSIVE_MTP]:
                block: Union[GroupRecursiveGPT2Block, GroupRecursiveGPT2MTPBlock] = self.transformer.h[layer_idx]  # type: ignore
                if self.training:
                    self.record_recurrence_usage_stats(layer_idx, block)
            
            if self.track_recurrence_embd_similarity:
                if self.training:
                    self.record_recurrence_embedding_similarity_stats(cosine_similarity, mse_similarity, layer_idx)

        if self.separate_lm_head_norm:
            # this representation is given by the last iteration of the recurrent block
            h = self.transformer.lm_head_norm[-1](h)
        else:
            h = self.transformer.lm_head_norm(h) if hasattr(self.transformer, "lm_head_norm") else h
        h = self.transformer.lm_head(h) if hasattr(self.transformer, "lm_head") else h
        self.reset_processed_layers_cnt()
        
        final_output = {}
        if ponder_regularization_losses:
            final_output["ponder_regularization_loss"] = torch.stack(ponder_regularization_losses).mean()

        if self.track_recurrence_embd_similarity:
            final_output["recurrence_embedding_cosine_similarity"] = torch.stack(recurrence_cosine_similarities).mean() if recurrence_cosine_similarities else torch.tensor(0.0)
            final_output["recurrence_embedding_mse_similarity"] = torch.stack(recurrence_mse_similarities).mean() if recurrence_mse_similarities else torch.tensor(0.0)
        if self.return_each_recurrence_output:
            final_output["each_recurrence_logits"] = []
            if self.return_each_recurrence_logits_entropy:
                final_output["each_recurrence_entropy"] = []

            if self.use_per_iter_norms:
                # ATT_CROSS_EMBD layout:
                #   [0] = combined_output logits  (shared lm_head_norm, already in h)
                #   [1] = lm_head_norm(r1) → lm_head  (r1 aligned to t+1, auxiliary)
                #   [2] = lm_head_norm(r2) → lm_head  (r2 aligned to t+2)
                #   ...
                # Per-iteration input normalization is handled inside GroupRecursiveGPT2MTPBlock
                # via per-iteration prev_iter_embd_norms; the shared lm_head_norm is used here.
                final_output["each_recurrence_logits"].append(h)  # combined first
                if self.return_each_recurrence_logits_entropy:
                    entropy = get_logits_entropy(h)
                    final_output["each_recurrence_entropy"].append(entropy)
                    if self.training:
                        self.record_recurrence_logits_entropy_stats(self.max_recurrences[-1]-1, entropy)

                for r in range(len(each_recurrence_outputs)):
                    o = self.transformer.lm_head_norm(each_recurrence_outputs[r]) if hasattr(self.transformer, "lm_head_norm") else each_recurrence_outputs[r]
                    o = self.transformer.lm_head(o) if hasattr(self.transformer, "lm_head") else o
                    final_output["each_recurrence_logits"].append(o)

                    if self.return_each_recurrence_logits_entropy:
                        entropy = get_logits_entropy(o)
                        final_output["each_recurrence_entropy"].append(entropy)
                        if self.training:
                            self.record_recurrence_logits_entropy_stats(r, entropy.detach())
            else:
                # Existing layout:
                #   [0..K-2] = r1..r_{K-1} logits
                #   [K-1]    = combined_output logits (appended last)
                for r in range(len(each_recurrence_outputs)-1):
                    if self.separate_lm_head_norm:
                        o = self.transformer.lm_head_norm[r](each_recurrence_outputs[r])
                    else:
                        o = self.transformer.lm_head_norm(each_recurrence_outputs[r]) if hasattr(self.transformer, "lm_head_norm") else each_recurrence_outputs[r]
                    o = self.transformer.lm_head(o) if hasattr(self.transformer, "lm_head") else o
                    final_output["each_recurrence_logits"].append(o)

                    if self.return_each_recurrence_logits_entropy:
                        entropy = get_logits_entropy(o)
                        final_output["each_recurrence_entropy"].append(entropy)
                        if self.training:
                            self.record_recurrence_logits_entropy_stats(r, entropy.detach())

                final_output["each_recurrence_logits"].append(h)  # add the final output as well
                if self.return_each_recurrence_logits_entropy:
                    entropy = get_logits_entropy(h)
                    final_output["each_recurrence_entropy"].append(entropy)  # add the final output as well
                    if self.training:
                        self.record_recurrence_logits_entropy_stats(self.max_recurrences[-1]-1, entropy)

            final_output["each_recurrence_logits"] = torch.stack(final_output["each_recurrence_logits"])
            if self.return_each_recurrence_logits_entropy:
                final_output["each_recurrence_entropy"] = torch.stack(final_output["each_recurrence_entropy"])
        
        final_output["gates"] = current_gates if self.use_combined_representation else None
        final_output["gates_normalized"] = current_gates_normalized if self.use_combined_representation else None
        
        if self.use_combined_representation:
            final_output["pn_tensor"] = output["pn_tensor"] 
            
        if self.use_last_iteration_output_as_final:
            final_output["logits"] = h
        else:
            final_output["logits"] = final_output["each_recurrence_logits"][0] 
        return final_output if len(final_output) > 1 else h
    

def get_logits_entropy(logits: torch.Tensor):
    """
    Compute Shannon entropy of the logits.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)  # add small value to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    normalized_entropy = entropy / torch.log(torch.tensor(logits.size(-1), dtype=logits.dtype, device=logits.device))
    normalized_entropy = normalized_entropy.mean()  # mean over batch and sequence length
    return normalized_entropy

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """
    Compute scaled dot product attention.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, num_queries, query_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_keys, key_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_values, value_dim).
        attn_mask (torch.Tensor, optional): The attention mask tensor of shape (num_queries, num_keys).
            Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether the attention is causal or not. Defaults to False.
        scale (float, optional): The scaling factor. Defaults to None.

    Returns:
        torch.Tensor: The attention weights tensor of shape (batch_size, num_queries, num_keys).

    Note:
        Taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(
        L, S, dtype=query.dtype, device=query.device
    )  # device added (not part of the original code)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)  # device added
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
