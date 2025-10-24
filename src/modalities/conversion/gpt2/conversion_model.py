from typing import Union
import torch
import torch.nn as nn
from tqdm import tqdm
from enum import Enum
import copy

from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM, RecursiveGPT2DecoderLayer, GroupRecursiveGPT2DecoderLayer
from modalities.models.components.layer_norms import LayerNormConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block, PositionTypes
from modalities.models.model import SwiGLU
from modalities.models.utils import ModelTypeEnum, get_model_from_config


def convert_model_checkpoint(modalities_config: dict) -> tuple[GPT2ForCausalLM, GPT2LLM]:
    """Converts the modalities model to a Huggingface transformers model.
       Both the loaded modalities model and the converted Huggingface model are returned
       so that they can be compared.

    Args:
        modalities_config (dict): Modalities config dictionary.

    Returns:
        tuple[GPT2ForCausalLM, GPT2LLM]: Converted Hugging Face model and the original modalities model.
    """
    gpt2_config = convert_model_config(modalities_config)
    hf_model = GPT2ForCausalLM(gpt2_config).to(dtype=torch.bfloat16).to("cpu")
    modalities_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL).to("cpu")
    
    _copy_weights_model(hf_model, modalities_model)
    return hf_model, modalities_model


def convert_model_config(modalities_config: dict) -> GPT2Config:
    """Converts the modalities model configuration to a Huggingface transformers configuration.
       For this the model_raw or model section of the modalities config is used.
       Corresponding entries are mapped to the Huggingface configuration.

    Args:
        modalities_config (dict): Modalities config dictionary.

    Returns:
        GPT2Config: Converted Huggingface model configuration.
    """
    config = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]
    _check_conversion_criteria(config)

    ffn_norm_key = "ffn_norm" if "ffn_norm" in config else "ffn_norm_config"
    blocks_types = generate_blocks_types(config)
    return GPT2Config(
        vocab_size=config["vocab_size"],
        hidden_size=config["n_embd"],
        pad_token_id=None,
        num_hidden_layers=config["n_layer"],
        num_key_value_heads=config["n_head_kv"],
        num_attention_heads=config["n_head_q"],
        intermediate_size=SwiGLU._get_hidden_dim(ffn_hidden=config["ffn_hidden"]),
        attention_bias=config["bias"],
        mlp_bias=config["bias"],
        hidden_act="silu",
        layer_norm_eps=_get_layer_norm_value(config[ffn_norm_key]["config"], "eps"),
        layer_norm_elementwise_affine=_get_layer_norm_value(config[ffn_norm_key]["config"], "elementwise_affine"),
        layer_norm_bias=_get_layer_norm_value(config[ffn_norm_key]["config"], "bias"),
        max_position_embeddings=config["sequence_length"],
        rope_theta=config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
        _attn_implementation=_map_attention_type(config),
        output_attentions=False,
        recurrent_blocks_indices=config.get("recurrent_blocks_indices", []),
        sample_iterations=config.get("sample_iterations", None),
        k_last_recurrence_gradient_backprops=config.get("k_last_recurrence_gradient_backprops", None),
        recurrent_blocks_max_recurrences=config.get("recurrent_blocks_max_recurrences", None),
        use_recurrence_embedding=config.get("use_recurrence_embedding", False),
        blocks_types=blocks_types
    )

class BlockTypes(str, Enum):
    """
    Enum class representing different block types.

    Attributes:
        STANDARD (str): Represents the standard GPT2 block type.
        RECURSIVE (str): Represents the recursive GPT2 block type.
        GROUP_RECURSIVE (str): Represents the group recursive GPT2 block type.
    """

    STANDARD = "STANDARD"
    RECURSIVE = "RECURSIVE"
    GROUP_RECURSIVE = "GROUP_RECURSIVE"

def generate_blocks_types(config: GPT2Config) -> list[str]:
    """Generates a list indicating the type of each block in the model (group_recurrent / recurrent / standard).

    Args:
        config (GPT2Config): The GPT2 model configuration.

    Returns:
        list[str]: A list where each entry is either "recurrent" or "standard" indicating the block type.
    """
    block_types = []
    recurrent_indices = config["recurrent_blocks_indices"]

    layer_index = 0
    while layer_index < config["n_layer"]:
        if layer_index in recurrent_indices:
            block_types.append(BlockTypes.RECURSIVE)
            layer_index += 1
            continue
        
        is_group_recursive = False
        for group in recurrent_indices:
            if isinstance(group, list) and layer_index == group[0]:
                # block_types += [BlockTypes.GROUP_RECURSIVE] * len(group)
                block_types.append(BlockTypes.GROUP_RECURSIVE)
                # layer_index += len(group)
                layer_index += 1
                is_group_recursive = True
                break
        if is_group_recursive:
            continue
        
        block_types.append(BlockTypes.STANDARD)
        layer_index += 1

    return block_types

    
            
    
    


def check_converted_model(hf_model: GPT2ForCausalLM, modalities_model: GPT2LLM, num_testruns: int, vocab_size: int, max_new_tokens: int = 50):
    """Tests the converted model by inputting a random token sequence and comparing the output logits of both models.

    Args:
        hf_model (GPT2ForCausalLM): Huggingface transformers model.
        modalities_model (GPT2LLM): Modalities model.
        num_testruns (int): Number of test runs to perform.
        vocab_size (int): Vocabulary size of the model. (Required for generating random input tokens.)
    """

    hf_model.eval()
    modalities_model.eval()

    hf_model = hf_model.to("cuda")
    modalities_model = modalities_model.to("cuda")
    for i in tqdm(range(num_testruns), desc="Testing converted model"):
        input_ids = torch.randint(0, vocab_size, (1, modalities_model.sequence_length - max_new_tokens -10*i), device=hf_model.device)
        inputs = {modalities_model.sample_key: input_ids.to(modalities_model.transformer.wte.weight.device)}
        llama_input_ids = copy.deepcopy(input_ids)
        modalities_inputs = copy.deepcopy(inputs)

        for t in range(max_new_tokens): # 10 new tokens to validate the caching
            with torch.no_grad():
                llama_logits = hf_model(input_ids=llama_input_ids).logits
                modalities_logits = modalities_model(modalities_inputs)[modalities_model.prediction_key]

            print(f"t: {t}, Len(input_ids): {llama_input_ids.shape}, Max diff: {torch.max(torch.abs(llama_logits - modalities_logits))}, Mean diff: {torch.mean(torch.abs(llama_logits - modalities_logits))}")
            assert llama_logits.shape == modalities_logits.shape
            assert torch.equal(llama_logits, modalities_logits)

            llama_logits = llama_logits[:, -1, :]
            modalities_logits = modalities_logits[:, -1, :]

            llama_next_token_id = torch.argmax(llama_logits, dim=-1).unsqueeze(0)
            modalities_next_token_id = torch.argmax(modalities_logits, dim=-1).unsqueeze(0)

            assert llama_next_token_id.item() == modalities_next_token_id.item()

            llama_input_ids = torch.cat((llama_input_ids, llama_next_token_id), dim=1)
            modalities_inputs[modalities_model.sample_key] = torch.cat((modalities_inputs[modalities_model.sample_key], modalities_next_token_id), dim=1)

            


def _check_conversion_criteria(model_config: dict) -> None:
    """Checks that the modalities config fulfills criteria necessary for conversion

    Args:
        model_config (dict): model or model_raw part of the Modalities config dictionary.

    Returns:
        None
    """
    assert model_config["poe_type"] == PositionTypes.NOPE
    assert model_config["activation_type"] == "swiglu"
    assert model_config["attention_implementation"] in ["pytorch_flash", "manual"]

    if "attention_norm" in model_config:
        norms = ["attention_norm", "ffn_norm", "lm_head_norm"]
        norm_type_key = "variant_key"
    else:
        norms = ["attention_norm_config", "ffn_norm_config", "lm_head_norm_config"]
        norm_type_key = "norm_type"
    for norm in norms:
        assert model_config[norm][norm_type_key] == "layer_norm"

    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "bias") for norm in norms)) == 1
    ), "All norms must have the same bias setting."
    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "elementwise_affine") for norm in norms)) == 1
    ), "All norms must have the same elementwise_affine setting."
    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "eps") for norm in norms)) == 1
    ), "All norms must have the same eps setting."


def _get_layer_norm_value(config: dict, field: str) -> bool | float | int:
    default = LayerNormConfig.model_fields[field].default
    return config.get(field, default)


def _map_attention_type(config: dict):
    if config["attention_implementation"] == "pytorch_flash":
        attention_impl = "sdpa"
    elif config["attention_implementation"] == "manual":
        attention_impl = "eager"
    else:
        raise ValueError(f"Unknown or unsupported attention implementation {config['attention_implementation']}.")
    return attention_impl


def _copy_weights_model(hf_model: GPT2ForCausalLM, modalities_model: GPT2LLM):
    """Copies the weights of the modalities model to the Huggingface transformers model.

    Args:
        hf_model (GPT2ForCausalLM): The uninitialized Huggingface transformers model.
                                    The weights will be copied here.
        modalities_model (GPT2LLM): The modalities model from which the weights will be copied.
    """
    hf_model.model.embed_tokens.weight.data.copy_(modalities_model.transformer.wte.weight.data)
    for hf_layer, modalities_layer in zip(hf_model.model.layers, modalities_model.transformer.h):
        _copy_weights_attention(hf_layer, modalities_layer)
        _copy_weights_mlp(hf_layer, modalities_layer)
        _copy_weights_layer_norms(hf_layer, modalities_layer)
        _copy_weights_recurrence_embedding(hf_layer, modalities_layer)
    _copy_weights_base_modules(hf_model.lm_head, modalities_model.transformer.lm_head)
    _copy_weights_base_modules(hf_model.model.norm, modalities_model.transformer.lm_head_norm)



    

def _copy_weights_attention(hf_layer: Union[GPT2DecoderLayer, RecursiveGPT2DecoderLayer, GroupRecursiveGPT2DecoderLayer], modalities_layer: GPT2Block):
    if isinstance(hf_layer, GroupRecursiveGPT2DecoderLayer):
        for i in range(len(hf_layer.gpt2_blocks)):
            hf_block = hf_layer.gpt2_blocks[i]
            modalities_block = modalities_layer.gpt2_blocks[i]
            _copy_weights_base_modules(hf_block.self_attn.q_proj, modalities_block.attn.q_attn)
            _copy_weights_base_modules(hf_block.self_attn.k_proj, modalities_block.attn.k_attn)
            _copy_weights_base_modules(hf_block.self_attn.v_proj, modalities_block.attn.v_attn)
            _copy_weights_base_modules(hf_block.self_attn.o_proj, modalities_block.attn.c_proj)
    else:
        _copy_weights_base_modules(hf_layer.self_attn.q_proj, modalities_layer.attn.q_attn)
        _copy_weights_base_modules(hf_layer.self_attn.k_proj, modalities_layer.attn.k_attn)
        _copy_weights_base_modules(hf_layer.self_attn.v_proj, modalities_layer.attn.v_attn)
        _copy_weights_base_modules(hf_layer.self_attn.o_proj, modalities_layer.attn.c_proj)


def _copy_weights_mlp(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    if isinstance(hf_layer, GroupRecursiveGPT2DecoderLayer):
        for i in range(len(hf_layer.gpt2_blocks)):
            hf_block = hf_layer.gpt2_blocks[i]
            modalities_block = modalities_layer.gpt2_blocks[i]
            _copy_weights_base_modules(hf_block.mlp.down_proj, modalities_block.mlp.W_2)
            _copy_weights_base_modules(hf_block.mlp.gate_proj, modalities_block.mlp.W)
            _copy_weights_base_modules(hf_block.mlp.up_proj, modalities_block.mlp.V)
    else:
        _copy_weights_base_modules(hf_layer.mlp.down_proj, modalities_layer.mlp.W_2)
        _copy_weights_base_modules(hf_layer.mlp.gate_proj, modalities_layer.mlp.W)
        _copy_weights_base_modules(hf_layer.mlp.up_proj, modalities_layer.mlp.V)


def _copy_weights_layer_norms(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    if isinstance(hf_layer, GroupRecursiveGPT2DecoderLayer):
        for i in range(len(hf_layer.gpt2_blocks)):
            hf_block = hf_layer.gpt2_blocks[i]
            modalities_block = modalities_layer.gpt2_blocks[i]  
            _copy_weights_base_modules(hf_block.input_layernorm, modalities_block.attention_norm)
            _copy_weights_base_modules(hf_block.post_attention_layernorm, modalities_block.ffn_norm)
    else:
        _copy_weights_base_modules(hf_layer.input_layernorm, modalities_layer.attention_norm)
        _copy_weights_base_modules(hf_layer.post_attention_layernorm, modalities_layer.ffn_norm)

def _copy_weights_recurrence_embedding(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    if isinstance(hf_layer, GroupRecursiveGPT2DecoderLayer):
        try:  
            # _copy_weights_base_modules(hf_layer.recurrence_embd, modalities_layer.recurrence_embd)
            assert hf_layer.recurrence_embd.weight.shape == modalities_layer.recurrence_embd.weight.shape
            hf_layer.recurrence_embd.weight.data.copy_(modalities_layer.recurrence_embd.weight.data)
            print("="*20, "weightes copied!")
        except Exception as e:
            print(e)
    # elif isinstance(hf_layer, GPT2DecoderLayer):
    #    pass
    elif hasattr(hf_layer, "recurrence_embd"):
        raise NotImplementedError("_copy_weights_recurrence_embedding hasn't been implemented yet for RecursiveGPT2DecoderLayer")


def _copy_weights_base_modules(m1: nn.Linear | nn.LayerNorm, m2: nn.Linear | nn.LayerNorm):
    assert m1.weight.shape == m2.weight.shape
    assert (m1.bias is None and m2.bias is None) or m1.bias.shape == m2.bias.shape
    m1.weight.data.copy_(m2.weight.data)
    if m1.bias is not None:
        m1.bias.data.copy_(m2.bias.data)