import logging
from typing import Dict, Tuple
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# from modalities.conversion.pondering.configuration_pondering import PonderingConfig
# from modalities.conversion.pondering.modeling_pondering import PonderingForCausalLM
# from modalities.pondering_lm.custom_modules import PonderingModelForCausalLM, PonderingModelConfig
from modalities.conversion.gpt2.conversion_model import _copy_weights_model
from modalities.conversion.pondering.conversion_code2 import PonderingModelForCausalLM, PonderingModelConfig
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.conversion.gpt2.conversion_model import _get_layer_norm_value, _map_attention_type
from modalities.models.model import SwiGLU


logging.basicConfig(
    level=logging.DEBUG,  # Set minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_model_checkpoint(config: Dict) -> Tuple[PonderingModelForCausalLM, torch.nn.Module]:
    """Converts a modalities pondering model checkpoint to Huggingface format.

    Args:
        config (Dict): The modalities config dictionary containing model and checkpoint info.

    Returns:
        Tuple[PonderingModelForCausalLM, torch.nn.Module]: The converted HF model and original modalities model.
    """
    # Get model config section
    model_config_key = "model_raw" if "model_raw" in config else "model"
    # modalities_model_config = config[model_config_key]["config"]
    
    # Load the modalities model
    logger.info("Loading modalities model...")
    from modalities.registry.components import COMPONENTS
    
    # model_component = COMPONENTS[config[model_config_key]["component_key"]][
    #     config[model_config_key]["variant_key"]
    # ]
    # modalities_model = model_component(**modalities_model_config)
    modalities_model = get_model_from_config(config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    
    # Load checkpoint if specified
    if "checkpointed_model" in config:
        checkpoint_path = config["checkpointed_model"]["config"]["checkpoint_path"]
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        modalities_model.load_state_dict(state_dict)
    
    # Extract base model config
    # base_model = modalities_model
    if hasattr(modalities_model, "model"):
        # where the model is of type PonderingForCausalLM
        base_model = modalities_model.model.base_model
    else:
        # where the model is of type GPT2LLM
        base_model = modalities_model
    base_model_config = _extract_base_model_config(base_model, config['model_raw']['config'])
    
    # Create pondering config
    ffn_norm_key = "ffn_norm" if "ffn_norm" in config else "ffn_norm_config"
    pondering_config = PonderingModelConfig(
        base_model_config=base_model_config,
        base_model_type="gpt2",
        pondering_steps=config['pondering_model']['config']["pondering_steps"],
        topk=config['pondering_model']['config']["topk"],
        softmax_temperature=config['pondering_model']['config']["softmax_temperature"],
        apply_embed_scale=config['pondering_model']['config']["apply_embed_scale"],
        inverse_scale=config['pondering_model']['config']["inverse_scale"],
        grad_checkpointing=config['pondering_model']['config']["grad_checkpointing"],
        # seed=config['pondering_model']['config']["seed"]
        vocab_size=base_model_config['vocab_size'],
        hidden_size=base_model_config["n_embd"],
        pad_token_id=None,
        num_hidden_layers=base_model_config["n_layer"],
        num_key_value_heads=base_model_config["n_head_kv"],
        num_attention_heads=base_model_config["n_head_q"],
        intermediate_size=SwiGLU._get_hidden_dim(ffn_hidden=base_model_config["ffn_hidden"]),
        attention_bias=base_model_config["bias"],
        mlp_bias=base_model_config["bias"],
        hidden_act="silu",
        layer_norm_eps=_get_layer_norm_value(base_model_config[ffn_norm_key]["config"], "eps"),
        layer_norm_elementwise_affine=_get_layer_norm_value(base_model_config[ffn_norm_key]["config"], "elementwise_affine"),
        layer_norm_bias=_get_layer_norm_value(base_model_config[ffn_norm_key]["config"], "bias"),
        max_position_embeddings=base_model_config["sequence_length"],
        rope_theta=base_model_config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
        _attn_implementation=_map_attention_type(base_model_config),
        output_attentions=False,
    )
    
    # Create HF model
    logger.info("Creating HuggingFace model...")
    hf_model = PonderingModelForCausalLM(pondering_config).to(dtype=torch.bfloat16)

    if hasattr(modalities_model, "model"):
        # where the model is of type PonderingForCausalLM
        _copy_weights_model(hf_model.inner_model, modalities_model.model.base_model)
    else:
        # where the model is of type GPT2LLM
        _copy_weights_model(hf_model.inner_model, modalities_model)


    # # Convert state dict
    # logger.info("Converting state dict...")
    # hf_state_dict = _convert_state_dict(modalities_model.state_dict())
    # hf_model.load_state_dict(hf_state_dict, strict=True)
    
    logger.info("Conversion completed successfully!")
    return hf_model, modalities_model


def _extract_base_model_config(base_model: torch.nn.Module, config: Dict) -> Dict:
    """Extract base model configuration for HF config.

    Args:
        base_model: The base model instance
        config: The modalities config dict

    Returns:
        Dict: Configuration dictionary for base model
    """
   
    return {k: v for k, v in config.items()}


def _convert_model_keys(modules_dict: Dict[str, torch.nn.Module]) -> Dict[str, torch.nn.Module]:
    """Convert modalities model keys to HuggingFace format.

    Args:
        modules_dict: State dict from modalities model

    Returns:
        Dict[str, torch.nn.Module]: Converted state dict for HF model
    """
    converted_dict = {}
    
    for key, module in modules_dict.items():
        # # Remove 'model.' prefix if present (from PonderingModelForCausalLM wrapper)
        # if key.startswith("model."):
        #     key = key[6:]  # Remove "model."
        
        # # Convert base_model weights to model.base_model
        # if not key.startswith("base_model."):
        #     new_key = "model." + key
        # else:
        #     new_key = key
        new_key = "model.base_model." + key
        
        converted_dict[new_key] = module
    
    return converted_dict


def _convert_state_dict(modalities_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert modalities state dict to HuggingFace format.

    Args:
        modalities_state_dict: State dict from modalities model

    Returns:
        Dict[str, torch.Tensor]: Converted state dict for HF model
    """
    hf_state_dict = {}
    
    for key, value in modalities_state_dict.items():        
        # Convert base_model weights to model.base_model
        if key.startswith("model.base_model."):
            new_key = key[6:]  # Remove "model."
        else:
            new_key = key

        new_key = new_key.replace("base_model", "inner_model")
        
        hf_state_dict[new_key] = value
    
    return hf_state_dict


def check_converted_model(
    hf_model: PonderingModelForCausalLM,
    modalities_model: torch.nn.Module,
    num_testruns: int,
    vocab_size: int,
):
    """Check if the converted model produces the same outputs as the original.

    Args:
        hf_model: The converted HuggingFace model
        modalities_model: The original modalities model
        num_testruns: Number of test runs to perform
        vocab_size: Vocabulary size for generating random inputs
    """
    logger.info(f"Running {num_testruns} test runs to verify conversion...")
    
    hf_model.eval()
    modalities_model.eval()
    
    for i in range(num_testruns):
        # Generate random input
        batch_size = 2
        # get the model's maximum sequence length
        if hasattr(modalities_model, "model"):
            # where the model is of type PonderingForCausalLM
            seq_len = modalities_model.model.base_model.sequence_length
        else:
            # where the model is of type GPT2LLM
            seq_len = modalities_model.sequence_length            
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        sample_key = modalities_model.model.base_model.sample_key if hasattr(modalities_model, "model") else modalities_model.sample_key
        device = modalities_model.model.base_model.transformer.wte.weight.device if hasattr(modalities_model, "model") else modalities_model.transformer.wte.weight.device
        inputs = {sample_key: input_ids.to(device)}

        prediction_key = modalities_model.model.base_model.prediction_key if hasattr(modalities_model, "model") else modalities_model.prediction_key
        # Get outputs from both models
        with torch.no_grad():
            # Modalities model
            modalities_logits = modalities_model(inputs)[prediction_key].to("cpu")

            # HF model
            hf_output = hf_model(inputs)
            hf_logits = hf_output['logits'].to("cpu")
        
        # Compare logits
        max_diff = torch.max(torch.abs(modalities_logits - hf_logits)).item()
        mean_diff = torch.mean(torch.abs(modalities_logits - hf_logits)).item()
        
        logger.info(f"Test run {i+1}/{num_testruns}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        if max_diff > 1e-4:
            logger.warning(f"Large difference detected! max_diff={max_diff}")
            raise ValueError("Converted model outputs do not match original model outputs.")
        else:
            logger.info("✓ Outputs match!")
    
    logger.info("Verification complete!")