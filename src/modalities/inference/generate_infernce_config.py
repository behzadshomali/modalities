import os
import yaml
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--train_config_path", type=str)
parser.add_argument("--destination_path", type=str)


args = parser.parse_args()
train_config_path = args.train_config_path
destination_path = args.destination_path

with open(train_config_path, "r") as f:
    train_config = yaml.safe_load(f)


text_generation_config = {}

# settings
text_generation_config["settings"] = {}
text_generation_config["settings"]["device"] = 0
text_generation_config["settings"]["model_path"] = "" if "model_path" not in train_config["settings"] else train_config["settings"]["model_path"]
text_generation_config["settings"]["referencing_keys"] = train_config["settings"]["referencing_keys"]
text_generation_config["settings"]["sequence_length"] = 128

# tokenizer
if "tokenizer" in train_config:
    text_generation_config["tokenizer"] = train_config["tokenizer"]
else:
    text_generation_config["tokenizer"] = {}
    text_generation_config["tokenizer"]["component_key"] = "tokenizer"
    text_generation_config["tokenizer"]["variant_key"] = "pretrained_sp_tokenizer"
    text_generation_config["tokenizer"]["config"] = {"tokenizer_model_file": "/raid/s3/opengptx/behzad_shomali/modalities/Eurolingua_tokenizer/tokenizer.model"}

# text inference
text_generation_config["text_inference_component"] = {}
text_generation_config["text_inference_component"]["component_key"] = "inference_component"
text_generation_config["text_inference_component"]["variant_key"] = "text"
text_generation_config["text_inference_component"]["config"] = {
    "device": "${settings.device}",
    "model": {
        "instance_key": "checkpointed_model",
        "pass_type": "BY_REFERENCE"
    },
    "tokenizer": {
        "instance_key": "tokenizer",
        "pass_type": "BY_REFERENCE"
    },
    "sequence_length": "${settings.sequence_length}",
    "eod_token": "<|endoftext|>",
    "prompt_template": "{prompt_input}",
    "chat_template": "{user_prompt}",
    "temperature": 1
}

# model
keys_to_keep = ["model_raw", "wrapped_model", "model", "checkpointed_model"]
for k in keys_to_keep:
    if k in train_config:
        text_generation_config[k] = train_config[k]
    elif k == "checkpointed_model":
        print("!"*20, "You have to add the checkpoint path/model manually!", "!"*20)
        text_generation_config["checkpointed_model"] = {}
        text_generation_config["checkpointed_model"]["component_key"] = "model"
        text_generation_config["checkpointed_model"]["variant_key"] = "fsdp1_checkpointed"
        text_generation_config["checkpointed_model"]["config"] = {
            "checkpoint_loading": {
                "component_key": "checkpoint_loading",
                "variant_key": "torch",
                "config": {
                    "device": "${settings.device}",
                    "precision": "BF16"
                }
            },
            "model": {
                "instance_key": "model_raw",
                "pass_type": "BY_REFERENCE"
            },
            "checkpoint_path": "${settings.model_path}"
        }

text_generation_config["model_raw"]["config"]["prediction_key"] = r"${settings.referencing_keys.prediction_key}"
text_generation_config["model_raw"]["config"]["sequence_length"] = r"${settings.sequence_length}"


train_config_path = Path(train_config_path)
train_config_file_name = train_config_path.name
text_generation_config_path = os.path.join(destination_path, "text_generation_"+train_config_file_name)


with open(text_generation_config_path, "w") as f:
    yaml.dump(text_generation_config, f, default_flow_style=False, sort_keys=False)

print("The new config file has been saved under:", text_generation_config_path)