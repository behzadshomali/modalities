import os
import sys


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from utils import load_config

from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig, LlamaConfig
from recursive_llama2.recursive_llama import RecursiveLlamaConfig, RecursiveLlamaForCausalLM, BlockRecursiveModule


config_path = sys.argv[1]
config = load_config(config_path).copy()



import gc
import shutil
import json


from utils import (
    load_config, 
    clean_coda_alpaca, 
    print_trainable_params, 
    transform, 
    SavePeftModelCallback, 
    WandbOffsetCallback, 
    set_cache_dirs, 
    EvalCallback, 
    DiagnosticCallback,
    GraduallyIncreaseRecursionsCallback,
    register_global_gradient_tracking
)

set_cache_dirs(new_cache_dir=config["new_cache_dir"])

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import setup_chat_format, clone_chat_template
from trl.trainer import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

import evaluate

print("!"*20, "visible devices:", os.environ["CUDA_VISIBLE_DEVICES"], "!"*20)


sacrebleu = evaluate.load("sacrebleu")

def preprocess_logits_for_metrics(logits, labels, temperature=0.8):
    global config
    packing = config['sft']['packing']
    
    # Apply temperature
    logits = logits / temperature
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    if not packing:
        flat_probs = probs.reshape(-1, probs.size(-1))
        # Sample from the distribution
        sampled_tokens = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        # sampled_tokens = flat_probs.argmax(dim=-1)  # pick max prob token
        sampled_tokens = sampled_tokens.view(probs.size(0), probs.size(1))
    else:
        # sampled_tokens = probs.squeeze().argmax(dim=-1).unsqueeze(0)
        sampled_tokens = torch.multinomial(probs.squeeze(), num_samples=1).squeeze().unsqueeze(0)
    
    return sampled_tokens

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}

model_name = config["model_name"]

final_train_dataset, final_val_dataset = None, None
final_train_datasets = []
final_val_datasets = []
for dataset_obj in config["datasets"]:
    if dataset_obj['is_in_HF']:
        if "subset" in dataset_obj:
            dataset = load_dataset(dataset_obj['name'], dataset_obj["subset"], split=dataset_obj["split"])
        else:
            dataset = load_dataset(dataset_obj['name'], split=dataset_obj["split"])
    else:
        with open(dataset_obj['name'], "r") as f:
            raw_data = json.load(f)
        if "code_alpaca_20k" in dataset_obj['name']:
            cleaned_data = clean_coda_alpaca(raw_data)

        dataset = Dataset.from_list(cleaned_data)
    
    if "instruction_col" in dataset_obj and "response_col" in dataset_obj:
        dataset = dataset.map(
            config["preprocess_function"],
            remove_columns=dataset_obj["remove_columns"],
            fn_kwargs={
                "instruction_col": dataset_obj['instruction_col'],
                "response_col": dataset_obj['response_col']
            }
        )
    else:
        dataset = dataset.map(transform)
        dataset = dataset.map(
            config["preprocess_function"],
            remove_columns=dataset_obj["remove_columns"],
            fn_kwargs={
                "instruction_col": "instruction_col",
                "response_col": "response_col"
            }
        )
    

weights = [d["weight"] for d in config["datasets"]]
weight_sum = sum(weights)
norm_weights = [w / weight_sum for w in weights]  # normalize to sum=1

max_train_size = config["train_size"]
max_val_size = config["val_size"]

# --- sample datasets proportionally ---
for dataset_obj, nw in zip(config["datasets"], norm_weights):
    train_size = int(max_train_size * nw)
    val_size = int(max_val_size * nw)
    
    split = dataset.train_test_split(
        train_size=train_size,
        test_size=val_size,
        seed=config['random_seed'],
        shuffle=True
    )
    
    sampled_train = split["train"]
    sampled_test = split["test"]

    final_train_datasets.append(sampled_train)
    final_val_datasets.append(sampled_test)

final_train_dataset = concatenate_datasets(final_train_datasets).shuffle(seed=config["random_seed"])
final_val_dataset = concatenate_datasets(final_val_datasets).shuffle(seed=config["random_seed"])

dataset_offset = config["dataset_offset"]
final_train_dataset = final_train_dataset.select(range(dataset_offset, len(final_train_dataset)))

print("Train size:", len(final_train_dataset))
print("Val size:", len(final_val_dataset))


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype="auto",
    device_map="auto",
    # max_memory={0: "81GiB", 1: "0GiB"}
)

if "llama" in model_name and "recursion_settings" in config:
    recursion_config = config["recursion_settings"]

    del model
    base_model= LlamaForCausalLM.from_pretrained(config["model_name"])

    recursive_llama_config = RecursiveLlamaConfig(
        model_name=config["model_name"],
        recursion_indices=recursion_config["recursion_indices"],
        num_recursions=recursion_config["num_recursions"],
        track_diagnostics=recursion_config["track_diagnostics"],
        recurrent_blocks_have_residual=recursion_config.get("recurrent_blocks_have_residual", True),
        increase_steps=recursion_config.get("increase_steps", None),
        gradually_increase_recursions=recursion_config.get("gradually_increase_recursions", False),
        sample_random_recursion=recursion_config.get("sample_random_recursion", False),
        concatenate_iteration_outputs=recursion_config.get("concatenate_iteration_outputs", False)
    )

    model = RecursiveLlamaForCausalLM(recursive_llama_config, use_bf16=config.get("use_bf16", True)) 



print_trainable_params(model)
if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer)
print_trainable_params(model)

wandb_name = config['wandb']['name'] if 'name' in config['wandb'] else config['sft']['output_dir'].split("/")[-1]

wandb.init(
    project=config['wandb']['project'], 
    name=wandb_name
)
wandb.config.update(config)

sft_config = config['sft']
if "output_dir_orig" in sft_config:
    del sft_config["output_dir_orig"]
sft_args = SFTConfig(**sft_config)

if "peft" in config:
    peft_config = LoraConfig(**config["peft"])
    
    print_trainable_params(model)
    
    model = get_peft_model(model, peft_config)

    print_trainable_params(model)

    keys = []
    if "learn_embed" in config and config["learn_embed"]:
        keys.append("embed")
    if "learn_normalization" in config and config["learn_normalization"]:
        keys.append("norm")

    for n, p in model.named_parameters():
        if any(k in n for k in keys):
            p.requires_grad = True
    
print_trainable_params(model)


callbacks = []

if "eval_device" in config:
    eval_callback = EvalCallback(
        eval_gpu=config["eval_device"], 
        eval_on_start=config["sft"].get("eval_on_start", True),
        source_model_path=config['model_name'], 
        hf_home=config['new_cache_dir']
    )
    callbacks.append(eval_callback)

if "recursion_settings" in config:
    if config["recursion_settings"]["track_diagnostics"]:
        diagnostic_callback = DiagnosticCallback(
            model,
            block_modules=[layer for layer in model.model.layers if isinstance(layer, BlockRecursiveModule)],
            output_dir=os.path.join(config["sft"]["output_dir"], "diagnostic"),
            save_to_file=True
        )
        callbacks.append(diagnostic_callback)

        register_global_gradient_tracking(model)



    if config["recursion_settings"].get("gradually_increase_recursions", False):
        gradually_increase_callback = GraduallyIncreaseRecursionsCallback(
            block_modules=[layer for layer in model.model.layers if isinstance(layer, BlockRecursiveModule)],
            start_recursions=1,
            max_recursions=config["recursion_settings"]["num_recursions"],
            increase_steps=config["recursion_settings"].get("increase_steps", None),
            increase_every_n_steps=config["recursion_settings"].get("increase_every_n_steps", None),
            reset_optimizer=config["recursion_settings"].get("reset_optimizer", False)
        )
        callbacks.append(gradually_increase_callback)


trainer = SFTTrainer(
    model=model,
    train_dataset=final_train_dataset,
    eval_dataset=final_val_dataset,
    processing_class=tokenizer,
    args=sft_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=callbacks,
)

try:
    trainer.train(config.get("resume_from_checkpoint", False))
except KeyboardInterrupt:
    # shutil.rmtree(config["sft"]["output_dir"])
    pass

wandb.finish()

del model, tokenizer, trainer

# Run garbage collector
gc.collect()

# Empty CUDA cache
torch.cuda.empty_cache()
