import os
import gc
import torch
import shutil
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import SFTTrainer, SFTConfig, setup_chat_format, clone_chat_template
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import evaluate


from utils import load_config, preprocess_function, LightEvalCallback, preprocess_coda_alpaca


sacrebleu = evaluate.load("sacrebleu")

def preprocess_logits_for_metrics(logits, labels, temperature=1.0):
    global config
    packing = config['sft']['packing']
    
    # Apply temperature
    logits = logits / temperature
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    if not packing:
        flat_probs = probs.reshape(-1, probs.size(-1))
        sampled_tokens = torch.multinomial(flat_probs, num_samples=1)#.squeeze(-1).unsqueeze(0)
        sampled_tokens = sampled_tokens.view(probs.size(0), probs.size(1))
    else:
        sampled_tokens = torch.multinomial(probs.squeeze(), num_samples=1).squeeze(-1).unsqueeze(0)
    
    return sampled_tokens

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}


config_path = "/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/gsm8k_socratic_config.yaml"
config = load_config(config_path).copy()


model_name = "Behzadshomali/Teuken3.7B"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
# model, tokenizer = setup_chat_format(model, tokenizer)
model, tokenizer, added_tokens = clone_chat_template(model, tokenizer, "Qwen/Qwen1.5-72B-Chat")


# if 'test_split_ratio' in config["dataset"] and 'subset' not in config["dataset"]:
final_train_dataset, final_val_dataset = None, None
final_train_datasets = []
final_val_datasets = []
for dataset_obj in config["datasets"]:
    if dataset_obj['is_in_HF']:
        if "subset" in dataset_obj:
            dataset = load_dataset(dataset_obj['name'], dataset_obj["subset"], split="train")
        else:
            dataset = load_dataset(dataset_obj['name'], split="train")
    else:
        with open(dataset_obj['name'], "r") as f:
            raw_data = json.load(f)
        if "code_alpaca_20k" in dataset_obj['name']:
            preprocessed_data = preprocess_coda_alpaca(raw_data)

        dataset = Dataset.from_list(preprocessed_data)
    
    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset_obj["remove_columns"],
        fn_kwargs={
            "instruction_col": dataset_obj['instruction_col'],
            "response_col": dataset_obj['response_col']
        }
    )
    
    split = dataset.train_test_split(
        test_size=dataset_obj['test_split_ratio'],
        seed=dataset_obj['random_seed']
    )
    
    dataset_obj["train_len"] = len(split["train"])
    dataset_obj["val_len"] = len(split["test"])
    dataset_obj["split"] = split

# --- compute max feasible size based on weights ---
weights = [d["weight"] for d in config["datasets"]]
weight_sum = sum(weights)
norm_weights = [w / weight_sum for w in weights]  # normalize to sum=1

# train set
scales = [d["train_len"] / nw for d, nw in zip(config["datasets"], norm_weights)]
max_train_size = int(min(scales))  # maximum possible while satisfying all
# test set
scales = [d["val_len"] / nw for d, nw in zip(config["datasets"], norm_weights)]
max_val_size = int(min(scales))

# --- sample datasets proportionally ---
for d, nw in zip(config["datasets"], norm_weights):
    train_size = int(max_train_size * nw)
    val_size = int(max_val_size * nw)
    
    sampled_train = d["split"]["train"].shuffle(seed=42).select(range(train_size))
    sampled_test = d["split"]["test"].shuffle(seed=42).select(range(val_size))
    
    final_train_datasets.append(sampled_train)
    final_val_datasets.append(sampled_test)

# Concatenate all datasets
final_train_dataset = concatenate_datasets(final_train_datasets).shuffle(seed=42)
final_val_dataset = concatenate_datasets(final_val_datasets).shuffle(seed=42)


wandb.init(
    project=config['wandb']['project'], 
    name=config['sft']['output_dir'].split("/")[-1]
)
wandb.config.update(config)


sft_args = SFTConfig(**config['sft'])

peft_config = None if "peft" not in config else LoraConfig(**config["peft"])

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    train_dataset=final_train_dataset,
    eval_dataset=final_val_dataset,
    processing_class=tokenizer,
    args=sft_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # callbacks=[LightEvalCallback(cuda_devices="4", output_dir=config['sft']['output_dir'])]
)


try:
    trainer.train()
except KeyboardInterrupt:
    shutil.rmtree(config["sft"]["output_dir"])

wandb.finish()

del model, tokenizer, trainer

# Run garbage collector
gc.collect()

# Empty CUDA cache
torch.cuda.empty_cache()
# except Exception as e:
#     print(c, e)
#     wandb.finish()

#     del model, tokenizer, trainer

#     # Run garbage collector
#     gc.collect()

#     # Empty CUDA cache
#     torch.cuda.empty_cache()