import yaml
import time
import os
from pathlib import Path
from transformers import TrainerCallback
import json
import shutil
from datetime import datetime
import asyncio
import threading
import wandb
from typing import Dict, Any
from merge_lora import merge_lora_adapter

import logging
import torch

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

PREFIX_CHECKPOINT_DIR = "checkpoint"


def transform(example):
    user_message = None
    assistant_message = None
    
    for msg in example["messages"]:
        if msg["role"] == "user":
            user_message = msg["content"]
        elif msg["role"] == "assistant":
            assistant_message = msg["content"]
    
    return {
        "instruction_col": user_message,
        "response_col": assistant_message
    }

import os
import yaml
from datetime import datetime

def load_config(config_path, overwrite_config=True):
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
    
    sft_args = args['sft']
    now = datetime.now()
    dir_name = now.strftime("%Y_%m_%d-%H_%M_%S")

    # Handle wandb project name if present
    project_name = args.get("wandb", {}).get("name", "")

    if not args.get("resume_from_checkpoint", False):
        # Always build output dir from the original one (not the already-modified one)
        base_dir = args.get("output_dir_orig", sft_args['output_dir'])
        args['output_dir_orig'] = base_dir  # ensure stored once

        output_dir = os.path.join(f"{base_dir}_{project_name}", dir_name)
        sft_args['output_dir'] = output_dir
    
    # Cast learning rate to float for safety
    sft_args['learning_rate'] = float(sft_args['learning_rate'])

    args['sft'] = sft_args    

    # Resolve preprocess function string to actual function
    preprocess_function_str = args['preprocess_function']
    if preprocess_function_str == "format_openmathinstruct2":
        args['preprocess_function'] = format_openmathinstruct2
    elif preprocess_function_str == "preprocess_function_simple":
        args['preprocess_function'] = preprocess_function_simple

    for d in args["datasets"]:
        if 'offset' in d:
            offset_value = 1
            if type(d['offset']) is str:
                for number in d['offset'].split(','):
                    number = int(number.strip())
                    offset_value *= number
            else:
                offset_value = d['offset']
        else:
            offset_value = 0

        d['offset'] = offset_value

    if overwrite_config:
        # Save back without dumping function objects
        args_to_save = dict(args)
        args_to_save['preprocess_function'] = preprocess_function_str
        with open(config_path, "w") as f:
            yaml.safe_dump(args_to_save, f)

    return args

async def lighteval_async(checkpoint_path, cuda_devices="3,4"):
    checkpoint_path = merge_lora_adapter(checkpoint_path)
    output_dir = "/raid/s3/opengptx/behzad_shomali/evaluation_results/sft_intermediate_results/"
    multi_gpu_command = "--multi_gpu" if len(cuda_devices.split(',')) > 1 else ""

    command = f"""\
CUDA_VISIBLE_DEVICES={cuda_devices} accelerate launch \
    --main_process_port 2000 \
    --num_processes {len(cuda_devices.split(','))} \
    -m \
    lighteval accelerate \
    "model_name={checkpoint_path},trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|7|0,leaderboard|hellaswag|7|0" \
    "--max-samples 100" \
    --output-dir {output_dir} \
    --use-chat-template
"""
    
    # Run subprocess asynchronously
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=os.getcwd()
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print("Evaluation failed!", stderr.decode())
        return None

    # Find latest results JSON
    output_dir = Path(output_dir)
    json_files = [f for f in output_dir.glob("*.json") if f.is_file()]

    if not json_files:
        return None

    json_files.sort(key=lambda f: f.stat().st_ctime, reverse=True)
    results_file = json_files[0]

    with open(results_file, "r") as f:
        results_dict = json.load(f)["results"]

    final_results = {
        benchmark: v["qem"]
        for benchmark, v in results_dict.items()
        if benchmark != "all"
    }

    return final_results


class LightEvalCallback(TrainerCallback):
    def __init__(self, output_dir, cuda_devices="0,1"):
        super().__init__()
        self.cuda_devices = cuda_devices
        self.output_dir = output_dir
        self.pending_tasks = []
        self.checked_checkpoints = []

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        dirs = [d for d in Path(self.output_dir).iterdir() if d.is_dir()]
        if dirs:
            last_created_dir = max(dirs, key=lambda d: d.stat().st_ctime)
            checkpoint_path = os.path.join(self.output_dir, last_created_dir)


            shutil.copy("/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py", checkpoint_path)

            print(f"[LightEvalCallback] Checkpoint saved: {checkpoint_path}")
        else:
            print("No directories found in", self.output_dir)
            checkpoint_path = None

        if checkpoint_path and checkpoint_path not in self.checked_checkpoints:
            time.sleep(10)
            # Run evaluation in a separate thread to avoid blocking training
            thread = threading.Thread(
                target=self._run_eval_thread, 
                args=(checkpoint_path, state.global_step)
            )
            thread.start()
            self.pending_tasks.append(thread)
            self.checked_checkpoints.append(checkpoint_path)

    def _run_eval_thread(self, checkpoint_path, step):
        """Thread target to run async evaluation safely."""
        import asyncio
        try:
            asyncio.run(self.run_eval(checkpoint_path, step))
        except Exception as e:
            print(f"[LightEvalCallback] Evaluation failed for {checkpoint_path}: {e}")

    async def run_eval(self, checkpoint_path, step):
        print(f"[LightEvalCallback] Starting evaluation for {checkpoint_path}")
        results = await lighteval_async(
            cuda_devices=self.cuda_devices,
            checkpoint_path=checkpoint_path
        )

        if results is None:
            print("[LightEvalCallback] Evaluation failed or returned no results.")
            return

        # Log results to wandb
        wandb.log({f"lighteval/{k}": v for k, v in results.items()}, step=step)
        print(f"[LightEvalCallback] Logged results to wandb: {results}")

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all pending evaluations before exiting."""
        if self.pending_tasks:
            print("[LightEvalCallback] Waiting for all pending evaluations to finish...")
            for thread in self.pending_tasks:
                thread.join()
            print("[LightEvalCallback] All evaluations finished and logged.")


def clean_coda_alpaca(raw_data):
    final_data = []
    for row in raw_data:
        if len(row['input']) == 0:
            final_data.append({
                "instruction": row["instruction"],
                "output": row["output"]
            })

    print(f"Kept {len(final_data)}, {len(final_data)/len(raw_data)} from code alpaca!")
    return final_data

def format_openmathinstruct2(
    example: Dict[str, Any], 
    instruction_col: str, 
    response_col: str
) -> Dict[str, Any]:
    """Format OpenMathInstruct-2 dataset to chat format with instruction template."""
    instruction = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
    formatted_problem = f"{instruction}\n\n{example[instruction_col]}"

    return {
        "messages": [
            {"role": "user", "content": formatted_problem},
            {"role": "assistant", "content": example[response_col]},
        ]
    }

def preprocess_function_simple(example, instruction_col, response_col):
    return {
        "messages": [
            {"role": "user", "content": example[instruction_col]},
            {"role": "assistant", "content": example[response_col]}
        ]
    }

def set_cache_dirs(new_cache_dir):
    transformers_cache = os.path.join(new_cache_dir, "transformers")
    datasets_cache = os.path.join(new_cache_dir, "datasets")
    tokenizers_cache = os.path.join(new_cache_dir, "tokenizers")
    hub_cache = os.path.join(new_cache_dir, "hub")

    for path in [transformers_cache, datasets_cache, tokenizers_cache, hub_cache]:
        os.makedirs(path, exist_ok=True)

    os.environ["HF_HOME"] = new_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["HF_TOKENIZERS_CACHE"] = tokenizers_cache
    os.environ["HF_HUB_CACHE"] = hub_cache

def print_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"Trainable params: {trainable_params} | "
          f"All params: {all_params} | "
          f"Trainable%: {100 * trainable_params / all_params:.2f}")
    
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)

        modules_to_save = []
        for module_name in ["embed", "norm"]:
            if len(module_name.strip()) > 0:
                modules_to_save.append(module_name)

        # Save trainable parameters if exist
        if modules_to_save:
            state_dict = kwargs["model"].state_dict()
            to_save = {}
            for key, value in state_dict.items():
                if any(module_name in key for module_name in modules_to_save):
                    to_save[key.replace("base_model.model.", "")] = value
            torch.save(to_save, os.path.join(checkpoint_folder, "trainable_params.bin"))
            logging.info(f"Trainable parameters saved at: {checkpoint_folder}")

        # Save LoRA adapter weight
        kwargs["model"].save_pretrained(checkpoint_folder)
        logging.info(f"LoRA adapter weights saved at: {checkpoint_folder}")

        return control