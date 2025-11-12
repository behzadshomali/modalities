import yaml
from pathlib import Path
from transformers import TrainerCallback
import json
from datetime import datetime
import wandb
from typing import Dict, Any, Union
from merge_lora import merge_lora_adapter

import shutil 
import random

import logging
import torch

"""Evaluation utilities for model assessment."""
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from filelock import FileLock
import wandb

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

import os

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

    if "recursion_settings" in args:
        num_recursions = args["recursion_settings"]["num_recursions"]
        if isinstance(num_recursions, int):
            num_recursions = [num_recursions]
        recursion_indices = args["recursion_settings"]["recursion_indices"]
    block_name = f"block__{'__'.join([f'{start}_{end}' for (start, end) in recursion_indices])}__{'-'.join([str(n) for n in num_recursions])}"

    # Handle wandb project name if present
    # project_name = args.get("wandb", {}).get("name", "")

    if not args.get("resume_from_checkpoint", False):
        # Always build output dir from the original one (not the already-modified one)
        base_dir = args.get("output_dir_orig", sft_args['output_dir'])
        args['output_dir_orig'] = base_dir  # ensure stored once

        output_dir = os.path.join(f"{base_dir}", block_name, dir_name)
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
    elif preprocess_function_str == "format_openmathinstruct2_only_answer":
        args['preprocess_function'] = format_openmathinstruct2_only_answer
    else:
        raise ValueError(
            f"Preprocess function '{args['preprocess_function']}' is not valid. "
            f"Please choose from [format_openmathinstruct2/preprocess_function_simple/format_openmathinstruct2_only_answer]"
        )

    if "dataset_offset" in args:
        dataset_offset_value = 1
        if type(args['dataset_offset']) is str:
            for number in args['dataset_offset'].split('*'):
                number = int(number.strip())
                dataset_offset_value *= number
        else:
            dataset_offset_value = args['dataset_offset']

    else:
        dataset_offset_value = 0

    args['dataset_offset'] = dataset_offset_value

    if "global_step" in args:
        global_step_value = args['global_step']
    else:
        global_step_value = 1
    args['global_step'] = global_step_value


    wandb_name = f"ga{args['sft']['gradient_accumulation_steps']}-lr{args['sft']['learning_rate']}-wd{args['sft']['weight_decay']}-mgn{args['sft']['max_grad_norm']}"
    

    if "recursion_settings" in args:
        recursion_str = ""
        for k, v in args["recursion_settings"].items():
            if k == "start_layer":
                k = "beg"
            elif k == "end_layer":
                k = "end"
            elif k == "num_recursions":
                k = "num"
            elif k == "layer_indices":
                k = "layers"
            elif k == "type":
                k = ""
            elif k == "sample_random_recursion":
                k = "RAND"
            elif k == "track_diagnostics":
                k = "track"
            elif k == "neft_alpha":
                k = ""
            elif k == "gradually_increase_recursions":
                k = "gradual"
            elif k == "increase_steps":
                k = "incs"
            elif k == "reset_optimizer":
                k = "resetOpt"
            elif k == "recurrent_blocks_have_residual":
                k = "residual"
            elif k == "recursion_indices":
                k = "indices"
            elif k not in ["neft"]:
                raise ValueError(f"{k} is not valid!")

            recursion_str += f"{k}{v}-"
        wandb_name = f"{recursion_str}" + wandb_name

    args['wandb']['name'] = wandb_name + args['wandb'].get("name", "")
    if "peft" in args:
        wandb_name += f"-lora{args['peft']['r']}"


    # if overwrite_config:
    args_to_save = dict(args)
    args_to_save['preprocess_function'] = preprocess_function_str
    if not os.path.exists(sft_args['output_dir']):
        os.makedirs(sft_args['output_dir'], exist_ok=False)
    
    with open(os.path.join(sft_args['output_dir'], "config.yaml"), "w") as f:
        yaml.safe_dump(args_to_save, f)

    return args


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

def format_openmathinstruct2_only_answer(
    example: Dict[str, Any], 
    instruction_col: str, 
    response_col: str
) -> Dict[str, Any]:
    """Format OpenMathInstruct-2 dataset to chat format with instruction template."""
    instruction = "Solve the following math problem and put the final answer in \\boxed{}."
    formatted_problem = f"{instruction}\n\n{example[instruction_col]}"

    formatted_answer = f"The answer of this questions is: \\boxed{example['expected_answer']}"

    return {
        "messages": [
            {"role": "user", "content": formatted_problem},
            {"role": "assistant", "content": formatted_answer},
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
    
class WandbOffsetCallback(TrainerCallback):
    def __init__(self, step_offset=1):
        super().__init__()
        self.step_offset = step_offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step + self.step_offset)

class DiagnosticCallback(TrainerCallback):
    def __init__(self, model, block_modules, output_dir, log_frequency=1, save_to_file=True):
        super().__init__()
        self.model = model
        self.log_frequency = log_frequency
        self.block_modules = block_modules
        self.step_counter = 0
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        self.recur_cosine_file_name = os.path.join(output_dir, f"recur_cosine_{file_name}")
        self.recur_grad_file_name = os.path.join(output_dir, f"recur_grad_{file_name}")
        self.grad_file_name = os.path.join(output_dir, f"grad_{file_name}")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        with open(self.recur_cosine_file_name, 'w') as f:
            f.write(f"recur_block_idx,global_step,iteration,mean,min,max,std\n")
        
        with open(self.recur_grad_file_name, 'w') as f:
            f.write(f"recur_block_idx,global_step,step,layer,layer_name,is_recurrent,iteration,norm,mean,max,std\n")

        
        with open(self.grad_file_name, 'w') as f:
            f.write(f"global_step,step,layer,layer_name,norm,mean,max,std\n")


    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step_counter += 1
        
        # Determine if we should log this step
        should_log = False
        should_log = (self.step_counter % self.log_frequency == 0)

        if should_log and self.block_modules is not None:

            # Save to file if requested
            if self.save_to_file:
                self._save_diagnostics_to_file(state.global_step)

    def _save_diagnostics_to_file(self, global_step):
        """Save diagnostics to CSV file."""
        for i in range(len(self.block_modules)):
            block = self.block_modules[i]
            diagnostics = block.get_diagnostics()

            with open(self.recur_cosine_file_name, 'a') as f:
                # Write cosine similarities
                for sim in diagnostics.get('cosine_similarities', []):
                    f.write(f"{i},")
                    f.write(f"{global_step},{sim['iteration']},{sim['mean']:.6f},")
                    f.write(f"{sim['min']:.6f},{sim['max']:.6f},{sim['std']:.6f}\n")

            block.cosine_similarities = []

            with open(self.recur_grad_file_name, 'a') as f:
                grad_info = diagnostics.get('gradient_history', [{}])
                if grad_info == []:
                    grad_info = [{}]
                for grad in grad_info:
                    f.write(f"{i},{global_step:7d},")
                    f.write(f"{grad.get('step', 0):7d},")
                    f.write(f"{grad.get('layer', 0):1d},")
                    f.write(f"{grad.get('layer_name', )},")
                    f.write(f"{grad.get('is_recurrent', False)},")
                    f.write(f"{grad.get('iteration', 0):1d},")
                    f.write(f"{grad.get('grad_norm', 0):.6f},")
                    f.write(f"{grad.get('grad_mean', 0):.6f},")
                    f.write(f"{grad.get('grad_max', 0):.6f},")
                    f.write(f"{grad.get('grad_std', 0):.6f}\n")

            block.gradient_history = []

            with open(self.grad_file_name, 'a') as f:
                grad_info = self.model.gradient_history
                if grad_info == []:
                    grad_info = [{}]
                for grad in grad_info:
                    f.write(f"{i},{global_step:7d},")
                    f.write(f"{grad.get('step', 0):7d},")
                    f.write(f"{grad.get('layer', 0):1d},")
                    f.write(f"{grad.get('layer_name', )},")
                    f.write(f"{grad.get('grad_norm', 0):.6f},")
                    f.write(f"{grad.get('grad_mean', 0):.6f},")
                    f.write(f"{grad.get('grad_max', 0):.6f},")
                    f.write(f"{grad.get('grad_std', 0):.6f}\n")

        print("The results saved to:")
        print(self.recur_cosine_file_name)
        print(self.recur_grad_file_name)
        print(self.grad_file_name)






logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def setup_wandb_metrics():
    """Setup WandB metrics to allow out-of-order logging for evaluation metrics."""
    if wandb.run is not None:
        # Define evaluation metrics with step_metric to allow out-of-order logging
        wandb.define_metric("eval/*", step_metric="eval_step")
        wandb.define_metric("eval_step")
        logger.info("✅ WandB metrics configured for out-of-order evaluation logging")


def merge_peft_model(peft_path: str, base_model_path: str, output_dir: str) -> bool:
    """Merge PEFT adapters into base model."""
    try:
        logger.info(f"Merging PEFT model: {peft_path} with base: {base_model_path}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        # Load PEFT adapters
        model = PeftModel.from_pretrained(base_model, peft_path)

        # Merge adapters
        merged_model = model.merge_and_unload()

        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)

        # Copy custom files if they exist
        for filename in ["modeling_gpt2.py", "configuration_gpt2.py"]:
            src_file = Path(base_model_path) / filename
            if src_file.exists():
                dst_file = Path(output_dir) / filename
                dst_file.write_text(src_file.read_text())
                logger.info(f"Copied {filename}")

        logger.info(f"Successfully merged model to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to merge PEFT model: {e}")
        return False


def run_lighteval_cli(
    checkpoint_path: str, 
    step: int, 
    eval_gpu: int, 
    source_model_path: str, 
    eval_tasks: str, 
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Run LightEval CLI evaluation and return results."""
    rand_int = random.randint(1000, 100000)
    # This creates a unique lock file for the given id
    lock_path = f"/tmp/b_lighteval_gpu_{eval_gpu}_{rand_int}.lock"
    gpu_lock = FileLock(lock_path)

    try:
        logger.info(f"Process for step {step} is WAITING for GPU {eval_gpu} lock...")
        with gpu_lock: # pause here until the lock is acquired
            logger.info(f"Process for step {step} has ACQUIRED lock for GPU {eval_gpu}. Starting evaluation.")
            logger.info(f"Starting CLI evaluation for step {step} on {checkpoint_path}")

            checkpoint_dir = Path(checkpoint_path)
            eval_model_path = checkpoint_path
            merged_dir = None

            # Check if it's a PEFT model and merge if needed
            if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                logger.info("PEFT model detected, merging with base model...")
                merged_dir = os.path.join(checkpoint_path, "lora_merged")
                if not os.path.exists(merged_dir):
                    if not merge_lora_adapter(checkpoint_path, source_model_path, **kwargs):
                        return None
                else:
                    logger.info(f"Using existing merged model at {merged_dir}")
                eval_model_path = merged_dir
                checkpoint_dir = Path(merged_dir)

            model_args = (
                f"model_name={eval_model_path},"
                "use_chat_template=True,"
                "trust_remote_code=True,"
                "batch_size=16,"
                'generation_parameters={"temperature": 0.00001, "max_new_tokens": 1024}'
            )
            cmd_string = (
                f"lighteval accelerate "
                f'"{model_args}" '
                f'"{eval_tasks}" '
                f"--max-samples 100 "
                "--save-details "
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
            env["HF_HOME"] = hf_home
            logger.info(f"Running command: CUDA_VISIBLE_DEVICES={eval_gpu} {cmd_string}")

            result = subprocess.run(
                cmd_string,
                shell=True,
                env=env,
                capture_output=False,
                text=True,
                check=False,
                preexec_fn=os.setsid,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                logger.error(f"Evaluation failed for step {step}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None

            logger.info(f"CLI evaluation completed for step {step}")
            json_files = list(checkpoint_dir.glob("results_*.json"))
            if not json_files:
                logger.error(f"No results JSON file found in {checkpoint_dir}")

                home_dir = str(Path.home())
                possible_dir = Path(os.path.join(home_dir, "results/results", *source_model_path.split("/")))
                logger.info(f"Start looking into: {possible_dir}")
                
                json_files = list(possible_dir.glob("results_*.json"))
                if not json_files:
                    logger.error(f"Still no results JSON file found in {possible_dir}")
                    return None

            results_file = max(json_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Reading results from {results_file}")
            with open(results_file, "r") as f:
                eval_results = json.load(f)

            # The lock is automatically released when the 'with' block exits.
            logger.info(f"Process for step {step} has RELEASED lock for GPU {eval_gpu}.")
            return eval_results

    except Exception as e:
        logger.error(f"Evaluation failed for step {step}: {e}")
        return None


def parse_and_log_results(eval_results: Dict[str, Any], step: int) -> Dict[str, float]:
    """Parse LightEval results and log to WandB."""
    if not eval_results or "results" not in eval_results:
        logger.error(f"❌ No valid results to log for step {step}")
        return {}

    results_to_log = {}
    # Parse individual task results
    for task_name, metrics in eval_results["results"].items():
        if task_name == "all":  # Skip the aggregated results
            continue

        # Clean up task name for logging
        clean_task_name = task_name.replace("leaderboard|", "").split("|")[0]
        for metric_name, value in metrics.items():
            log_key = f"eval/{clean_task_name}_{metric_name}"
            results_to_log[log_key] = value

    # Log to WandB if available - include eval_step for out-of-order logging
    if results_to_log and wandb.run is not None:
        # Add the eval_step to the metrics
        results_to_log["eval_step"] = step
        # Log without specifying step parameter!
        wandb.log(results_to_log)
        logger.info(f"📊 Logged {len(results_to_log)} metrics to WandB for eval_step {step}")
        for key, value in results_to_log.items():
            if key != "eval_step":
                logger.info(f"  {key}: {value}")
    else:
        logger.warning(f"❌ No metrics to log for step {step}")

    return results_to_log


class AsyncEvaluator:
    """Asynchronous evaluator for running evaluations in background."""

    def __init__(
            self, 
            max_workers: int, 
            eval_gpu: int, 
            source_model_path: str, 
            eval_tasks: str,
            hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
            **kwargs
        ):
        self.hf_home = hf_home
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.eval_gpu = eval_gpu
        self.source_model_path = source_model_path
        self.eval_tasks = eval_tasks
        self.kwargs = kwargs

        # Setup WandB metrics when evaluator is created
        setup_wandb_metrics()

    def submit_evaluation(self, checkpoint_path: str, step: int):
        """Submit an evaluation job."""

        def eval_and_log(eval_step, eval_checkpoint_path):
            results = run_lighteval_cli(eval_checkpoint_path, eval_step, self.eval_gpu, self.source_model_path, self.eval_tasks, self.hf_home, **self.kwargs)
            if results:
                return parse_and_log_results(results, eval_step)
            return {}

        future = self.executor.submit(eval_and_log, step, checkpoint_path)  # Pass explicitly
        self.futures.append((future, step))

        # Clean up completed futures
        self.futures = [(f, s) for f, s in self.futures if not f.done()]

        logger.info(f"🎯 Evaluation job submitted for step {step}")

    def wait_for_completion(self):
        """Wait for all evaluations to complete."""
        if self.futures:
            logger.info("⏳ Waiting for remaining evaluations to complete...")
            for future, step in self.futures:
                try:
                    future.result()
                    logger.info(f"✅ Evaluation completed for step {step}")
                except Exception as e:
                    logger.error(f"❌ Evaluation failed for step {step}: {e}")

        self.executor.shutdown(wait=True)
        logger.info("✅ All evaluations completed")

    def get_completed_results(self) -> List[tuple]:
        """Get results from completed evaluations."""
        completed = []
        remaining = []

        for future, step in self.futures:
            if future.done():
                try:
                    result = future.result()
                    completed.append((step, result))
                except Exception as e:
                    logger.error(f"Error getting result for step {step}: {e}")
                    completed.append((step, None))
            else:
                remaining.append((future, step))

        self.futures = remaining
        return completed
    
class EvalCallback(TrainerCallback):
    """Callback to trigger async LightEval CLI on checkpoint saves and at training start."""

    def __init__(self, eval_gpu: int, source_model_path: str, hf_home: str, **kwargs):
        self.eval_gpu = eval_gpu
        self.source_model_path = source_model_path
        self.hf_home = hf_home
        self.evaluator = AsyncEvaluator(
            max_workers=1,  
            eval_gpu=eval_gpu, 
            source_model_path=source_model_path,
            eval_tasks="leaderboard|gsm8k|8|1,leaderboard|hellaswag|5|1",
            hf_home=hf_home,
            **kwargs
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """Run initial evaluation on the base model at step 0."""
        logger.info("🔍 Running initial evaluation on base model at step 0...")
        self.evaluator.submit_evaluation(self.source_model_path, step=0)

    def on_save(self, args, state, control, **kwargs):
        """Trigger evaluation when checkpoint is saved."""

        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # # manually copy the modeling_gpt2.py file
        # src_path = "/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/conversion/gpt2/modeling_gpt2.py"
        # dst_path = os.path.join(checkpoint_path, "modeling_gpt2.py")

        # shutil.copy(src_path, dst_path)

        # manually copy the modeling_recursive_llama.py file
        src_path = "/raid/s3/opengptx/behzad_shomali/modalities/src/recursive_llama2/recursive_llama.py"
        dst_path = os.path.join(checkpoint_path, "modeling_recursive_llama.py")
        shutil.copy(src_path, dst_path)

        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config.get('model_type', '') == 'recursive-llama':
            config['auto_map'] = {
                'AutoConfig': 'modeling_recursive_llama.RecursiveLlamaConfig',
                'AutoModelForCausalLM': 'modeling_recursive_llama.RecursiveLlamaForCausalLM',
            }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        
        if os.path.exists(checkpoint_path):
            logger.info(f"💾 Checkpoint saved at step {state.global_step}, triggering evaluation...")
            self.evaluator.submit_evaluation(checkpoint_path, state.global_step)
        else:
            logger.warning(f"⚠️ Checkpoint path {checkpoint_path} does not exist, skipping evaluation")

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all evaluations to complete."""
        self.evaluator.wait_for_completion()



class GraduallyIncreaseRecursionsCallback(TrainerCallback):
    """Callback to gradually increase the number of recursions during training."""

    def __init__(self, block_modules, start_recursions: int, max_recursions: int, increase_steps: list = None, increase_every_n_steps: int = None, reset_optimizer: bool = False):
        self.block_modules = block_modules
        self.start_recursions = start_recursions
        self.max_recursions = max_recursions
        self.increase_steps = increase_steps
        self.increase_every_n_steps = increase_every_n_steps
        self.reset_optimizer = reset_optimizer

        print(f"🔢 Setting initial recursions to {self.start_recursions}")
        print(f"Reset optimizer on recursion increase: {self.reset_optimizer}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Set initial number of recursions at training start."""
        for block_module in self.block_modules:
            block_module.set_num_recursions(self.start_recursions)
        print(f"🚀 Training started with {self.start_recursions} recursions")

    def on_step_end(self, args, state, control, **kwargs):
        """Increase recursions at specified intervals."""
        trainer = kwargs.get("trainer", None)  # access trainer (for optimizer)
        if state.global_step > 0:
            for i, block_module in enumerate(self.block_modules):
                current_recursions = block_module.num_recursions

                def increase_recursions():
                    new_recursions = min(current_recursions + 1, self.max_recursions)
                    block_module.set_num_recursions(new_recursions)
                    print(f"🔄 Increased recursions to {new_recursions} at step {state.global_step} for block w/ (relative) index: {i}")

                    if self.reset_optimizer and trainer is not None:
                        print("🧹 Resetting optimizer state...")
                        self._reset_optimizer(trainer.optimizer)
                        # if trainer.lr_scheduler is not None:
                        #     trainer.lr_scheduler.last_epoch = -1

                if self.increase_every_n_steps is not None:
                    if (state.global_step % self.increase_every_n_steps == 0
                            and current_recursions < self.max_recursions):
                        increase_recursions()
                elif self.increase_steps is not None:
                    if (state.global_step in self.increase_steps
                            and current_recursions < self.max_recursions):
                        increase_recursions()
                else:
                    raise ValueError(
                        "Either increase_every_n_steps or increase_steps list must be provided."
                    )

    @staticmethod
    def _reset_optimizer(optimizer):
        """Reset optimizer states (e.g., Adam’s moment estimates)."""
        for param_group in optimizer.param_groups:
            param_group["step"] = 0
        optimizer.state = {}  # clears momentum, exp averages, etc.


def register_global_gradient_tracking(model):
    """
    Register hooks on all parameters in the model to track their gradients.
    
    Args:
        model: The full nn.Module (e.g., the Llama model)
        tracker: An object (e.g., block_module) with a `.gradient_history` list
                 and `.step_count` counter to store gradient statistics.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        def make_param_grad_hook(param_name):
            def hook(grad):
                if "model.layers." in param_name:
                    layer_idx = int(param_name.split(".")[2] )
                else:
                    layer_idx = -100
                model.gradient_history.append({
                    'layer_name': param_name,
                    'layer': layer_idx,
                    'step': model.step_count,
                    'grad_norm': grad.norm().item(),
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_max': grad.abs().max().item(),
                })
                model.step_count += 1
                return grad
            return hook

        param.register_hook(make_param_grad_hook(name))
