import os
import sys
import wandb
import time
import json
import glob
import gc
import torch
import subprocess
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merge_lora import merge_lora_adapter

def run_lighteval(
    base_model,
    checkpoint_path, 
    cuda_devices="0",
    benchmarks='"leaderboard|gsm8k|5|1,leaderboard|hellaswag|5|1,leaderboard|arc:challenge|5|1,leaderboard|truthfulqa:mc|5|1"',
    max_sampels=100
):
    checkpoint_path = merge_lora_adapter(lora_model=checkpoint_path, base_model=base_model)
    # output_dir = os.path.join(checkpoint_path, "intermediate_results/")
    multi_gpu_command = "--multi_gpu" if len(cuda_devices.split(',')) > 1 else ""

    command = f"""\
CUDA_VISIBLE_DEVICES={cuda_devices} accelerate launch \
--main_process_port 2000 \
--num_processes {len(cuda_devices.split(','))} \
-m \
lighteval accelerate \
"model_name={checkpoint_path},trust_remote_code=True,use_chat_template=True" \
{benchmarks} \
--max-samples {max_sampels} \
--output-dir {checkpoint_path} \
--use-chat-template
"""
    
    try:
        print("Running command:\n", command)
        process = subprocess.run(command, shell=True, check=True, text=True)
        print("Process executed successfully!")
        print("Output:\n", process.stdout)
    except subprocess.CalledProcessError as e:
        print("Process failed with return code:", e.returncode)
        print("Error output:\n", e.stderr)

    gc.collect()

    torch.cuda.empty_cache()
    return checkpoint_path

def get_checkpoint_steps(checkpoint_dir):
    # checkpoint_dir: /.../.../checkpoint-XXXX
    return int(checkpoint_dir.split("/")[-1].split("-")[1])

def is_newest_checkpoint_evaluated(path, evaluated_checkpoints):
    path = Path(path)
    dirs = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]
    
    if not dirs:
        print("No directories found!")
        return None  # No directories found
    
    latest_checkpoint_path = max(dirs, key=os.path.getmtime)

    return latest_checkpoint_path in evaluated_checkpoints, latest_checkpoint_path

def parse_results(file_dir):
    benchmarks_metrics = {
        "leaderboard|arc:challenge": "acc_norm",
        "leaderboard|gsm8k": "qem",
        "leaderboard|hellaswag": "acc_norm",
        "leaderboard|mmlu:high_school_mathematics": "acc",
        "leaderboard|truthfulqa": "truthfulqa_mc2",
    }

    # Find latest results JSON
    file_dir = Path(file_dir)
    json_files = [f for f in file_dir.glob("*.json") if f.is_file() and "results" in f.name]

    if not json_files:
        print("No results json file found!")
        return None

    json_files.sort(key=lambda f: f.stat().st_ctime, reverse=True)
    results_file = json_files[0]

    with open(results_file, "r") as f:
        results_dict = json.load(f)["results"]

    final_results = {}
    for benchmark, v in results_dict.items():
        benchmark_str = "|".join(benchmark.split("|")[:2])
        if benchmark != "all":
            metric = benchmarks_metrics[benchmark_str]
            final_results[benchmark] = v[metric]

    return final_results

def setup_wandb(config):
    run = wandb.init(
        project=config['wandb']['project'],
        name = config['wandb']['name'],
        resume="allow"
    )


def start_evaluation_loop(config, experiment_dir, additional_checkpoint_to_evaluate=[]):
    start_time = time.time()
    evaluated_checkpoints = []
    is_evaluated, latest_checkpoint_path = is_newest_checkpoint_evaluated(experiment_dir, evaluated_checkpoints)
    checkpoint_steps = get_checkpoint_steps(latest_checkpoint_path)

    for checkpoint in additional_checkpoint_to_evaluate:
        print(f"Evaluating checkpoint: {checkpoint}")
        evaluated_checkpoints.append(checkpoint)
        results_dir = run_lighteval(
            config["model_name"],
            checkpoint,
            cuda_devices="1,2",
            max_sampels=100
        )
        results = parse_results(results_dir)

        for benchmark, value in results.items():
            # Add the checkpoint step as a step for logging
            wandb.log({benchmark: value}, step=checkpoint_steps)
        print(f"Logged metrics for checkpoint {latest_checkpoint_path}")

    while True:
        # Check the newest checkpoint
        is_evaluated, latest_checkpoint_path = is_newest_checkpoint_evaluated(experiment_dir, evaluated_checkpoints)
        checkpoint_steps = get_checkpoint_steps(latest_checkpoint_path)

        # Evaluate if not done yet
        if not is_evaluated:
            print(f"Evaluating checkpoint: {latest_checkpoint_path}")
            evaluated_checkpoints.append(latest_checkpoint_path)
            results_dir = run_lighteval(
                config["model_name"],
                latest_checkpoint_path,
                cuda_devices="1,2",
                max_sampels=100
            )
            results = parse_results(results_dir)
            if results:
                for benchmark, value in results.items():
                    # Add the checkpoint step as a step for logging
                    wandb.log({benchmark: value}, step=checkpoint_steps)
                print(f"Logged metrics for checkpoint {latest_checkpoint_path}")

        # Stop if checkpoint steps is not divisible by 1000 
        # e.g. 2000 --> 4000 --> 6000 --> 6084 finish
        if checkpoint_steps % 1000 != 0:
            print(f"Checkpoint {checkpoint_steps} reached, exiting loop.")
            break

        up_time = (time.time() - start_time) / 3600
        if up_time >= 24:
            print("Reached 24 hours of uptime, stopping...")
            break

        # Wait 30 minutes
        print(f"Sleeping 30 minutes... Current uptime: {up_time:.2f}h")
        time.sleep(30 * 60)
