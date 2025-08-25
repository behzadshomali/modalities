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

def load_config(config_path):
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
    
    sft_args = args['sft']
    now = datetime.now()
    dir_name = now.strftime("%Y_%m_%d-%H_%M_%S")

    if "wandb" in args:
        project_name = args['wandb']['project']
    else:
        project_name = ""

    output_dir = os.path.join(sft_args['output_dir'], project_name, dir_name)
    sft_args['output_dir'] = output_dir
    sft_args['learning_rate'] = float(sft_args['learning_rate'])

    args['sft'] = sft_args
    
    return args

def preprocess_function(example, instruction_col, response_col):
    return {
        "messages": [
            {"role": "user", "content": example[instruction_col]},
            {"role": "assistant", "content": example[response_col]}
        ]
    }

async def lighteval_async(checkpoint_path, cuda_devices="3,4"):
    output_dir = "/raid/s3/opengptx/behzad_shomali/evaluation_results/sft_intermediate_results/"
    multi_gpu_command = "--multi_gpu" if len(cuda_devices.split(',')) > 1 else ""

    command = f"""\
CUDA_VISIBLE_DEVICES={cuda_devices} accelerate launch \
    --main_process_port 2000 \
    --num_processes {len(cuda_devices.split(','))} \
    -m \
    lighteval accelerate \
    "model_name={checkpoint_path},trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0" \
    "--max-samples 100" \
    --output-dir {output_dir} \
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


def preprocess_coda_alpaca(raw_data):
    final_data = []
    for row in raw_data:
        if len(row['input']) == 0:
            final_data.append({
                "instruction": row["instruction"],
                "output": row["output"]
            })

    print(f"Kept {len(final_data)/len(raw_data)} from code alpaca!")
    return final_data