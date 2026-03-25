import wandb
import os
import sys
import re
import json
import traceback
from pathlib import Path

from oe_eval.launch import resolve_task_suite
from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from olmes_evaluator import evaluate_modalities_checkpoint

# --- CONFIGURATION MATCHING YOUR CLI ---
ENTITY = "behzadshomali"
PROJECT = "nemotron_MATH_PartialMTP_Gating"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/behzad_shomali/checkpoints/"
BENCHMARK_ROOT = "./benchmarks_v1" 
MAX_LENGTH = 2048

# 1. TASKS
TASKS_TO_RUN = [
    "modalities:base_easy:math_bpb",
    "modalities:base_easy:qa_rc",
    "modalities:base_easy:code_bpb",
    "modalities:base_easy:qa_bpb",
    "modalities::math_ac"  
]

# 2. LIMIT
EVAL_LIMIT = 128

# 3. BATCH SIZE
BATCH_SIZE = 4

# 4. MAPPING:
FOLDER_MAPPING = {
#     ## "folder_unique_identifier" : "wandb_run_id"
#     # "2026-02-11__10-36-48_abff36d0a7122c6b": "usbeeci5", # baseline
#     # "2026-02-11__16-28-52_b6df3e76ae591a56": "efniolfw"
#     # "2026-02-11__14-19-13_b6df3e76ae591a56": "5jh562ey"
#     # "2026-02-13__16-30-25_5c39a717dcaa5422": "oyc9aqld",
#     # "2026-02-13__16-26-16_93878c455150a246": "gwtf33ol",
#     # "2026-02-13__16-11-06_2519d6a77d994c97": "6lkwm5bu",
#     # "2026-02-13__16-10-36_6995ab4a053107ec": "ljbcigeo",
#     # "2026-02-15__19-31-28_6995ab4a053107ec": "v5zeu8gq"
#     "2026-02-17__17-44-01_304a053dc87afbb6": "0hpuqurm",
    # "2026-02-17__14-33-27_304a053dc87afbb6": "6lutihak"
    "2026-03-23__14-46-08_4ae0ab4f58ce6dac": "j1av1daf",
    "2026-03-23__14-44-54_74e7a6b65da9dd2f": "5mwyo6q2",
    "2026-03-23__14-06-54_542ec6512304bdd2": "0u8yej12",
    "2026-03-23__12-17-42_4b89ee4fb9772121": "ihb2jgsg",

}

WANDB_FOLDERS = [
    # "run-20260220_204436-0ylladgq"
]

def get_run_id_from_wandb_folder(folder_name):
    return folder_name.split("-")[-1]

def get_folder_for_run_id(run_name):
    # ..._recurEmbed=True_2026-02-15__19-31-28_6995ab4a053107ec --> 2026-02-15__19-31-28_6995ab4a053107ec
    folder_name = "_".join(run_name.split("_")[-4:])
    return folder_name

def get_run_id_for_folder(folder_name):
    for key, run_id in FOLDER_MAPPING.items():
        if key in folder_name:
            return run_id
    return None


def resolve_checkpoint_folder_path(folder_hint):
    folder_path = Path(folder_hint)
    if folder_path.exists():
        return folder_path

    rooted_path = Path(CHECKPOINTS_ROOT) / folder_hint
    if rooted_path.exists():
        return rooted_path

    return None


def build_eval_targets():
    """
    Build unified eval targets as tuples of:
    (run_id, checkpoint_folder_hint, source_label)
    """
    targets = []
    seen = set()

    for folder_name, run_id in FOLDER_MAPPING.items():
        if run_id and run_id not in seen:
            targets.append((run_id, folder_name, f"mapping:{folder_name}"))
            seen.add(run_id)

    for wandb_folder in WANDB_FOLDERS:
        run_id = get_run_id_from_wandb_folder(wandb_folder)
        if run_id and run_id not in seen:
            targets.append((run_id, None, f"wandb_folder:{wandb_folder}"))
            seen.add(run_id)

    return targets

def get_step_from_subfolder(subfolder_name):
    match = re.search(r"seen_steps_(\d+)", subfolder_name)
    return int(match.group(1)) if match else None

def parse_olmes_results(results_dict):
    """Parses OLMES results into a flat WandB dictionary."""
    wandb_metrics = {}
    
    # Handle case where input is the raw list (from JSON file) vs wrapper dict (from function)
    if isinstance(results_dict, list):
        all_metrics = results_dict
    else:
        all_metrics = results_dict.get("metrics", [])
    
    for item in all_metrics:
        # Get Alias
        task_name = item.get("task_config", {}).get("metadata", {}).get("alias")
        if not task_name:
            task_name = item.get("task_name", "unknown")
            
        # Get Score
        score = item.get("metrics", {}).get("primary_score")
        if score is not None:
            wandb_metrics[f"eval/{task_name}"] = score
            
    return wandb_metrics


def _resolve_suite_tasks(suite_name):
    task_suite_parent = {}
    try:
        return resolve_task_suite(suite_name, task_suite_parent)
    except Exception:
        return []


def build_summary_metrics(metrics_to_log):
    """Create per-suite summaries and an overall average across suites."""
    summary_metrics = {}
    eval_items = {
        key: value
        for key, value in metrics_to_log.items()
        if key.startswith("eval/") and isinstance(value, (int, float))
    }

    suite_values = []
    for suite_name in TASKS_TO_RUN:
        resolved_tasks = []
        if suite_name in TASK_SUITE_CONFIGS:
            resolved_tasks = _resolve_suite_tasks(suite_name)
        else:
            resolved_tasks = [suite_name]

        # Average over available task metrics for this suite
        scores = []
        for task in resolved_tasks:
            key = f"eval/{task}"
            if key in eval_items:
                scores.append(eval_items[key])

        if scores:
            suite_score = sum(scores) / len(scores)
            summary_metrics[f"summary/{suite_name}"] = suite_score
            suite_values.append(suite_score)

    if suite_values:
        summary_metrics["summary/avg_all_suites"] = sum(suite_values) / len(suite_values)

    return summary_metrics

def get_checkpoints_for_folder(folder_path):
    # 1. Config & Checkpoints
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    config_path = None
    yaml_files = list(folder_path.glob("*.yaml"))
    if yaml_files: config_path = str(yaml_files[0])

    checkpoints = []
    for item in folder_path.iterdir():
        if item.is_dir() and "seen_steps" in item.name:
            step = get_step_from_subfolder(item.name)
            if step is not None:
                checkpoints.append((step, item))
    checkpoints.sort(key=lambda x: x[0])
    return config_path, checkpoints

def evaluate_run_target(run_id, checkpoint_folder_hint=None, source_label=None):
    # 2. INITIALIZE WANDB (With Custom X-Axis Fix)
    try:
        run = wandb.init(id=run_id, project=PROJECT, entity=ENTITY, resume="must", reinit=True)
        print(f"\n=== Processing run_id={run_id} ({source_label or 'unknown_source'}) ===")
        print(f"    -> CONNECTED: {wandb.run.url}")
        
        run_name = run.name
        if checkpoint_folder_hint:
            checkpoint_folder_path = resolve_checkpoint_folder_path(checkpoint_folder_hint)
        else:
            inferred_folder_name = get_folder_for_run_id(run_name)
            checkpoint_folder_path = resolve_checkpoint_folder_path(inferred_folder_name)

        if checkpoint_folder_path is None:
            print("    !!! Could not resolve checkpoint folder. Skipping run.")
            wandb.finish()
            return

        config_path, checkpoints = get_checkpoints_for_folder(checkpoint_folder_path)
        if not checkpoints:
            print("    !!! No valid checkpoints found in this folder. Skipping.")
            wandb.finish()
            return
        
        # Define X-Axis
        wandb.define_metric("seen_steps")
        wandb.define_metric("eval/*", step_metric="seen_steps")
        wandb.define_metric("summary/*", step_metric="seen_steps")
    except Exception as e:
        print(f"    !!! CRITICAL ERROR: {e}")
        return

    # 3. Loop Steps
    summary_table = None
    summary_columns = None

    for step, ckpt_path in checkpoints:
        
        # --- THE CACHE CHECK ---
        # Look for: ./benchmarks_v1/RUN_ID/step_STEP/all_results.json
        expected_json = Path(BENCHMARK_ROOT) / run_id / f"step_{step}" / "all_results.json"
        
        eval_output = None
        
        if expected_json.exists():
            print(f"    >> [CACHE HIT] Found results at {expected_json}")
            try:
                with open(expected_json, 'r') as f:
                    # Your file is a list [...], we load it directly
                    eval_output = {"metrics": json.load(f)}
            except Exception as e:
                print(f"       Error reading JSON: {e}. Will re-run eval.")
        
        # If no cache or error reading cache, run the eval
        if eval_output is None:
            print(f"    >> [COMPUTING] Running Eval on Step {step}...")
            try:
                eval_output = evaluate_modalities_checkpoint(
                    checkpoint_path=str(ckpt_path),
                    config_path=config_path, 
                    tasks=TASKS_TO_RUN,          
                    limit=EVAL_LIMIT,            
                    batch_size=BATCH_SIZE,       
                    output_dir=f"{BENCHMARK_ROOT}/{run_id}/step_{step}",
                    max_length=MAX_LENGTH
                )
            except Exception as e:
                print(f"       ERROR executing eval on step {step}: {e}")
                traceback.print_exc()
                continue

        # # --- LOGGING ---
        # if eval_output:
            metrics_to_log = parse_olmes_results(eval_output)
            
            if metrics_to_log:
                summary_metrics = build_summary_metrics(metrics_to_log)
                metrics_to_log.update(summary_metrics)

                # Build or append to per-step summary table for dashboard visibility
                if summary_table is None:
                    summary_columns = ["seen_steps"] + sorted(summary_metrics.keys())
                    summary_table = wandb.Table(columns=summary_columns)

                row = [step] + [summary_metrics.get(col, None) for col in summary_columns[1:]]
                summary_table.add_data(*row)

                # Add the custom X-axis step
                metrics_to_log["seen_steps"] = step
                
                # Log without 'step=' argument
                wandb.log(metrics_to_log)
                print(f"       Logged {len(metrics_to_log)} metrics (seen_steps={step}).")
            else:
                print("       Warning: No valid metrics found to log.")

    # Emit summary artifacts once per run
    if summary_table is not None:
        wandb.log({"summary/table": summary_table})
        if "summary/avg_all_suites" in summary_columns:
            last_avg = summary_table.data[-1][summary_columns.index("summary/avg_all_suites")]
            wandb.summary["summary/avg_all_suites_last"] = last_avg

    wandb.finish()

if __name__ == "__main__":
    # Ensure benchmark root exists
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)

    targets = build_eval_targets()
    if not targets:
        print("No evaluation targets found. Populate FOLDER_MAPPING and/or WANDB_FOLDERS.")
        sys.exit(0)

    for run_id, checkpoint_folder_hint, source_label in targets:
        evaluate_run_target(
            run_id=run_id,
            checkpoint_folder_hint=checkpoint_folder_hint,
            source_label=source_label,
        )