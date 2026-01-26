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
PROJECT = "nemotron_MATH_PartialMTP"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/behzad_shomali/checkpoints/"
BENCHMARK_ROOT = "./benchmarks_v1" 
MAX_LENGTH = 2048

# 1. TASKS
TASKS_TO_RUN = [
    "modalities:base_easy:math_bpb",
    "modalities:base_easy:qa_rc",
    "modalities:base_easy:code_bpb",
    "modalities:base_easy:qa_bpb",
    "modalities:base_easy:math_ac",
    "basic_skills:rc::olmes:modalities",
    "basic_skills:rc:bpb::olmes:modalities"
]

# 2. LIMIT
EVAL_LIMIT = 64

# 3. BATCH SIZE
BATCH_SIZE = 2

# 4. MAPPING:
FOLDER_MAPPING = {
    ## "folder_unique_identifier" : "wandb_run_id"
    # "2026-01-14__17-23-38_7f85baa58f096684": "9hyrlxu2",
    # "2026-01-15__15-22-41_ae159b022daf96cb": "xp0999b0", # Increase hidden baseline

    # "2026-01-12__15-52-31_d5de661a639f354a": "w38slo0n", # 257M_GradProj_MTPT
    # "2026-01-04__12-48-20_7669e5288dc2f8f3": "83ar3gdh",

    "2026-01-22__17-52-12_3a5b61a2043d85a1": "urh3ro9v", # best so far small blocks grad proj MTP 2
    "2026-01-22__15-57-43_a7f9a98973c6c283": "q5nfhpr1", # MTP 2 wo/ grad proj
    "2026-01-22__15-57-08_e9056200d931eab1": "nw7hdvf5", # MTP 2 grad proj
    "2026-01-22__18-07-34_bbc0148afae6b0fa": "kralywlh", # baseline,
    "2026-01-23__10-31-03_0eaacff18025714c": "042nlf1g", # MTP 2 wo/ grad proj small blocks
}

def get_run_id_for_folder(folder_name):
    for key, run_id in FOLDER_MAPPING.items():
        if key in folder_name:
            return run_id
    return None

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

def evaluate_model_folder(model_folder_path):
    folder_name = model_folder_path.name
    run_id = get_run_id_for_folder(folder_name)
    
    if not run_id:
        return

    print(f"\n=== Processing: {folder_name} ===")
    print(f"    -> Mapped to WandB Run ID: {run_id}")

    # 1. Config & Checkpoints
    config_path = None
    yaml_files = list(model_folder_path.glob("*.yaml"))
    if yaml_files: config_path = str(yaml_files[0])

    checkpoints = []
    for item in model_folder_path.iterdir():
        if item.is_dir() and "seen_steps" in item.name:
            step = get_step_from_subfolder(item.name)
            if step is not None:
                checkpoints.append((step, item))
    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints: return

    # 2. INITIALIZE WANDB (With Custom X-Axis Fix)
    try:
        wandb.init(id=run_id, project=PROJECT, entity=ENTITY, resume="must", reinit=True)
        print(f"    -> CONNECTED: {wandb.run.url}") 
        
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

        # --- LOGGING ---
        if eval_output:
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
    
    root = Path(CHECKPOINTS_ROOT)
    for item in root.iterdir():
        if item.is_dir():
            evaluate_model_folder(item)