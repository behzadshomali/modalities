"""Evaluate the LATEST checkpoint in a given directory on the FULL modalities::math_ac benchmark."""

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

# --- CONFIGURATION ---
ENTITY = "behzadshomali"
PROJECT = "nemotron_MATH_PartialMTP_Gating"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/behzad_shomali/checkpoints/"
BENCHMARK_ROOT = "./benchmarks_full_v1"
MAX_LENGTH = 2048

# Only math_ac, full evaluation (no limit)
TASKS_TO_RUN = ["modalities::math_ac"]
EVAL_LIMIT = None  # No limit → evaluate on full dataset
BATCH_SIZE = 18

# MAPPING: folder_unique_identifier → wandb_run_id
FOLDER_MAPPING = {
    # "2026-04-12__15-10-12_5ad02f0f6debfb5f": "60nd14fb",
    # "2026-04-11__13-08-00_dcfa3730754d817f": "4hp3xq5w",
    # "2026-04-10__13-58-22_2d326a1b8e55ad77": "eds4wahr",
    # "2026-04-08__13-41-08_c16ee3bff9052caa": "gywl97vs",
    # "2026-04-07__16-09-41_9d97f3dbe24067f0": "z7ul6eva",
    # "2026-04-03__11-08-19_abff36d0a7122c6b": "0s760ixs",
    # "2026-02-11__10-36-48_abff36d0a7122c6b": "usbeeci5"
    # "2026-04-14__10-15-10_8d7de5d6449e217e": "0rta3we6",
    # "2026-04-14__12-58-01_4209bf8f68a75e14": "qf25ei5g"
    "2026-04-15__16-33-46_4209bf8f68a75e14": "oe9tbd86",
    "2026-04-13__21-19-16_8d7de5d6449e217e": "fd57jnzj"
}


def get_step_from_subfolder(subfolder_name):
    match = re.search(r"seen_steps_(\d+)", subfolder_name)
    return int(match.group(1)) if match else None


def resolve_checkpoint_folder_path(folder_hint):
    folder_path = Path(folder_hint)
    if folder_path.exists():
        return folder_path
    rooted_path = Path(CHECKPOINTS_ROOT) / folder_hint
    if rooted_path.exists():
        return rooted_path
    return None


def get_latest_checkpoint(folder_path):
    """Return (config_path, (step, ckpt_path)) for the latest checkpoint only."""
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    config_path = None
    yaml_files = list(folder_path.glob("*.yaml"))
    if yaml_files:
        config_path = str(yaml_files[0])

    checkpoints = []
    for item in folder_path.iterdir():
        if item.is_dir() and "seen_steps" in item.name:
            step = get_step_from_subfolder(item.name)
            if step is not None:
                checkpoints.append((step, item))

    if not checkpoints:
        return config_path, None

    # Return only the latest (highest step)
    latest = max(checkpoints, key=lambda x: x[0])
    return config_path, latest


def parse_olmes_results(results_dict):
    wandb_metrics = {}
    if isinstance(results_dict, list):
        all_metrics = results_dict
    else:
        all_metrics = results_dict.get("metrics", [])

    for item in all_metrics:
        task_name = item.get("task_config", {}).get("metadata", {}).get("alias")
        if not task_name:
            task_name = item.get("task_name", "unknown")
        score = item.get("metrics", {}).get("primary_score")
        if score is not None:
            wandb_metrics[f"eval_full/{task_name}"] = score
    return wandb_metrics


def evaluate_run_target(run_id, checkpoint_folder_hint, source_label=None):
    try:
        run = wandb.init(id=run_id, project=PROJECT, entity=ENTITY, resume="must", reinit=True)
        print(f"\n=== Processing run_id={run_id} ({source_label or 'unknown_source'}) ===")
        print(f"    -> CONNECTED: {wandb.run.url}")

        checkpoint_folder_path = resolve_checkpoint_folder_path(checkpoint_folder_hint)
        if checkpoint_folder_path is None:
            print("    !!! Could not resolve checkpoint folder. Skipping run.")
            wandb.finish()
            return

        config_path, latest = get_latest_checkpoint(checkpoint_folder_path)
        if latest is None:
            print("    !!! No valid checkpoints found. Skipping.")
            wandb.finish()
            return

        step, ckpt_path = latest
        print(f"    -> Latest checkpoint: step {step} at {ckpt_path}")

        wandb.define_metric("seen_steps_full")
        wandb.define_metric("eval_full/*", step_metric="seen_steps_full")
    except Exception as e:
        print(f"    !!! CRITICAL ERROR: {e}")
        return

    # --- CACHE CHECK ---
    expected_json = Path(BENCHMARK_ROOT) / run_id / f"step_{step}" / "all_results.json"
    eval_output = None

    if expected_json.exists():
        print(f"    >> [CACHE HIT] Found results at {expected_json}")
        try:
            with open(expected_json, "r") as f:
                eval_output = {"metrics": json.load(f)}
        except Exception as e:
            print(f"       Error reading JSON: {e}. Will re-run eval.")

    if eval_output is None:
        print(f"    >> [COMPUTING] Running FULL math_ac eval on step {step}...")
        try:
            eval_output = evaluate_modalities_checkpoint(
                checkpoint_path=str(ckpt_path),
                config_path=config_path,
                tasks=TASKS_TO_RUN,
                limit=EVAL_LIMIT,       # None → full dataset
                batch_size=BATCH_SIZE,
                output_dir=f"{BENCHMARK_ROOT}/{run_id}/step_{step}",
                max_length=MAX_LENGTH,
            )
        except Exception as e:
            print(f"       ERROR executing eval on step {step}: {e}")
            traceback.print_exc()
            wandb.finish()
            return

    # --- LOGGING ---
    metrics_to_log = parse_olmes_results(eval_output)

    if metrics_to_log:
        metrics_to_log["seen_steps_full"] = step
        wandb.log(metrics_to_log)
        print(f"       Logged {len(metrics_to_log)} metrics (seen_steps={step}).")
    else:
        print("       Warning: No valid metrics found to log.")

    wandb.finish()


if __name__ == "__main__":
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)

    if not FOLDER_MAPPING:
        print("No evaluation targets found. Populate FOLDER_MAPPING.")
        sys.exit(0)

    for folder_name, run_id in FOLDER_MAPPING.items():
        evaluate_run_target(
            run_id=run_id,
            checkpoint_folder_hint=folder_name,
            source_label=f"mapping:{folder_name}",
        )
