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
PROJECTS = [
    "nemotron_MATH_PartialMTP_Gating",
    "nemotron_MATH_PartialMTP_Gating_CC"
]

CHECKPOINTS_ROOT = os.environ["CHECKPOINT_SAVING_PATH"]
BENCHMARK_ROOT = os.environ["BENCHMARK_ROOT"]
MAX_LENGTH = 2048

WANDB_FOLDER_PREFIX = os.environ.get("WANDB_PATH_M", ".") + "/wandb/"


# 1. TASKS
TASKS_TO_RUN = [
    "modalities:base_easy:math_bpb",
    "modalities:base_easy:qa_rc",
    "modalities:base_easy:code_bpb",
    "modalities:base_easy:qa_bpb",
]

# 2. LIMIT
EVAL_LIMIT = 128

# 3. BATCH SIZE
BATCH_SIZE = 8

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
}

WANDB_FOLDERS = [
    "offline-run-20251209_172903-9zwkeq9p",
    "offline-run-20251209_173053-5kgi0001",
    "offline-run-20251212_032643-429minjf",
    "offline-run-20251212_032850-25ox9ltu",
    "offline-run-20251212_033104-zsjzl6nm",
    "offline-run-20251212_033309-x2jn4g40",
    "offline-run-20251212_033523-wwr22kjd",
    "offline-run-20251212_033735-vwrg1p6j",
    "offline-run-20251212_033930-3vipiodv",
    "offline-run-20251212_034128-9hrw5y59",
    "offline-run-20251212_092310-1hflaxjj",
    "offline-run-20251212_092326-v13vokia",
    "offline-run-20251212_102455-kj8un5ks",
    "offline-run-20251212_102648-4xpz59c4",
    "offline-run-20251212_103018-j02bhs5k",
    "offline-run-20251212_123357-qm4xcurn",
    "offline-run-20251212_133226-y8eogb7o",
    "offline-run-20251212_134835-7dwfpx3h",
    "offline-run-20251212_102943-u7fhg45i",
    "offline-run-20251212_102944-2s6muk92",
    "offline-run-20251212_102946-qh05cezz",
    "offline-run-20251212_102954-ottank38",
    "offline-run-20251212_102954-ssu01sxe",
    "offline-run-20251212_103000-n788gmb4",
    "offline-run-20251213_210146-yhuiag5r",
    "offline-run-20251213_210737-bq3lcpsa",
    "offline-run-20251213_211948-exch5oh9",
    "offline-run-20251213_215420-y1mrebnq",
    "offline-run-20251213_220557-7maya25e",
    "offline-run-20251213_224444-jzh0qmti",
    "offline-run-20251214_092526-v8wlbbzp",
    "offline-run-20251214_101739-oxv8tpfd",
    "offline-run-20251222_151412-r28owd14",
    "offline-run-20251222_151800-hoywn93z",
    "offline-run-20251222_160027-ijcbkegk",
    "offline-run-20251222_175236-t6yjkzfh",
    "offline-run-20251223_154209-36qh0a0e",
    "offline-run-20251223_154209-7eluclma",
    "offline-run-20251225_120049-q2zybeok",
    "offline-run-20251225_120207-3kyzvemt",
    "offline-run-20251225_120337-1k90okau",
    "offline-run-20251225_183420-txsajiiz",
    "offline-run-20251226_020020-l2ki1x9g",
    "offline-run-20251226_020221-pz6nquqx",
    "offline-run-20251227_135703-ner6yfa4",
    "offline-run-20251227_200543-evwjfxpt",
    "offline-run-20251227_135841-h9wchg74",
    "offline-run-20260209_210745-9xlz03q6",
    "offline-run-20260209_211322-c2ul3bhi",
    "offline-run-20260209_211920-cd2j4orp",
    "offline-run-20260209_214009-i2ycodxc",
    "offline-run-20260209_214355-asmlmvs4",
    "offline-run-20260209_214938-27ior06i",
    "offline-run-20260209_221050-zlfidryl",
    "offline-run-20260209_221433-slvgwaie",
    "offline-run-20260209_222046-tgjk8xut",
    "offline-run-20260209_222843-awnbpwo3",
    "offline-run-20260209_224035-1meo2vpc",
    "offline-run-20260209_224448-4dury8c0",
    "offline-run-20260209_225418-j23td2h7",
    "offline-run-20260209_225915-fmtfzttj",
    "offline-run-20260209_231147-gmcz125d",
    "offline-run-20260209_231509-d69izd11",
    "offline-run-20260209_231607-d82neh5h",
    "offline-run-20260209_232137-zghxwm61",
    "offline-run-20260209_232952-hssn105u",
    "offline-run-20260209_233146-c8h4o18b",
    "offline-run-20260209_234202-7crt6xrj",
    "offline-run-20260209_234503-qltowari",
    "offline-run-20260209_234556-qqhyay9d",
    "offline-run-20260209_235211-19oq3ktp",
    "offline-run-20260209_235915-gech1qrr",
    "offline-run-20260210_100852-1gi7eh5f",
    "offline-run-20260210_100852-33ww3bo8",
    "offline-run-20260210_100852-ar6y41gd",
    "offline-run-20260210_100852-kmzorq3o",
    "offline-run-20260210_100852-lghtna7m",
    "offline-run-20260210_100852-u9vpaap6",
    "offline-run-20260210_101347-z9io7kr5",
    "offline-run-20260210_135626-hwu2820j",
    "offline-run-20260210_143411-da53e12z",
    "offline-run-20260210_153824-frm0pbig",
    "offline-run-20260210_163829-uzv0wfa2",
    "offline-run-20260210_164455-p05mi5mf",
    "offline-run-20260210_164455-8roo6h16",
    "offline-run-20260210_165158-qmmi1plm",
    "offline-run-20260210_170125-3h2tne4t",
    "offline-run-20260210_170715-5wiq5r2q",
    "offline-run-20260210_171827-egbkekns",
    "offline-run-20260213_115335-qegxaw75",
    "offline-run-20260213_120235-gjoal7wu",
    "offline-run-20260213_120445-sqq9ol97",
    "offline-run-20260213_120601-6wodkjp0",
    "offline-run-20260213_120729-szvglis0",
    "offline-run-20260213_125453-pto0z0mt",
    "offline-run-20260213_154136-qdynxmb3",
    "offline-run-20260214_002235-5t15ufsh",
    "offline-run-20260213_231441-43fsv06a",
    "offline-run-20260214_002632-8shnybjn",
    "offline-run-20260214_005523-cokvpuym",
    "offline-run-20260214_012127-8aclcvi1",
    "offline-run-20260214_030734-t2ldhib7",
    "offline-run-20260214_034552-vt5ybv4r",
    "offline-run-20260214_041238-u1zpsc4s",
    "offline-run-20260214_044521-np6kr41d",
    "offline-run-20260214_095822-b5esvz21",
    "offline-run-20260214_105042-u64rkftd",
    "offline-run-20260214_113704-rm7s9mhq",
    "offline-run-20260219_153732-xfrlqom6",
    "offline-run-20260219_153732-6nj67my1",
    "offline-run-20260219_153732-39x9n8r2",
    "offline-run-20260219_153746-5cvrjn31",
    "offline-run-20260219_153843-8go2rm43",
    "offline-run-20260221_134715-zzzi1hoy",
    "offline-run-20260221_134715-zmmu87lb",
    "offline-run-20260221_134715-zm0oakz8",
    "offline-run-20260221_134715-yvw1d8py",
    "offline-run-20260221_134715-xqb73wvh",
    "offline-run-20260221_134715-uk81ebfj",
    "offline-run-20260221_134715-u4glkvq9",
    "offline-run-20260221_134715-r7zvhmkt",
    "offline-run-20260221_134715-pp1r5pns",
    "offline-run-20260221_134715-o9z6sx5o",
    "offline-run-20260221_134715-o25gglvs",
    "offline-run-20260221_134715-ni1rxeht",
    "offline-run-20260221_134715-nahij4n1",
    "offline-run-20260221_134715-mx15edza",
    "offline-run-20260221_134715-mmt0dv9a",
    "offline-run-20260221_134715-m3ouqebx",
    "offline-run-20260221_134715-lmwxty6g",
    "offline-run-20260221_134715-kz7kmkd6",
    "offline-run-20260221_134715-jd87w6dl",
    "offline-run-20260221_134715-cv3obc06",
    "offline-run-20260221_134715-bkvpqb77",
    "offline-run-20260221_134715-9x7braem",
    "offline-run-20260221_134715-90kwm5e8",
    "offline-run-20260221_134715-8vmfq2xl",
    "offline-run-20260221_134715-81b3baxb",
    "offline-run-20260221_134715-7um5917c",
    "offline-run-20260221_134715-3ncg0jmi",
    "offline-run-20260221_134715-31pobw8w"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_run_id_from_wandb_folder(folder_name):
    """Extract the wandb run-id (last dash-separated token) from a folder name."""
    return folder_name.split("-")[-1]


def get_folder_for_run_id(run_name):
    """
    Extract the checkpoint-folder suffix from a wandb run name.
    e.g. '..._recurEmbed=True_2026-02-15__19-31-28_6995ab4a053107ec'
         -> '2026-02-15__19-31-28_6995ab4a053107ec'
    """
    return "_".join(run_name.split("_")[-4:])


def extract_checkpoint_folder_from_wandb_log(wandb_folder_path):
    """
    Extract the checkpoint folder name (YYYY-MM-DD__HH-MM-SS_<experiment_id>)
    from a wandb offline run's binary log file by regex-searching the raw bytes.
    """
    wandb_dir = Path(wandb_folder_path)
    wandb_files = list(wandb_dir.glob("run-*.wandb"))
    if not wandb_files:
        return None

    logfile = wandb_files[0]
    try:
        with open(logfile, "rb") as f:
            data = f.read(200_000)
        text = data.decode("utf-8", errors="replace")
        matches = set(re.findall(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}_[a-f0-9]{16}", text))

        for m in sorted(matches):
            candidate = Path(CHECKPOINTS_ROOT) / m
            if candidate.is_dir():
                return m

        if matches:
            return sorted(matches)[0]
    except Exception as e:
        print(f"    Warning: Could not extract checkpoint folder from {logfile}: {e}")

    return None


def resolve_checkpoint_folder_path(folder_hint):
    if folder_hint is None:
        return None
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
      (run_id, checkpoint_folder_hint, wandb_offline_folder_path, source_label)

    wandb_offline_folder_path is the EXISTING offline folder for this run so we
    can point WANDB_DIR at it and append eval metrics there.
    """
    targets = []
    seen = set()

    # 1. Targets coming from FOLDER_MAPPING (no pre-existing offline folder known)
    for folder_name, run_id in FOLDER_MAPPING.items():
        if run_id and run_id not in seen:
            targets.append((run_id, folder_name, None, f"mapping:{folder_name}"))
            seen.add(run_id)

    # 2. Targets coming from WANDB_FOLDERS (existing offline run folder known)
    for wandb_folder in WANDB_FOLDERS:
        wandb_folder_path = WANDB_FOLDER_PREFIX + wandb_folder
        run_id = get_run_id_from_wandb_folder(wandb_folder_path)
        if run_id and run_id not in seen:
            ckpt_folder = extract_checkpoint_folder_from_wandb_log(wandb_folder_path)
            if ckpt_folder:
                print(f"    [wandb_folder] run_id={run_id} -> checkpoint folder: {ckpt_folder}")
            else:
                print(f"    [wandb_folder] run_id={run_id} -> WARNING: could not extract checkpoint folder")
            targets.append((run_id, ckpt_folder, wandb_folder_path, f"wandb_folder:{wandb_folder_path}"))
            seen.add(run_id)

    return targets


def get_step_from_subfolder(subfolder_name):
    match = re.search(r"seen_steps_(\d+)", subfolder_name)
    return int(match.group(1)) if match else None


def parse_olmes_results(results_dict):
    """Parse OLMES results into a flat WandB metric dictionary."""
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
        if suite_name in TASK_SUITE_CONFIGS:
            resolved_tasks = _resolve_suite_tasks(suite_name)
        else:
            resolved_tasks = [suite_name]

        scores = [eval_items[f"eval/{task}"] for task in resolved_tasks if f"eval/{task}" in eval_items]

        if scores:
            suite_score = sum(scores) / len(scores)
            summary_metrics[f"summary/{suite_name}"] = suite_score
            suite_values.append(suite_score)

    if suite_values:
        summary_metrics["summary/avg_all_suites"] = sum(suite_values) / len(suite_values)

    return summary_metrics


def get_checkpoints_for_folder(folder_path):
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
    checkpoints.sort(key=lambda x: x[0])
    return config_path, checkpoints


# ---------------------------------------------------------------------------
# Core: initialise wandb so it writes INTO the existing offline folder
# ---------------------------------------------------------------------------

def _wandb_dir_for_offline_folder(wandb_offline_folder_path):
    """
    wandb expects WANDB_DIR to be the *parent* of the 'wandb/' sub-directory.
    The offline folder is something like:
        /some/path/wandb/offline-run-TIMESTAMP-RUNID/
    So the parent of 'wandb/' is:
        /some/path/
    """
    folder = Path(wandb_offline_folder_path).resolve()
    # Walk upward until we find the directory named 'wandb'
    for parent in folder.parents:
        if parent.name == "wandb":
            # parent.parent is the directory that *contains* the 'wandb/' folder
            return str(parent.parent)
    # Fallback: two levels up (offline-run-X is inside wandb/ which is inside WANDB_DIR)
    return str(folder.parent.parent)


def init_wandb_offline(run_id, project, entity, wandb_offline_folder_path=None):
    """
    Initialise wandb in offline mode, writing logs into the same folder tree as
    the existing offline run so that `wandb sync --all` on WANDB_FOLDER_PREFIX
    picks them up together.

    Returns the wandb Run object, or None on failure.
    """
    # Save original WANDB_DIR so we can restore it afterwards
    original_wandb_dir = os.environ.get("WANDB_DIR")

    if wandb_offline_folder_path:
        target_wandb_dir = _wandb_dir_for_offline_folder(wandb_offline_folder_path)
        os.environ["WANDB_DIR"] = target_wandb_dir
        print(f"    -> Setting WANDB_DIR={target_wandb_dir} (maps to existing offline folder)")
    else:
        # No existing folder known; fall back to WANDB_FOLDER_PREFIX parent
        fallback_dir = str(Path(WANDB_FOLDER_PREFIX).parent)
        os.environ["WANDB_DIR"] = fallback_dir
        print(f"    -> No existing offline folder; WANDB_DIR={fallback_dir}")

    try:
        run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            reinit=True,
            resume="allow",   # 'allow' works offline; 'must' requires internet to verify
            mode="offline",
        )
        return run
    except Exception as e:
        print(f"    !!! wandb.init failed: {e}")
        return None
    finally:
        # Restore WANDB_DIR so subsequent runs are not affected
        if original_wandb_dir is None:
            os.environ.pop("WANDB_DIR", None)
        else:
            os.environ["WANDB_DIR"] = original_wandb_dir


# ---------------------------------------------------------------------------
# Main evaluation loop for a single run
# ---------------------------------------------------------------------------

def evaluate_run_target(run_id, checkpoint_folder_hint=None, wandb_offline_folder_path=None, source_label=None):
    print(f"\n=== Processing run_id={run_id} ({source_label or 'unknown_source'}) ===")

    # --- Resolve checkpoint folder ---
    if checkpoint_folder_hint:
        checkpoint_folder_path = resolve_checkpoint_folder_path(checkpoint_folder_hint)
    else:
        checkpoint_folder_path = None

    if checkpoint_folder_path is None:
        print("    !!! Could not resolve checkpoint folder. Skipping run.")
        return

    config_path, checkpoints = get_checkpoints_for_folder(checkpoint_folder_path)

    if not checkpoints:
        print("    !!! No valid checkpoints found in this folder. Skipping.")
        return

    # --- Initialise wandb (always offline, writing into the existing folder) ---
    run = None
    for project in PROJECTS:
        print(f"    Trying project '{project}'...")
        run = init_wandb_offline(run_id, project, ENTITY, wandb_offline_folder_path)
        if run is not None:
            print(f"    -> Offline run initialised. Files will appear under WANDB_DIR.")
            break

    if run is None:
        print("    !!! Could not initialise wandb for any project. Skipping run.")
        return

    # Define custom X-axis
    wandb.define_metric("seen_steps")
    wandb.define_metric("eval/*", step_metric="seen_steps")
    wandb.define_metric("summary/*", step_metric="seen_steps")

    # --- Loop over checkpoints ---
    summary_table = None
    summary_columns = None

    for step, ckpt_path in checkpoints:

        # Cache check
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
            print(f"    >> [COMPUTING] Running Eval on Step {step}...")
            try:
                eval_output = evaluate_modalities_checkpoint(
                    checkpoint_path=str(ckpt_path),
                    config_path=config_path,
                    tasks=TASKS_TO_RUN,
                    limit=EVAL_LIMIT,
                    batch_size=BATCH_SIZE,
                    output_dir=f"{BENCHMARK_ROOT}/{run_id}/step_{step}",
                    max_length=MAX_LENGTH,
                )
            except Exception as e:
                print(f"       ERROR executing eval on step {step}: {e}")
                traceback.print_exc()
                continue

        if eval_output:
            metrics_to_log = parse_olmes_results(eval_output)

            if metrics_to_log:
                summary_metrics = build_summary_metrics(metrics_to_log)
                metrics_to_log.update(summary_metrics)

                if summary_table is None:
                    summary_columns = ["seen_steps"] + sorted(summary_metrics.keys())
                    summary_table = wandb.Table(columns=summary_columns)

                row = [step] + [summary_metrics.get(col, None) for col in summary_columns[1:]]
                summary_table.add_data(*row)

                metrics_to_log["seen_steps"] = step
                wandb.log(metrics_to_log)
                print(f"       Logged {len(metrics_to_log)} metrics (seen_steps={step}).")
            else:
                print("       Warning: No valid metrics found to log.")

    # Emit summary artifacts once per run
    if summary_table is not None:
        wandb.log({"summary/table": summary_table})
        if summary_columns and "summary/avg_all_suites" in summary_columns:
            last_avg = summary_table.data[-1][summary_columns.index("summary/avg_all_suites")]
            wandb.summary["summary/avg_all_suites_last"] = last_avg

    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)

    targets = build_eval_targets()
    if not targets:
        print("No evaluation targets found. Populate FOLDER_MAPPING and/or WANDB_FOLDERS.")
        sys.exit(0)

    for run_id, checkpoint_folder_hint, wandb_offline_folder_path, source_label in targets:
        evaluate_run_target(
            run_id=run_id,
            checkpoint_folder_hint=checkpoint_folder_hint,
            wandb_offline_folder_path=wandb_offline_folder_path,
            source_label=source_label,
        )