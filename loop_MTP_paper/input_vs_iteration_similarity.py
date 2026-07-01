#!/usr/bin/env python
"""Per-iteration input-vs-output similarity — MTP vs. non-MTP.

For each recurrence iteration ``r``, measure the cosine similarity between
**that iteration's own input** and **its output**. The recurrence chains raw
hidden states, so the input to iteration ``r`` is the block input for ``r == 0``
and the previous iteration's output (``each_recurrence_hidden_states[r-1]``)
otherwise; the output is ``each_recurrence_hidden_states[r]``. This measures how
much each step transforms its input (closer to 1 = smaller change).

The analysis runs for two models — one trained *with* MTP and one *without* — and
overlays their per-iteration curves on the same plot. Models are processed one at
a time (loaded, measured, freed) so only one is resident in memory at a moment.

The expensive measurement is cached per model under ``CACHE_DIR``. On re-run the
cached DataFrames are loaded instead of recomputing, so you can freely edit the
plotting section (or pass ``--plot-only``) and get an instant turnaround.

Usage
-----
    python input_vs_iteration_similarity.py                # compute (using cache) then plot
    python input_vs_iteration_similarity.py --recompute    # ignore cache, recompute all
    python input_vs_iteration_similarity.py --plot-only     # only plot from cache (no model load)

* Tokens: a flat uint32 GPT-2 token stream (``TOKEN_CACHE_FILE``).
* Model dynamics: ``src/modalities/models/gpt2/gpt2_model.py`` (``GroupRecursiveGPT2MTPBlock``).
* Metric: per-token cosine (``dim=-1``), averaged over sequence positions and all windows.
"""

import argparse
import gc
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")  # safe for headless / non-interactive runs
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Make the modalities package importable.
# --------------------------------------------------------------------------- #
WORKSPACE_ROOT = Path("/leonardo_work/EUHPC_D21_101/bshomali/")
MODALITIES_SRC = WORKSPACE_ROOT / "modalities" / "src"
if str(MODALITIES_SRC) not in sys.path:
    sys.path.insert(0, str(MODALITIES_SRC))


# =========================================================================== #
# Configuration
# =========================================================================== #
# ----- Checkpoints to compare -----
# Processed one at a time; only one model is in memory at any moment.
CHECKPOINT_PREFIX = "/leonardo_scratch/large/userexternal/bshomali/checkpoints/"
MODELS = [
    {
        "label": "MTP",
        "checkpoint_path": CHECKPOINT_PREFIX + "2026-05-28__01-48-52_a04797059e817388",
        "config_path": None,  # None for DCP; path to config yaml otherwise
    },
    {
        "label": "no-MTP",
        "checkpoint_path": CHECKPOINT_PREFIX + "2026-05-08__02-55-20_7b57d910d04cc5a8",
        "config_path": None,
    },
]
MODEL_KEY = "model_raw"

# ----- Device / batching -----
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # windows per forward pass
MAX_WINDOWS = None  # set to an int (e.g. 256) for a quick run; None = all
SEQ_LEN = None  # None -> use each model's sequence_length

# ----- Tokens -----
TOKEN_CACHE_FILE = Path(
    "/leonardo_work/EUHPC_D21_101/bshomali/modalities/Loop_MTP_paper_exp/.ppl_cache/fwedu_sample-10BT_n5000000_seed42.npy"
)

# ----- Result cache / output -----
CACHE_DIR = Path("./.similarity_cache")
PLOTS_DIR = Path("./plots")
PLOT_PATH = PLOTS_DIR / "per_iteration_input_vs_output_similarity_mtp_vs_nomtp.pdf"
PLOT_PATH_TO_FIRST = PLOTS_DIR / "per_iteration_output_vs_first_output_similarity_mtp_vs_nomtp.pdf"


# =========================================================================== #
# Cache helpers
# =========================================================================== #
def _cache_key(spec) -> str:
    """A cache filename that is stable for a given model + measurement settings.

    Including the relevant knobs means changing batching/window settings produces a
    distinct cache entry instead of silently reusing stale numbers.
    """
    payload = "|".join(
        str(x)
        for x in (
            "per_iter_input_vs_output_and_out0_vs_future",  # metric tag: distinguishes from other similarity metrics
            spec["label"],
            spec["checkpoint_path"],
            spec["config_path"],
            MAX_WINDOWS,
            SEQ_LEN,
            TOKEN_CACHE_FILE.name,
        )
    )
    digest = hashlib.md5(payload.encode()).hexdigest()[:10]
    safe_label = spec["label"].replace("/", "_")
    return f"{safe_label}_{digest}.parquet"


def _cache_path(spec) -> Path:
    return CACHE_DIR / _cache_key(spec)


# =========================================================================== #
# Per-model analysis
# =========================================================================== #
def analyze_model(spec, token_buffer):
    """Load one model, measure per-iteration (block-input vs. output) cosine, free it.

    Returns a per-iteration similarity ``DataFrame``. Only one model is resident in
    memory at a time: the model is deleted and the CUDA cache emptied before return.
    """
    # Imported here so ``--plot-only`` never needs the heavy deps.
    from modalities.evaluation.olmes_evaluator import load_modalities_model
    from modalities.models.gpt2.gpt2_model import BlockTypes

    recurrent_types = (BlockTypes.GROUP_RECURSIVE_MTP, BlockTypes.GROUP_RECURSIVE)

    label = spec["label"]
    print(f'=== [{label}] loading {spec["checkpoint_path"]} ===')

    temp_conversion_dir = tempfile.mkdtemp(prefix="modalities_converted_")
    model, tokenizer, loaded_config = load_modalities_model(
        checkpoint_path=spec["checkpoint_path"],
        config_path=spec["config_path"],
        model_key=MODEL_KEY,
        converted_output_dir=temp_conversion_dir,
    )
    model = model.to(DEVICE).eval()

    # Expose per-iteration hidden states; skip per-iteration LM-head logits (not needed).
    model.return_each_recurrence_output = True
    model.need_mtp_logits = False
    model.track_recurrence_embd_similarity = False

    # Locate the recurrent block and turn on its per-iteration output flag.
    recurrent_layer_ids = [
        li for li in model.transformer.h if model.blocks_types[int(li)] in recurrent_types
    ]
    assert len(recurrent_layer_ids) >= 1, f"[{label}] No recurrent (GROUP_RECURSIVE*) block found."
    if len(recurrent_layer_ids) > 1:
        print(
            f"WARNING: [{label}] {len(recurrent_layer_ids)} recurrent blocks {recurrent_layer_ids}; "
            'hooking the FIRST one as the "input embedding" reference.'
        )
    rec_layer_id = recurrent_layer_ids[0]
    recurrent_block = model.transformer.h[rec_layer_id]
    recurrent_block.return_each_recurrence_output = True

    K = int(recurrent_block.max_recurrence)
    seq_len = SEQ_LEN or int(model.sequence_length)
    print(f"[{label}] recurrent block @ layer {rec_layer_id}, max_recurrence (K) = {K}, seq_len = {seq_len}")

    # Capture the tensor fed INTO the recurrent block (the "first input embedding").
    captured = {}

    def _capture_block_input(module, args, kwargs):
        captured["block_input"] = args[0].detach()

    hook_handle = recurrent_block.register_forward_pre_hook(_capture_block_input, with_kwargs=True)

    # Window the token stream for this model's sequence length.
    n_windows = len(token_buffer) // seq_len
    windows = token_buffer[: n_windows * seq_len].reshape(n_windows, seq_len)
    windows = torch.from_numpy(windows.astype(np.int64))
    if MAX_WINDOWS is not None:
        windows = windows[:MAX_WINDOWS]
    print(f"[{label}] {windows.shape[0]:,} windows of length {seq_len}")

    sample_key = model.sample_key  # key the model expects, e.g. 'input_ids'
    cos_sum = torch.zeros(K, dtype=torch.float64)  # sum of per-token cosine over all positions
    cos_sq_sum = torch.zeros(K, dtype=torch.float64)  # for std
    # Output of iteration 0 vs. output of iteration r (how later steps drift from the first).
    cos_first_sum = torch.zeros(K, dtype=torch.float64)
    cos_first_sq_sum = torch.zeros(K, dtype=torch.float64)
    token_count = 0

    with torch.no_grad():
        for start in tqdm(range(0, windows.shape[0], BATCH_SIZE), desc=f"{label} forward"):
            batch = windows[start : start + BATCH_SIZE].to(DEVICE)
            out = model({sample_key: batch})
            out = out[model.prediction_key] if model.prediction_key in out else out

            block_input = captured["block_input"].float()  # (B, T, D) -> input to iteration 0
            per_iter = out["each_recurrence_hidden_states"]  # list of K x (B, T, D)
            assert len(per_iter) == K, (len(per_iter), K)

            # Per-iteration "input vs output": the recurrence chains raw hidden states
            # (x = x_after fed into the next step), so the input to iteration r is the
            # block input for r == 0 and the previous iteration's output otherwise.
            first_output = per_iter[0].float()  # output of iteration 0 (reference)
            for r, h_r in enumerate(per_iter):
                h_r = h_r.float()
                iter_input = block_input if r == 0 else per_iter[r - 1].float()
                cos = F.cosine_similarity(iter_input, h_r, dim=-1)  # (B, T)
                cos = cos.reshape(-1).double().cpu()
                cos_sum[r] += cos.sum()
                cos_sq_sum[r] += (cos**2).sum()

                # First iteration's output vs. iteration r's output (r == 0 -> 1.0).
                cos_f = F.cosine_similarity(first_output, h_r, dim=-1)  # (B, T)
                cos_f = cos_f.reshape(-1).double().cpu()
                cos_first_sum[r] += cos_f.sum()
                cos_first_sq_sum[r] += (cos_f**2).sum()
            token_count += block_input.shape[0] * block_input.shape[1]

    mean_cos = (cos_sum / token_count).numpy()
    std_cos = np.sqrt((cos_sq_sum / token_count).numpy() - mean_cos**2)
    mean_cos_first = (cos_first_sum / token_count).numpy()
    std_cos_first = np.sqrt((cos_first_sq_sum / token_count).numpy() - mean_cos_first**2)
    df = pd.DataFrame(
        {
            "label": label,
            "iteration": np.arange(1, K + 1),
            "mean_cosine": mean_cos,
            "std_cosine": std_cos,
            "mean_cosine_to_first": mean_cos_first,
            "std_cosine_to_first": std_cos_first,
        }
    )
    print(f"[{label}] aggregated over {token_count:,} token positions")

    # Free everything before the next model is loaded.
    hook_handle.remove()
    del model, tokenizer, recurrent_block, captured, windows, out, per_iter, block_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


def get_results(recompute=False, plot_only=False):
    """Return the concatenated per-model results, loading from cache where possible."""
    CACHE_DIR.mkdir(exist_ok=True)
    all_results = []
    token_buffer = None  # lazily loaded only when something actually needs computing

    for spec in MODELS:
        cache_path = _cache_path(spec)
        if cache_path.exists() and not recompute:
            print(f"[{spec['label']}] loading cached results from {cache_path}")
            all_results.append(pd.read_parquet(cache_path))
            continue

        if plot_only:
            raise FileNotFoundError(
                f"--plot-only set but no cache for {spec['label']!r} at {cache_path}. "
                "Run once without --plot-only to populate the cache."
            )

        # Validate checkpoint and (lazily) load tokens before the first real measurement.
        assert spec["checkpoint_path"] and os.path.exists(spec["checkpoint_path"]), (
            f"Set a valid checkpoint_path for {spec['label']!r}. Got: {spec['checkpoint_path']!r}"
        )
        if token_buffer is None:
            token_buffer = np.load(TOKEN_CACHE_FILE)
            print(f"Loaded {len(token_buffer):,} tokens from {TOKEN_CACHE_FILE.name}")

        df = analyze_model(spec, token_buffer)
        df.to_parquet(cache_path)
        print(f"[{spec['label']}] cached results -> {cache_path}")
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# =========================================================================== #
# Plotting  (edit freely — re-run with cached results for instant turnaround)
# =========================================================================== #
def _plot_curve(results, mean_col, std_col, ylabel, title, out_path):
    """Overlay each model's per-iteration curve for one metric and save to ``out_path``."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for spec in MODELS:
        grp = results[results["label"] == spec["label"]]
        ax.errorbar(
            grp["iteration"],
            grp[mean_col],
            yerr=grp[std_col],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=spec["label"],
        )
    ax.set_xlabel("Recurrence iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(results["iteration"].unique()))
    ax.legend(title="model")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved plot -> {out_path}")
    return fig


def plot_results(results):
    _plot_curve(
        results,
        mean_col="mean_cosine",
        std_col="std_cosine",
        ylabel="Cosine similarity\n(iteration input vs. output)",
        title="Per-iteration input vs. output similarity\n(MTP vs. non-MTP)",
        out_path=PLOT_PATH,
    )
    _plot_curve(
        results,
        mean_col="mean_cosine_to_first",
        std_col="std_cosine_to_first",
        ylabel="Cosine similarity\n(first iteration output vs. iteration output)",
        title="First-iteration output vs. later-iteration output similarity\n(MTP vs. non-MTP)",
        out_path=PLOT_PATH_TO_FIRST,
    )


# =========================================================================== #
# Entry point
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true", help="Ignore cache and recompute all models."
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only plot from cache; never load a model."
    )
    args = parser.parse_args()

    print("DEVICE =", DEVICE)
    print("Models to compare:", [m["label"] for m in MODELS])

    results = get_results(recompute=args.recompute, plot_only=args.plot_only)
    print(results)
    plot_results(results)


if __name__ == "__main__":
    main()
