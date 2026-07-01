"""
Read training + evaluation metrics from W&B and produce comparison plots.

Plots generated (one file per summary/* metric)
------------------------------------------------
1. CE loss (y)  vs  eval score (x),  marker-shape = seen_steps  – all runs overlaid
2. Aux loss (y) vs  eval score (x),  marker-shape = seen_steps  – all runs overlaid
3. Final-step-only version of 1 and 2

Usage
-----
    python read_from_wandb.py            # generates local PNG files
    python read_from_wandb.py --upload   # also resumes each existing run and uploads
                                         # interactive tables + scatter charts there
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENTITY  = "behzadshomali"
PROJECT = "nemotron_MATH_PartialMTP_Gating"

RUN_IDS = [
    "usbeeci5",  # baseline
    "rt9nlwww",
    "e4dn6jgf",
    "d3j7px9k",
    "robmw8d8",
    "fk8bcynb",
    "vnimrd42",
]

TRAIN_CE_LOSS_KEY  = "train train ce loss avg"
TRAIN_AUX_LOSS_KEY = "train train aux loss avg"  # set to None to skip

STEP_KEY = "seen_steps"

# Which summary/* keys to plot (None → auto-detect all summary/* keys)
SUMMARY_KEYS = None   # e.g. ["summary/modalities::math_ac", "summary/avg_all_suites"]

OUTPUT_DIR = Path("./wandb_plots")
# ---------------------------------------------------------------------------

# Colors distinguish runs; marker shapes distinguish seen_steps checkpoints
COLORS_RUNS   = plt.cm.Set2.colors
STEP_MARKERS  = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


def fetch_run_data(api: wandb.Api, run_id: str) -> tuple[str, pd.DataFrame]:
    """Return (run_display_name, history DataFrame)."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = run.history(samples=10_000, pandas=True)
    return run.name or run_id, df


def pick_summary_keys(dfs: list[pd.DataFrame]) -> list[str]:
    """Collect all summary/* columns across all run DataFrames."""
    if SUMMARY_KEYS:
        return list(SUMMARY_KEYS)
    seen, keys = set(), []
    for df in dfs:
        for c in df.columns:
            if c.startswith("summary/") and c != "summary/table" and c not in seen:
                keys.append(c)
                seen.add(c)
    return sorted(keys)


def align_loss_to_eval(df: pd.DataFrame, loss_key: str, summary_key: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns [STEP_KEY, loss_key, summary_key].

    Training-loss rows are logged by the training loop using W&B's internal
    _step counter and do NOT carry seen_steps.  Eval rows are logged by
    write_to_wandb.py with seen_steps explicitly set.

    Strategy:
      1. eval_df  – rows where summary_key is non-NaN  → keeps _step + seen_steps
      2. loss_df  – rows where loss_key  is non-NaN    → keeps _step + loss value
      3. merge_asof on _step (nearest preceding loss for each eval checkpoint)
      4. Use seen_steps from eval rows as the shape axis.
    """
    if loss_key not in df.columns or summary_key not in df.columns:
        return pd.DataFrame()
    if "_step" not in df.columns:
        return pd.DataFrame()

    wstep = "_step"
    work = df.copy()
    work[wstep] = work[wstep].astype(float)
    work = work.sort_values(wstep)

    eval_df = (
        work[[wstep, STEP_KEY, summary_key]]
        .dropna(subset=[summary_key])
        .copy()
    )
    loss_df = (
        work[[wstep, loss_key]]
        .dropna(subset=[loss_key])
        .copy()
    )

    if eval_df.empty or loss_df.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(eval_df, loss_df, on=wstep, direction="nearest")

    # If seen_steps is entirely NaN, fall back to using _step as the shape axis
    if merged[STEP_KEY].isna().all():
        merged[STEP_KEY] = merged[wstep]

    return merged[[STEP_KEY, loss_key, summary_key]].dropna()


def _short(key: str) -> str:
    return key.split("/", 1)[-1]


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Plot: all checkpoints, marker shape = seen_steps, color = run
# ---------------------------------------------------------------------------

def plot_loss_vs_eval_scatter(
    run_frames: dict[str, pd.DataFrame],
    loss_key: str,
    summary_key: str,
    out_path: Path,
    loss_label: str,
):
    """
    Scatter: x = eval score, y = train loss.
    Color  = run identity.
    Marker = seen_steps checkpoint (each unique step value gets its own shape).
    Trajectory lines connect checkpoints within each run in step order.
    """
    run_data: list[tuple[str, pd.DataFrame]] = []
    all_steps_set: set[float] = set()

    for run_name, df in run_frames.items():
        aligned = align_loss_to_eval(df, loss_key, summary_key)
        if aligned.empty:
            continue
        run_data.append((run_name, aligned))
        all_steps_set.update(aligned[STEP_KEY].dropna().tolist())

    if not run_data:
        return

    sorted_steps = sorted(all_steps_set)
    step_to_marker = {s: STEP_MARKERS[i % len(STEP_MARKERS)]
                      for i, s in enumerate(sorted_steps)}

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (run_name, aligned) in enumerate(run_data):
        color = COLORS_RUNS[i % len(COLORS_RUNS)]
        steps = aligned[STEP_KEY].values
        xs    = aligned[summary_key].values
        ys    = aligned[loss_key].values

        # Trajectory line (connect in step order)
        order = np.argsort(steps)
        ax.plot(xs[order], ys[order], color=color, linewidth=0.7, alpha=0.4, zorder=2)

        # One scatter call per unique step to get the right marker in the legend
        for step in sorted_steps:
            mask = steps == step
            if not mask.any():
                continue
            ax.scatter(xs[mask], ys[mask],
                       color=color,
                       marker=step_to_marker[step],
                       s=80, edgecolors="white", linewidths=0.4,
                       zorder=3,
                       # only label the run on the first step to avoid duplicates
                       label=run_name if step == sorted_steps[0] else "_nolegend_")

    # --- Legend: runs (colored lines) + steps (marker shapes) ---
    from matplotlib.lines import Line2D
    run_handles = [
        Line2D([0], [0], color=COLORS_RUNS[i % len(COLORS_RUNS)],
               marker="o", markersize=6, linewidth=1.5, label=name)
        for i, (name, _) in enumerate(run_data)
    ]
    step_handles = [
        Line2D([0], [0], color="gray", marker=step_to_marker[s],
               markersize=7, linewidth=0, label=f"step {int(s):,}")
        for s in sorted_steps
    ]
    leg1 = ax.legend(handles=run_handles, title="Run",
                     loc="upper left", fontsize=7, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=step_handles, title="Seen steps",
              loc="lower right", fontsize=7, framealpha=0.8)

    ax.set_xlabel(_short(summary_key), fontsize=11)
    ax.set_ylabel(f"Train {loss_label}", fontsize=11)
    ax.set_title(f"Train {loss_label}  vs  {_short(summary_key)}", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot: final checkpoint only, one dot per run
# ---------------------------------------------------------------------------

def plot_final_scatter(
    run_frames: dict[str, pd.DataFrame],
    loss_key: str,
    summary_key: str,
    out_path: Path,
    loss_label: str,
):
    """One dot per run at its last checkpoint: x = eval score, y = train loss."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for i, (run_name, df) in enumerate(run_frames.items()):
        aligned = align_loss_to_eval(df, loss_key, summary_key)
        if aligned.empty:
            continue
        row = aligned.loc[aligned[STEP_KEY].idxmax()]
        ax.scatter(float(row[summary_key]), float(row[loss_key]),
                   color=COLORS_RUNS[i % len(COLORS_RUNS)],
                   marker="o", s=100,
                   edgecolors="white", linewidths=0.5,
                   label=run_name, zorder=3)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel(_short(summary_key), fontsize=11)
    ax.set_ylabel(f"Train {loss_label} (final)", fontsize=11)
    ax.set_title(f"Final: Train {loss_label}  vs  {_short(summary_key)}", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8,
              title="Run", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(upload: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    print("Fetching run histories from W&B …")
    # Keep insertion order so RUN_IDS[i] ↔ run_frames entry i
    run_id_to_name: dict[str, str] = {}   # run_id → display name
    run_frames: dict[str, pd.DataFrame] = {}  # display name → df

    for run_id in RUN_IDS:
        try:
            name, df = fetch_run_data(api, run_id)
            print(f"  {run_id} → '{name}'  ({len(df)} rows, {len(df.columns)} cols)")
            run_id_to_name[run_id] = name
            run_frames[name] = df
        except Exception as e:
            print(f"  ERROR fetching {run_id}: {e}")

    if not run_frames:
        print("No runs fetched. Check RUN_IDS, ENTITY, PROJECT.")
        return

    summary_keys = pick_summary_keys(list(run_frames.values()))
    print(f"\nSummary keys found: {summary_keys}\n")

    loss_configs = [
        (TRAIN_CE_LOSS_KEY,  "CE loss"),
        (TRAIN_AUX_LOSS_KEY, "aux loss"),
    ]
    # Map loss label → raw W&B column name (used in upload loop)
    loss_key_map = {label: key for key, label in loss_configs if key}

    plots_for_local: list[Path] = []
    # Collect combined data for upload: {(loss_label, summary_key): DataFrame}
    # DataFrame columns: [run, STEP_KEY, loss_key, summary_key]
    combined_data: dict[tuple[str, str], pd.DataFrame] = {}

    for summary_key in summary_keys:
        for loss_key, loss_label in loss_configs:
            if not loss_key:
                continue
            safe_loss = _safe(loss_label)
            safe_eval = _safe(_short(summary_key))

            # Collect aligned data across all runs
            parts = []
            for run_name, df in run_frames.items():
                aligned = align_loss_to_eval(df, loss_key, summary_key)
                if not aligned.empty:
                    a = aligned.copy()
                    a["run"] = run_name
                    parts.append(a)
            if parts:
                combined_data[(loss_label, summary_key)] = pd.concat(
                    parts, ignore_index=True
                )

            # Local PNG – all checkpoints
            p = OUTPUT_DIR / f"{safe_loss}_vs_{safe_eval}.png"
            plot_loss_vs_eval_scatter(run_frames, loss_key, summary_key, p, loss_label)
            if p.exists():
                plots_for_local.append(p)

            # Local PNG – final checkpoint only
            p = OUTPUT_DIR / f"final__{safe_loss}_vs_{safe_eval}.png"
            plot_final_scatter(run_frames, loss_key, summary_key, p, loss_label)
            if p.exists():
                plots_for_local.append(p)

    # -----------------------------------------------------------------------
    # Upload: resume each existing run and log interactive tables + scatters.
    # Sections in W&B are created by the "/" prefix in the key name:
    #   "CE loss/<metric>"  and  "aux loss/<metric>"
    # -----------------------------------------------------------------------
    if upload and combined_data:
        print("\nUploading interactive data to each W&B run …")
        for run_id in RUN_IDS:
            run_name = run_id_to_name.get(run_id)
            if run_name is None:
                continue

            wandb_log: dict = {}
            for (loss_label, summary_key), full_df in combined_data.items():
                raw_loss_key = loss_key_map[loss_label]
                col_eval = _short(summary_key)

                # Rename raw column names to human-readable labels
                plot_df = full_df[["run", STEP_KEY, raw_loss_key, summary_key]].rename(
                    columns={raw_loss_key: loss_label, summary_key: col_eval,
                             STEP_KEY: "seen_steps"}
                )

                # --- all checkpoints ---
                tbl_all = wandb.Table(dataframe=plot_df)
                wandb_log[f"{loss_label}/{col_eval}"] = wandb.plot.scatter(
                    tbl_all,
                    x=col_eval,
                    y=loss_label,
                    title=f"{loss_label} vs {col_eval} (all steps)",
                )
                wandb_log[f"{loss_label}/{col_eval}_table"] = tbl_all

                # --- final step only ---
                final_df = (
                    plot_df.loc[plot_df.groupby("run")["seen_steps"].idxmax()]
                    .reset_index(drop=True)
                )
                tbl_final = wandb.Table(dataframe=final_df)
                wandb_log[f"{loss_label}/{col_eval}_final"] = wandb.plot.scatter(
                    tbl_final,
                    x=col_eval,
                    y=loss_label,
                    title=f"{loss_label} vs {col_eval} (final step)",
                )

            with wandb.init(id=run_id, project=PROJECT, entity=ENTITY,
                            resume="must", reinit=True):
                wandb.log(wandb_log)
            print(f"  uploaded to run '{run_name}' ({run_id})")

    print(f"\nDone. {len(plots_for_local)} local plot(s) in {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true",
                        help="Resume each W&B run and upload interactive tables/scatters")
    args = parser.parse_args()
    main(upload=args.upload)
