"""
Thought Anchors — Annotation Analysis & Visualization
======================================================
Reads the annotated JSON files produced by annotate_reasoning_traces.py
and generates plots + summary statistics inspired by the paper.

Outputs:
  1. Tag distribution (bar chart per model + combined)
  2. Tag distribution by position in trace (scatter, like Fig 3)
  3. Dependency depth & density stats
  4. Accuracy vs. tag composition
  5. Dependency graph for individual traces (networkx DAG)
  6. Cross-model comparison heatmap
  7. CSV exports of all extracted statistics

Usage:
    python analyze_annotations.py \
        --input-dir ./annotations \
        --output-dir ./analysis_results

    # Analyze a single file
    python analyze_annotations.py \
        --input-file ./annotations/model_a_annotated.json \
        --output-dir ./analysis_results

    # Plot dependency graph for a specific trace
    python analyze_annotations.py \
        --input-file ./annotations/model_a_annotated.json \
        --output-dir ./analysis_results \
        --plot-trace-id 0
"""

import argparse
import json
import csv
import math
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — install hints if missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAG_ORDER = [
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",
    "self_checking",
    "final_answer_emission",
    "unknown",
]

TAG_COLORS = {
    "problem_setup":          "#4E79A7",
    "plan_generation":        "#F28E2B",
    "fact_retrieval":         "#E15759",
    "active_computation":     "#76B7B2",
    "result_consolidation":   "#59A14F",
    "uncertainty_management": "#EDC948",
    "self_checking":          "#B07AA1",
    "final_answer_emission":  "#FF9DA7",
    "unknown":                "#BAB0AC",
}

TAG_SHORT = {
    "problem_setup":          "PS",
    "plan_generation":        "PG",
    "fact_retrieval":         "FR",
    "active_computation":     "AC",
    "result_consolidation":   "RC",
    "uncertainty_management": "UM",
    "self_checking":          "SC",
    "final_answer_emission":  "FAE",
    "unknown":                "UNK",
}

TAG_SHORT_TO_FULL = {abbr.upper(): full for full, abbr in TAG_SHORT.items()}
PLOT_TAG_ORDER = TAG_ORDER.copy()


def _get_plot_tag_order(include_unknown: bool = True) -> list[str]:
    tags = PLOT_TAG_ORDER
    if not include_unknown:
        tags = [t for t in tags if t != "unknown"]
    return tags


def _set_plot_tag_order(selected_tags: Optional[list[str]] = None):
    """Set active tag order for tag-based plots.

    If selected_tags is None, defaults to full TAG_ORDER.
    """
    global PLOT_TAG_ORDER
    if selected_tags is None:
        PLOT_TAG_ORDER = TAG_ORDER.copy()
        return

    selected_set = set(selected_tags)
    PLOT_TAG_ORDER = [t for t in TAG_ORDER if t in selected_set]


def _resolve_requested_plot_tags(raw_items: list[str]) -> tuple[list[str], list[str]]:
    """Resolve tag filters from abbreviations/full names.

    Accepts values like: PS AC, or problem_setup, active_computation.
    Returns (resolved_tag_names, unknown_tokens).
    """
    raw_tokens = []
    for item in raw_items:
        raw_tokens.extend([tok.strip() for tok in item.split(",") if tok.strip()])

    resolved = []
    unknown = []
    for tok in raw_tokens:
        upper_tok = tok.upper()
        if upper_tok in TAG_SHORT_TO_FULL:
            resolved.append(TAG_SHORT_TO_FULL[upper_tok])
        elif tok in TAG_ORDER:
            resolved.append(tok)
        else:
            unknown.append(tok)

    # Deduplicate while preserving TAG_ORDER
    resolved_set = set(resolved)
    resolved_in_order = [t for t in TAG_ORDER if t in resolved_set]
    return resolved_in_order, unknown


def get_bar_palette(n_colors: int) -> list:
    if n_colors <= 0:
        return []
    return [plt.cm.viridis(v) for v in np.linspace(0.1, 0.9, n_colors)]


# ============================================================================
# Data loading
# ============================================================================
def load_annotations(filepath: str) -> list[dict]:
    """Load annotations from JSON or JSONL."""
    path = Path(filepath)
    items = []

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # Standard JSON array
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            items = data
        else:
            # JSONL — one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

    return items


def load_all_models(input_dir: str) -> dict[str, list[dict]]:
    """Returns {model_name: [trace_dicts]}."""
    models = {}
    for p in sorted(Path(input_dir).glob("*_annotated.*")):
        if p.suffix.lower() not in (".json", ".jsonl", ".tmp"):
            continue
        name = p.stem.replace("_annotated", "")
        models[name] = load_annotations(str(p))
    return models


# ============================================================================
# Extraction helpers
# ============================================================================
def get_primary_tag(ann: dict) -> str:
    """Return the first function_tag for a sentence annotation."""
    tags = ann.get("function_tags", [])
    return tags[0] if tags else "unknown"


def extract_tag_counts(traces: list[dict]) -> Counter:
    """Count every tag occurrence across all traces."""
    counts = Counter()
    for tr in traces:
        for ann in tr.get("annotations", {}).values():
            if isinstance(ann, dict):
                for tag in ann.get("function_tags", []):
                    counts[tag] += 1
    return counts


def extract_tag_positions(traces: list[dict]) -> dict[str, list[float]]:
    """For each tag, collect the normalized position (0-1) where it appears."""
    positions = defaultdict(list)
    for tr in traces:
        n = tr.get("num_sentences", 0)
        if n == 0:
            continue
        for idx_str, ann in tr.get("annotations", {}).items():
            if not isinstance(ann, dict):
                continue
            try:
                pos = int(idx_str) / max(n - 1, 1)
            except ValueError:
                continue
            for tag in ann.get("function_tags", []):
                positions[tag].append(pos)
    return dict(positions)


def extract_dependency_stats(traces: list[dict]) -> list[dict]:
    """Per-trace dependency statistics."""
    stats = []
    for tr in traces:
        annotations = tr.get("annotations", {})
        n = tr.get("num_sentences", 0)
        if n == 0:
            continue

        total_deps = 0
        max_dep_distance = 0
        dep_distances = []
        in_degree = Counter()
        out_degree = Counter()

        for idx_str, ann in annotations.items():
            if not isinstance(ann, dict):
                continue
            deps = ann.get("depends_on", [])
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            out_degree[idx] = len(deps)
            for d in deps:
                try:
                    d_int = int(d)
                except (ValueError, TypeError):
                    continue
                in_degree[d_int] += 1
                total_deps += 1
                dist = idx - d_int
                dep_distances.append(dist)
                max_dep_distance = max(max_dep_distance, dist)

        avg_dep_distance = np.mean(dep_distances) if dep_distances else 0.0

        # Find "hub" sentences (high in-degree = many downstream sentences depend on them)
        hub_sentences = [
            (k, v) for k, v in in_degree.most_common(3)
        ]

        stats.append({
            "doc_id": tr.get("doc_id", ""),
            "num_sentences": n,
            "total_dependencies": total_deps,
            "density": total_deps / max(n * (n - 1) / 2, 1),
            "max_dep_distance": max_dep_distance,
            "avg_dep_distance": round(avg_dep_distance, 2),
            "hub_sentences": hub_sentences,
            "exact_match": tr.get("exact_match"),
            "label": tr.get("label", ""),
            "model_answer": tr.get("model_answer", ""),
        })
    return stats


def extract_per_sentence_table(traces: list[dict]) -> list[dict]:
    """Flat table: one row per sentence across all traces."""
    rows = []
    for tr in traces:
        doc_id = tr.get("doc_id", "")
        n = tr.get("num_sentences", 0)
        sentences = tr.get("sentences", {})
        annotations = tr.get("annotations", {})
        exact_match = tr.get("exact_match")

        for idx_str in sorted(annotations.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            ann = annotations[idx_str]
            if not isinstance(ann, dict):
                continue
            rows.append({
                "doc_id": doc_id,
                "sentence_idx": int(idx_str),
                "normalized_position": round(int(idx_str) / max(n - 1, 1), 4),
                "text": sentences.get(idx_str, ""),
                "primary_tag": get_primary_tag(ann),
                "all_tags": ", ".join(ann.get("function_tags", [])),
                "num_dependencies": len(ann.get("depends_on", [])),
                "depends_on": ", ".join(str(d) for d in ann.get("depends_on", [])),
                "exact_match": exact_match,
            })
    return rows


# ============================================================================
# Token counting (GPT-2 tokenizer)
# ============================================================================
_GPT2_TOKENIZER = None


def _get_gpt2_tokenizer():
    """Lazily load and cache the GPT-2 tokenizer."""
    global _GPT2_TOKENIZER
    if _GPT2_TOKENIZER is None:
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            sys.exit("transformers is required for token counting: pip install transformers")
        _GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    return _GPT2_TOKENIZER


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_gpt2_tokenizer().encode(text))


def _correctness_bucket(tr: dict) -> Optional[str]:
    """Return 'correct'/'incorrect'/None based on exact_match."""
    em = tr.get("exact_match")
    if em is None:
        return None
    return "correct" if em == 1.0 else "incorrect"


def extract_response_token_counts(traces: list[dict]) -> dict[str, list[int]]:
    """GPT-2 token count of each trace's full response.

    Returns {"all": [...], "correct": [...], "incorrect": [...]}.
    """
    out = {"all": [], "correct": [], "incorrect": []}
    for tr in traces:
        n_tokens = _count_tokens(tr.get("response", ""))
        if n_tokens == 0:
            continue
        out["all"].append(n_tokens)
        bucket = _correctness_bucket(tr)
        if bucket is not None:
            out[bucket].append(n_tokens)
    return out


def extract_tag_token_counts(traces: list[dict]) -> dict[str, dict[str, list[int]]]:
    """Per-tag GPT-2 token counts of the sentences carrying each tag.

    A sentence's token count is attributed to every function_tag it has.
    Returns {tag: {"all": [...], "correct": [...], "incorrect": [...]}}.
    """
    out = defaultdict(lambda: {"all": [], "correct": [], "incorrect": []})
    for tr in traces:
        sentences = tr.get("sentences", {})
        annotations = tr.get("annotations", {})
        bucket = _correctness_bucket(tr)
        for idx_str, ann in annotations.items():
            if not isinstance(ann, dict):
                continue
            tags = ann.get("function_tags", [])
            if not tags:
                continue
            n_tokens = _count_tokens(sentences.get(idx_str, ""))
            for tag in tags:
                out[tag]["all"].append(n_tokens)
                if bucket is not None:
                    out[tag][bucket].append(n_tokens)
    return dict(out)


# ============================================================================
# Plotting
# ============================================================================
def _draw_tag_distribution(
    ax,
    tag_counts: Counter,
    title: str,
    show_legend: bool = True,
    normalize: bool = False,
):
    tags = [t for t in _get_plot_tag_order() if t in tag_counts]
    counts = [tag_counts[t] for t in tags]
    colors = get_bar_palette(len(tags))
    labels = [TAG_SHORT.get(t, t) for t in tags]

    if not tags:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No tag data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    total = sum(counts)
    values = [count / total * 100 for count in counts] if normalize and total else counts

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    y_offset = max(values) * 0.01 if values else 0
    for bar, value in zip(bars, values):
        label = f"{value:.1f}%" if normalize else str(int(value))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Percentage" if normalize else "Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        patches = [mpatches.Patch(color=color, label=tag) for tag, color in zip(tags, colors)]
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9)


def plot_tag_distribution(tag_counts: Counter, title: str, outpath: str):
    """Bar chart of tag frequencies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_tag_distribution(ax, tag_counts, title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_tag_by_position(ax, tag_positions: dict, title: str):
    tags = [t for t in _get_plot_tag_order() if t in tag_positions and len(tag_positions[t]) >= 3]
    if not tags:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Not enough position data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return False

    data = [tag_positions[t] for t in tags]
    labels = [TAG_SHORT.get(t, t) for t in tags]
    colors = [TAG_COLORS.get(t, "#999") for t in tags]

    parts = ax.violinplot(data, positions=range(len(tags)), showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#333")

    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized position in trace (0 = start, 1 = end)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def plot_tag_by_position(tag_positions: dict, title: str, outpath: str):
    """Scatter / violin showing where each tag appears in the trace (Fig 3 style)."""
    if not any(t in tag_positions and len(tag_positions[t]) >= 3 for t in _get_plot_tag_order()):
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_tag_by_position(ax, tag_positions, title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_accuracy_vs_tags(ax, traces: list[dict], title: str, show_legend: bool = True):
    correct_counts = Counter()
    incorrect_counts = Counter()
    n_correct = 0
    n_incorrect = 0

    for tr in traces:
        em = tr.get("exact_match")
        if em is None:
            continue
        is_correct = em == 1.0
        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1
        for ann in tr.get("annotations", {}).values():
            if not isinstance(ann, dict):
                continue
            for tag in ann.get("function_tags", []):
                if is_correct:
                    correct_counts[tag] += 1
                else:
                    incorrect_counts[tag] += 1

    if n_correct == 0 or n_incorrect == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Need both correct and incorrect traces", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return False

    tags = [t for t in _get_plot_tag_order() if t in correct_counts or t in incorrect_counts]
    labels = [TAG_SHORT.get(t, t) for t in tags]
    correct_norm = [correct_counts.get(t, 0) / n_correct for t in tags]
    incorrect_norm = [incorrect_counts.get(t, 0) / n_incorrect for t in tags]

    x = np.arange(len(tags))
    w = 0.35

    ax.bar(x - w / 2, correct_norm, w, label=f"Correct (n={n_correct})", color="#59A14F", edgecolor="white")
    ax.bar(x + w / 2, incorrect_norm, w, label=f"Incorrect (n={n_incorrect})", color="#E15759", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg tags per trace")
    ax.set_title(title)
    if show_legend:
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def _draw_accuracy_distribution_vs_tags(ax, traces: list[dict], title: str, show_legend: bool = True):
    """Same values as _draw_accuracy_vs_tags (avg tags per trace), but correct
    and incorrect are stacked on top of each other instead of side-by-side."""
    correct_counts = Counter()
    incorrect_counts = Counter()
    n_correct = 0
    n_incorrect = 0

    for tr in traces:
        em = tr.get("exact_match")
        if em is None:
            continue
        is_correct = em == 1.0
        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1
        for ann in tr.get("annotations", {}).values():
            if not isinstance(ann, dict):
                continue
            for tag in ann.get("function_tags", []):
                if is_correct:
                    correct_counts[tag] += 1
                else:
                    incorrect_counts[tag] += 1

    if n_correct == 0 or n_incorrect == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Need both correct and incorrect traces", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return False

    tags = [t for t in _get_plot_tag_order() if t in correct_counts or t in incorrect_counts]
    labels = [TAG_SHORT.get(t, t) for t in tags]
    correct_norm = [correct_counts.get(t, 0) / n_correct for t in tags]
    incorrect_norm = [incorrect_counts.get(t, 0) / n_incorrect for t in tags]

    x = np.arange(len(tags))
    w = 0.6

    ax.bar(x, correct_norm, w, label=f"Correct (n={n_correct})", color="#59A14F", edgecolor="white")
    ax.bar(x, incorrect_norm, w, bottom=correct_norm, label=f"Incorrect (n={n_incorrect})", color="#E15759", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg tags per trace")
    ax.set_title(title)
    if show_legend:
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def plot_accuracy_distribution_vs_tags(traces: list[dict], title: str, outpath: str):
    """Single model: correct/incorrect avg-tags-per-trace stacked per tag."""
    has_correct = any(tr.get("exact_match") == 1.0 for tr in traces)
    has_incorrect = any(
        tr.get("exact_match") is not None and tr.get("exact_match") != 1.0
        for tr in traces
    )
    if not has_correct or not has_incorrect:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_accuracy_distribution_vs_tags(ax, traces, title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_tags(traces: list[dict], title: str, outpath: str):
    """Compare tag distributions for correct vs incorrect answers."""
    has_correct = any(tr.get("exact_match") == 1.0 for tr in traces)
    has_incorrect = any(
        tr.get("exact_match") is not None and tr.get("exact_match") != 1.0
        for tr in traces
    )
    if not has_correct or not has_incorrect:
        return  # Can't compare

    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_accuracy_vs_tags(ax, traces, title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_dep_distance_distribution(ax, traces: list[dict], title: str):
    distances = []
    for tr in traces:
        for idx_str, ann in tr.get("annotations", {}).items():
            if not isinstance(ann, dict):
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            for d in ann.get("depends_on", []):
                try:
                    distances.append(idx - int(d))
                except (ValueError, TypeError):
                    pass

    if not distances:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No dependency data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return False

    max_dist = max(distances)
    bins = range(1, min(max_dist + 2, 30))
    ax.hist(distances, bins=bins, color="#4E79A7", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Dependency distance (sentences apart)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def plot_dep_distance_distribution(traces: list[dict], title: str, outpath: str):
    """Histogram of dependency distances across all traces."""
    has_distances = any(
        ann.get("depends_on")
        for tr in traces
        for ann in tr.get("annotations", {}).values()
        if isinstance(ann, dict)
    )
    if not has_distances:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_dep_distance_distribution(ax, traces, title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model_comparison(all_models: dict[str, list[dict]], outpath: str):
    """Heatmap: rows = models, cols = tags, values = proportion."""
    if len(all_models) < 2:
        return

    model_names = sorted(all_models.keys())
    tags = _get_plot_tag_order(include_unknown=False)
    tag_labels = [TAG_SHORT[t] for t in tags]

    matrix = []
    for mname in model_names:
        counts = extract_tag_counts(all_models[mname])
        total = sum(counts.values()) or 1
        row = [counts.get(t, 0) / total for t in tags]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, max(3, len(model_names) * 0.8 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tag_labels, fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(len(model_names)):
        for j in range(len(tags)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.3 else "black")

    ax.set_title("Tag proportion by model")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Proportion")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_grid_dims(
    n_models: int,
    requested_rows: Optional[int] = None,
    requested_cols: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve subplot grid dimensions with safe fallbacks.

    Rules:
      - If neither rows nor cols is provided: use default layout.
      - If only one is provided: infer the other.
      - If both are provided but capacity is too small: fall back to default.
      - Non-positive values are treated as missing.
    """
    default_cols = min(2, n_models)
    default_rows = math.ceil(n_models / default_cols)

    rows = requested_rows if requested_rows and requested_rows > 0 else None
    cols = requested_cols if requested_cols and requested_cols > 0 else None

    if rows is None and cols is None:
        return default_rows, default_cols

    if rows is None:
        rows = math.ceil(n_models / cols)
    elif cols is None:
        cols = math.ceil(n_models / rows)

    if rows * cols < n_models:
        print(
            f"[warn] Requested grid {rows}x{cols} is too small for {n_models} models; "
            f"falling back to auto grid {default_rows}x{default_cols}."
        )
        return default_rows, default_cols

    return rows, cols


def _build_model_grid(
    all_models: dict[str, list[dict]],
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    **kwargs,
):
    model_names = sorted(all_models.keys(), key=lambda x: int(x[len("loop"):]) if x[len("loop"):].isdigit() else float('inf'))
    n_models = len(model_names)
    nrows, ncols = _resolve_grid_dims(n_models, grid_rows, grid_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, **kwargs)
    flat_axes = axes.flatten()
    return fig, flat_axes, model_names, nrows, ncols


def _apply_single_row_ylabel_rule(axes, nrows: int, ncols: int, n_used_axes: int):
    """If grid has one row, keep ylabel only on the first plotted axis."""
    if nrows != 1:
        return
    for idx, ax in enumerate(axes[:n_used_axes]):
        if idx % ncols != 0:
            ax.set_ylabel("")


def plot_multi_model_tag_distributions(
    all_models: dict[str, list[dict]],
    outpath: str,
    normalize: bool = False,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        sharey=True,
    )
    distribution_tags = _get_plot_tag_order(include_unknown=False)
    legend_handles = [
        mpatches.Patch(color=color, label=tag)
        for tag, color in zip(distribution_tags, get_bar_palette(len(distribution_tags)))
    ]

    for ax, model_name in zip(axes, model_names):
        _draw_tag_distribution(
            ax,
            extract_tag_counts(all_models[model_name]),
            model_name,
            show_legend=False,
            normalize=normalize,
        )
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    if len(axes) > 1:
        axes[1].set_ylabel("")
    if len(axes) > 3:
        axes[3].set_ylabel("")

    ylabel = "Percentage (%)" if normalize else "Count"
    axes[0].set_ylabel(ylabel, fontsize=15)
    if len(axes) > 2:
        axes[2].set_ylabel(ylabel, fontsize=15)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)

    fig.suptitle(
        "Tag distribution by model (%)" if normalize else "Tag distribution by model",
        y=0.99,
        fontsize=16,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center", 
        ncol=4, 
        fontsize=12, 
        bbox_to_anchor=(0.5, 0.95),
        frameon=True,
        framealpha=0.8,
        edgecolor="#cccccc"
    )

    # fig.legend(
    #     handles=legend_handles, 
    #     loc="upper center", 
    #     ncol=4, 
    #     fontsize=8, 
    #     bbox_to_anchor=(0.5, 0.96),
    #     frameon=True,
    #     framealpha=0.8,
    #     edgecolor="#cccccc"
    # )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_tag_positions(
    all_models: dict[str, list[dict]],
    outpath: str,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    any_plotted = False
    for ax, model_name in zip(axes, model_names):
        any_plotted = _draw_tag_by_position(ax, extract_tag_positions(all_models[model_name]), model_name) or any_plotted
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle("Tag position in trace by model", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_accuracy_vs_tags(
    all_models: dict[str, list[dict]],
    outpath: str,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        sharey=True,
    )
    any_plotted = False
    legend_handles = None
    legend_labels = None

    for ax, model_name in zip(axes, model_names):
        plotted = _draw_accuracy_vs_tags(ax, all_models[model_name], model_name, show_legend=False)
        any_plotted = plotted or any_plotted
        if plotted and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    # turn off y labels for secondary panels when present
    if len(axes) > 1:
        axes[1].set_ylabel("")
    if len(axes) > 3:
        axes[3].set_ylabel("")

    # axes[0].set_xticklabels([])
    # axes[1].set_xticklabels([])

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle("Tag composition: correct vs incorrect by model", y=0.95, fontsize=16)
    if legend_handles:
        # fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.96))
        legend_labels_changed = []
        for lb in legend_labels:
            if "Correct" in lb:
                legend_labels_changed.append("Correct")
            if "Incorrect" in lb:
                legend_labels_changed.append("Incorrect")
        fig.legend(
            legend_handles, 
            legend_labels_changed,
            loc="upper center", 
            ncol=4, 
            fontsize=12, 
            bbox_to_anchor=(0.5, 0.92),
            frameon=True,
            framealpha=0.8,
            edgecolor="#cccccc"
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_accuracy_distribution_vs_tags(
    all_models: dict[str, list[dict]],
    outpath: str,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        sharey=True,
    )
    any_plotted = False
    legend_handles = None
    legend_labels = None

    for ax, model_name in zip(axes, model_names):
        plotted = _draw_accuracy_distribution_vs_tags(ax, all_models[model_name], model_name, show_legend=False)
        any_plotted = plotted or any_plotted
        if plotted and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    # turn off y labels for secondary panels when present
    if len(axes) > 1:
        axes[1].set_ylabel("")
    if len(axes) > 3:
        axes[3].set_ylabel("")

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle("Tag composition: correct vs incorrect (stacked) by model", y=0.95, fontsize=16)
    if legend_handles:
        legend_labels_changed = []
        for lb in legend_labels:
            if "Correct" in lb:
                legend_labels_changed.append("Correct")
            if "Incorrect" in lb:
                legend_labels_changed.append("Incorrect")
        fig.legend(
            legend_handles,
            legend_labels_changed,
            loc="upper center",
            ncol=4,
            fontsize=12,
            bbox_to_anchor=(0.5, 0.92),
            frameon=True,
            framealpha=0.8,
            edgecolor="#cccccc"
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_dep_distances(
    all_models: dict[str, list[dict]],
    outpath: str,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    any_plotted = False
    for ax, model_name in zip(axes, model_names):
        any_plotted = _draw_dep_distance_distribution(ax, all_models[model_name], model_name) or any_plotted
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle("Dependency distances by model", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trace_dag(trace: dict, outpath: str):
    """Plot the dependency DAG for a single trace."""
    if not HAS_NX:
        print("  [skip] networkx not installed — skipping DAG plot")
        return

    annotations = trace.get("annotations", {})
    sentences = trace.get("sentences", {})
    n = trace.get("num_sentences", 0)
    if n == 0:
        return

    G = nx.DiGraph()
    for idx_str, ann in annotations.items():
        if not isinstance(ann, dict):
            continue
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        tag = get_primary_tag(ann)
        label_text = sentences.get(idx_str, "")
        short = label_text[:40] + "…" if len(label_text) > 40 else label_text
        G.add_node(idx, tag=tag, label=f"{idx}: {short}")
        for d in ann.get("depends_on", []):
            try:
                G.add_edge(int(d), idx)
            except (ValueError, TypeError):
                pass

    if not G.nodes:
        return

    # Layout: position nodes top-to-bottom by sentence index
    pos = {}
    nodes_sorted = sorted(G.nodes)
    for i, node in enumerate(nodes_sorted):
        # Stagger x slightly based on tag to reduce overlap
        tag = G.nodes[node].get("tag", "")
        tag_idx = TAG_ORDER.index(tag) if tag in TAG_ORDER else 0
        x_offset = (tag_idx - len(TAG_ORDER) / 2) * 0.3
        pos[node] = (x_offset, -i)

    node_colors = [TAG_COLORS.get(G.nodes[n].get("tag", ""), "#999") for n in G.nodes]
    node_labels = {n: str(n) for n in G.nodes}

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#aaa", arrows=True,
                           arrowsize=12, connectionstyle="arc3,rad=0.1", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=350, edgecolors="#333", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)

    # Add sentence text as annotations on the right
    for node in nodes_sorted:
        x, y = pos[node]
        label = G.nodes[node].get("label", "")
        tag = G.nodes[node].get("tag", "")
        ax.annotate(f"[{TAG_SHORT.get(tag, '?')}] {label}",
                    xy=(x, y), xytext=(3.5, y),
                    fontsize=6, va="center",
                    arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5))

    # Legend
    patches = [mpatches.Patch(color=TAG_COLORS[t], label=f"{TAG_SHORT[t]}: {t}")
               for t in TAG_ORDER if t != "unknown"]
    ax.legend(handles=patches, loc="lower left", fontsize=6, framealpha=0.9)

    doc_id = trace.get("doc_id", "?")
    em = trace.get("exact_match", "?")
    ax.set_title(f"Dependency graph — doc_id={doc_id}  (exact_match={em})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Token-count plots (GPT-2 tokenizer)
# ============================================================================
def _mean_std(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def plot_avg_tokens_per_response(all_models: dict[str, list[dict]], outpath: str):
    """Bar chart: average GPT-2 tokens per response, one bar per loop/model."""
    model_names = sorted(all_models.keys())
    means, stds = [], []
    for mname in model_names:
        counts = extract_response_token_counts(all_models[mname])
        mean, std = _mean_std(counts["all"])
        means.append(mean)
        stds.append(std)

    if not any(means):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.2), 5))
    colors = get_bar_palette(len(model_names))
    bars = ax.bar(model_names, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mean:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg tokens per response (GPT-2)")
    ax.set_title("Average response length by loop")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_avg_tokens_per_response_by_correctness(all_models: dict[str, list[dict]], outpath: str):
    """Grouped bar chart: avg response tokens per loop, correct vs incorrect."""
    model_names = sorted(all_models.keys())
    correct_means, correct_stds = [], []
    incorrect_means, incorrect_stds = [], []
    for mname in model_names:
        counts = extract_response_token_counts(all_models[mname])
        cm, cs = _mean_std(counts["correct"])
        im, isd = _mean_std(counts["incorrect"])
        correct_means.append(cm)
        correct_stds.append(cs)
        incorrect_means.append(im)
        incorrect_stds.append(isd)

    if not any(correct_means) and not any(incorrect_means):
        return

    x = np.arange(len(model_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 1.4), 5))
    ax.bar(x - w / 2, correct_means, w, yerr=correct_stds, capsize=3,
           label="Correct", color="#59A14F", edgecolor="white")
    ax.bar(x + w / 2, incorrect_means, w, yerr=incorrect_stds, capsize=3,
           label="Incorrect", color="#E15759", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Avg tokens per response (GPT-2)")
    ax.set_title("Average response length by loop: correct vs incorrect")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_avg_tokens_per_tag(all_models: dict[str, list[dict]], outpath: str):
    """Grouped bar chart: avg GPT-2 tokens per sentence of each tag, series = loops."""
    model_names = sorted(all_models.keys(), key=lambda x: int(x[len("loop"):]) if x[len("loop"):].isdigit() else float('inf'))
    per_model = {m: extract_tag_token_counts(all_models[m]) for m in model_names}

    tags = [t for t in _get_plot_tag_order() if any(t in per_model[m] for m in model_names)]
    if not tags:
        return

    labels = [TAG_SHORT.get(t, t) for t in tags]
    # extnded_labels = []
    # for l in labels:
    #     if l == "PS":
    #         extnded_labels.append("Problem \nSetup")
    #     elif l == "AC":
    #         extnded_labels.append("Active \nComputation")
    #     elif l == "UM":
    #         extnded_labels.append("Uncertainty \nManagement")
    #     elif l == "FR":
    #         extnded_labels.append("Fact \nRetrieval")
    #     elif l == "PG":
    #         extnded_labels.append("Plan \nGeneration")
    # labels = extnded_labels if extnded_labels else labels

    x = np.arange(len(tags))
    n_models = len(model_names)
    w = 0.8 / max(n_models, 1)
    colors = get_bar_palette(n_models)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, mname in enumerate(model_names):
        tag_counts = per_model[mname]
        means = [_mean_std(tag_counts.get(t, {}).get("all", []))[0] for t in tags]
        offset = (i - (n_models - 1) / 2) * w
        ax.bar(x + offset, means, w, label=mname, color=colors[i], edgecolor="white", linewidth=0.4)

        # horizon reference line for this loop (h = loop number)
        loop_num = int(mname[len("loop"):]) if mname.startswith("loop") and mname[len("loop"):].isdigit() else None
        if loop_num is not None:
            ax.axhline(loop_num, color=colors[i], linestyle="--", linewidth=0.9, alpha=0.8)
            ax.text(0.96, loop_num, f"h={loop_num}", color=colors[i],
                    fontsize=11, ha="left", va="bottom",
                    transform=ax.get_yaxis_transform())
            # ax.annotate(f"h={loop_num}", xy=(0.995, loop_num), xytext=(1.01, loop_num),
            #             xycoords=ax.get_yaxis_transform(),
            #             textcoords=ax.get_yaxis_transform(),
            #             color=colors[i], fontsize=9, va="center", ha="left",
            #             arrowprops=dict(arrowstyle="->", color=colors[i],
            #                             lw=0.8, shrinkA=0, shrinkB=2),
            #             annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg tokens per sentence")
    # ax.set_title("Average tokens per tag by loop")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_avg_tokens_per_tag_by_correctness(ax, traces: list[dict], title: str, show_legend: bool = True):
    tag_counts = extract_tag_token_counts(traces)
    tags = [t for t in _get_plot_tag_order() if t in tag_counts]
    correct_means = [_mean_std(tag_counts[t]["correct"])[0] for t in tags]
    incorrect_means = [_mean_std(tag_counts[t]["incorrect"])[0] for t in tags]

    # make the first letter uppercase for better title display
    title = title[0].upper() + title[1:] if title else title
    if not any(correct_means) and not any(incorrect_means):
        ax.set_title(title, fontsize=14)
        ax.text(0.5, 0.5, "Need both correct and incorrect traces",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return False

    labels = [TAG_SHORT.get(t, t) for t in tags]
    x = np.arange(len(tags))
    w = 0.35
    ax.bar(x - w / 2, correct_means, w, label="Correct", color="#59A14FE7", edgecolor="white")
    ax.bar(x + w / 2, incorrect_means, w, label="Incorrect", color="#E15759EA", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    # set yticks fotn size
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("Avg tokens per sentence", fontsize=14)
    ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(bbox_to_anchor=(0.55, 1.08))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def plot_avg_tokens_per_tag_by_correctness(traces: list[dict], title: str, outpath: str):
    """Single model: avg tokens per tag, correct vs incorrect."""
    fig, ax = plt.subplots(figsize=(11, 4))
    if not _draw_avg_tokens_per_tag_by_correctness(ax, traces, title):
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_avg_tokens_per_tag_by_correctness(
    all_models: dict[str, list[dict]],
    outpath: str,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    """Per-loop grid: avg tokens per tag, correct vs incorrect."""
    if len(all_models) < 2:
        return

    fig, axes, model_names, nrows, ncols = _build_model_grid(
        all_models,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        sharey=True,
    )
    any_plotted = False
    legend_handles = None
    legend_labels = None
    for ax, model_name in zip(axes, model_names):
        loop_num = int(model_name[len("loop"):]) if model_name.startswith("loop") and model_name[len("loop"):].isdigit() else None
        print(model_name, loop_num)
        plotted = _draw_avg_tokens_per_tag_by_correctness(
            ax, all_models[model_name], model_name, show_legend=False)
        if loop_num is not None:
            # ax.text(1.1, loop_num, f"h={loop_num}", color="#000000EF",
            #         fontsize=11, ha="right", va="bottom",
            #         transform=ax.get_yaxis_transform(),
            #         # bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.5, alpha=0.8)
            #         )
            ax.axhline(loop_num, color="#000000C3", linestyle="--", linewidth=0.8)
            # small connector arrow from the dashed line out to the offset text
            ax.annotate(f"h={loop_num}", xy=(0.98, loop_num), xytext=(0.975, loop_num+3),
                        xycoords=ax.get_yaxis_transform(),
                        textcoords=ax.get_yaxis_transform(),
                        arrowprops=dict(arrowstyle="->", color="#000000C3",
                                        lw=0.8, shrinkA=0, shrinkB=2),
                        annotation_clip=False)
        any_plotted = plotted or any_plotted
        if plotted and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(model_names):]:
        ax.axis("off")

    _apply_single_row_ylabel_rule(axes, nrows, ncols, len(model_names))

    if not any_plotted:
        plt.close(fig)
        return

    # fig.suptitle("Avg tokens per tag: correct vs incorrect by loop", y=0.95, fontsize=16)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center",
                   ncol=2, fontsize=13, bbox_to_anchor=(0.5, 1.07),
                   frameon=True, framealpha=0.8, edgecolor="#cccccc")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# CSV export
# ============================================================================
def write_csv(rows: list[dict], outpath: str):
    if not rows:
        return
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# Main
# ============================================================================
def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    if args.input_file:
        model_name = Path(args.input_file).stem.replace("_annotated", "")
        all_models = {model_name: load_annotations(args.input_file)}
    elif args.input_dir:
        all_models = load_all_models(args.input_dir)
    else:
        sys.exit("Provide --input-file or --input-dir")

    if not all_models:
        sys.exit("No annotated files found.")

    _set_plot_tag_order(None)
    if args.include_tags:
        selected_tags, unknown_tags = _resolve_requested_plot_tags(args.include_tags)
        if unknown_tags:
            print(f"[warn] Unknown tag filter(s) skipped: {unknown_tags}")
        if not selected_tags:
            sys.exit("No valid tags found for --include-tags.")

        _set_plot_tag_order(selected_tags)
        print(
            "Tag filter enabled: "
            f"{[TAG_SHORT.get(tag, tag) for tag in _get_plot_tag_order()]}"
        )

    if args.include_models:
        requested_models = []
        for item in args.include_models:
            requested_models.extend([m.strip() for m in item.split(",") if m.strip()])

        # Preserve user order while removing duplicates.
        requested_models = list(dict.fromkeys(requested_models))
        missing = [m for m in requested_models if m not in all_models]
        selected_models = {m: all_models[m] for m in requested_models if m in all_models}

        if missing:
            print(f"[warn] Requested model(s) not found and skipped: {missing}")
        if not selected_models:
            sys.exit("No matching models found for --include-models.")

        all_models = selected_models
        print(f"Model filter enabled: {list(all_models.keys())}")

    print(f"Loaded {len(all_models)} model(s): {list(all_models.keys())}")

    # ---- Per-model analysis ----
    for mname, traces in all_models.items():
        prefix = out / mname
        print(f"\n{'='*50}")
        print(f"Model: {mname}  ({len(traces)} traces)")
        print(f"{'='*50}")

        # 1) Tag distribution
        tag_counts = extract_tag_counts(traces)
        print(f"\n  Tag counts:")
        for t in TAG_ORDER:
            if t in tag_counts:
                print(f"    {TAG_SHORT[t]:>4} ({t}): {tag_counts[t]}")

        plot_tag_distribution(
            tag_counts,
            f"Tag distribution — {mname}",
            str(prefix) + "_tag_distribution.pdf",
        )

        # 2) Tag by position
        tag_pos = extract_tag_positions(traces)
        plot_tag_by_position(
            tag_pos,
            f"Tag position in trace — {mname}",
            str(prefix) + "_tag_positions.pdf",
        )

        # 3) Accuracy vs tags
        plot_accuracy_vs_tags(
            traces,
            f"Tag composition: correct vs incorrect — {mname}",
            str(prefix) + "_accuracy_vs_tags.pdf",
        )

        plot_accuracy_distribution_vs_tags(
            traces,
            f"Tag composition: correct vs incorrect (stacked) — {mname}",
            str(prefix) + "_accuracy_distribution_vs_tags.pdf",
        )

        # 4) Dependency distance distribution
        plot_dep_distance_distribution(
            traces,
            f"Dependency distances — {mname}",
            str(prefix) + "_dep_distances.pdf",
        )

        # 5) Dependency stats
        dep_stats = extract_dependency_stats(traces)
        write_csv(dep_stats, str(prefix) + "_dependency_stats.csv")

        avg_density = np.mean([s["density"] for s in dep_stats]) if dep_stats else 0
        avg_dist = np.mean([s["avg_dep_distance"] for s in dep_stats]) if dep_stats else 0
        print(f"\n  Dependency stats ({len(dep_stats)} traces):")
        print(f"    Avg graph density:        {avg_density:.4f}")
        print(f"    Avg dependency distance:  {avg_dist:.2f} sentences")

        # Accuracy summary
        em_vals = [s["exact_match"] for s in dep_stats if s["exact_match"] is not None]
        if em_vals:
            acc = sum(1 for v in em_vals if v == 1.0) / len(em_vals)
            print(f"    Accuracy:                 {acc:.2%} ({sum(1 for v in em_vals if v == 1.0)}/{len(em_vals)})")

        # 6) Per-sentence table
        sent_table = extract_per_sentence_table(traces)
        write_csv(sent_table, str(prefix) + "_sentences.csv")

        # 6b) Token-count analysis (GPT-2 tokenizer)
        resp_tokens = extract_response_token_counts(traces)
        tag_tokens = extract_tag_token_counts(traces)

        resp_mean, _ = _mean_std(resp_tokens["all"])
        c_mean, _ = _mean_std(resp_tokens["correct"])
        i_mean, _ = _mean_std(resp_tokens["incorrect"])
        print(f"\n  Response tokens (GPT-2):")
        print(f"    Avg per response:         {resp_mean:.1f}")
        if resp_tokens["correct"] or resp_tokens["incorrect"]:
            print(f"    Avg (correct):            {c_mean:.1f}")
            print(f"    Avg (incorrect):          {i_mean:.1f}")
        print(f"\n  Avg tokens per tag (GPT-2):")
        for t in TAG_ORDER:
            if t in tag_tokens:
                print(f"    {TAG_SHORT[t]:>4} ({t}): {_mean_std(tag_tokens[t]['all'])[0]:.1f}")

        plot_avg_tokens_per_tag_by_correctness(
            traces,
            f"Avg tokens per tag: correct vs incorrect — {mname}",
            # f"",
            str(prefix) + "_tokens_per_tag_by_correctness.pdf",
        )

        token_rows = [{
            "scope": "response",
            "avg_tokens_all": round(resp_mean, 2),
            "avg_tokens_correct": round(c_mean, 2),
            "avg_tokens_incorrect": round(i_mean, 2),
            "n_all": len(resp_tokens["all"]),
            "n_correct": len(resp_tokens["correct"]),
            "n_incorrect": len(resp_tokens["incorrect"]),
        }]
        for t in TAG_ORDER:
            if t not in tag_tokens:
                continue
            token_rows.append({
                "scope": t,
                "avg_tokens_all": round(_mean_std(tag_tokens[t]["all"])[0], 2),
                "avg_tokens_correct": round(_mean_std(tag_tokens[t]["correct"])[0], 2),
                "avg_tokens_incorrect": round(_mean_std(tag_tokens[t]["incorrect"])[0], 2),
                "n_all": len(tag_tokens[t]["all"]),
                "n_correct": len(tag_tokens[t]["correct"]),
                "n_incorrect": len(tag_tokens[t]["incorrect"]),
            })
        write_csv(token_rows, str(prefix) + "_token_counts.csv")

        # 7) Plot specific trace DAG if requested
        if args.plot_trace_id is not None:
            target = [t for t in traces if t.get("doc_id") == args.plot_trace_id]
            if target:
                plot_trace_dag(
                    target[0],
                    str(prefix) + f"_dag_doc{args.plot_trace_id}.pdf",
                )
                print(f"\n  DAG plotted for doc_id={args.plot_trace_id}")
            else:
                print(f"\n  [warn] doc_id={args.plot_trace_id} not found in {mname}")

    # ---- Cross-model comparison ----
    if len(all_models) >= 2:
        print(f"\n{'='*50}")
        print(f"Cross-model comparison")
        print(f"{'='*50}")
        plot_cross_model_comparison(all_models, str(out / "cross_model_heatmap.pdf"))
        plot_multi_model_tag_distributions(
            all_models,
            str(out / "cross_model_tag_distributions.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        plot_multi_model_tag_distributions(
            all_models,
            str(out / "cross_model_tag_distributions_pct.pdf"),
            normalize=True,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        plot_multi_model_tag_positions(
            all_models,
            str(out / "cross_model_tag_positions.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        plot_multi_model_accuracy_vs_tags(
            all_models,
            str(out / "cross_model_accuracy_vs_tags.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        plot_multi_model_accuracy_distribution_vs_tags(
            all_models,
            str(out / "cross_model_accuracy_distribution_vs_tags.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        plot_multi_model_dep_distances(
            all_models,
            str(out / "cross_model_dep_distances.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )

        # Token-count analysis across loops (GPT-2 tokenizer)
        plot_avg_tokens_per_response(all_models, str(out / "cross_model_tokens_per_response.pdf"))
        plot_avg_tokens_per_response_by_correctness(
            all_models, str(out / "cross_model_tokens_per_response_by_correctness.pdf"))
        plot_avg_tokens_per_tag(all_models, str(out / "cross_model_tokens_per_tag.pdf"))
        plot_multi_model_avg_tokens_per_tag_by_correctness(
            all_models,
            str(out / "cross_model_tokens_per_tag_by_correctness.pdf"),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )

        # Summary table
        rows = []
        for mname, traces in all_models.items():
            tc = extract_tag_counts(traces)
            total = sum(tc.values()) or 1
            em_vals = [
                tr.get("exact_match") for tr in traces
                if tr.get("exact_match") is not None
            ]
            row = {
                "model": mname,
                "num_traces": len(traces),
                "accuracy": round(sum(1 for v in em_vals if v == 1.0) / len(em_vals), 4) if em_vals else None,
                "avg_sentences": round(np.mean([tr.get("num_sentences", 0) for tr in traces]), 1),
            }
            for t in TAG_ORDER:
                row[f"pct_{TAG_SHORT[t]}"] = round(tc.get(t, 0) / total, 4)
            rows.append(row)
        write_csv(rows, str(out / "cross_model_summary.csv"))

        # Print comparison
        print("\n  Model             Traces  Acc     Avg_Sent  AC%     PG%     UM%     FR%")
        print("  " + "-" * 75)
        for r in rows:
            acc_str = f"{r['accuracy']:.2%}" if r["accuracy"] is not None else "N/A"
            print(f"  {r['model']:<18} {r['num_traces']:<7} {acc_str:<7} "
                  f"{r['avg_sentences']:<9} "
                  f"{r.get('pct_AC', 0):.2%}   {r.get('pct_PG', 0):.2%}   "
                  f"{r.get('pct_UM', 0):.2%}   {r.get('pct_FR', 0):.2%}")

    print(f"\nAll outputs saved to {out}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Thought Anchors annotations: plots, stats, DAGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--input-file", help="Single annotated JSON/JSONL file")
    grp.add_argument("--input-dir", help="Directory with *_annotated.json/.jsonl files")
    parser.add_argument("--output-dir", default="./analysis_results",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--plot-trace-id", type=int, default=None,
                        help="Plot dependency DAG for this doc_id")
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=None,
        help="Optional subplot grid rows for multi-model figures (auto if omitted)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=None,
        help="Optional subplot grid columns for multi-model figures (auto if omitted)",
    )
    parser.add_argument(
        "--include-models",
        nargs="+",
        default=None,
        help=(
            "Optional model names to include (space-separated and/or comma-separated). "
            "If omitted, all discovered models are analyzed."
        ),
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        default=None,
        help=(
            "Optional tag filters for plots (abbreviations like PS AC, or full tag names). "
            "If omitted, all tags are shown."
        ),
    )
    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        args.input_dir = "./annotations"

    run(args)


if __name__ == "__main__":
    main()