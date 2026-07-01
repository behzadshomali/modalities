"""
Reasoning Trace Annotator
=========================
Based on the "Thought Anchors" paper (Bogdan et al., 2025).
Annotates chain-of-thought reasoning traces with:
  - Function tags (problem_setup, plan_generation, fact_retrieval, etc.)
  - Sentence-level dependency graphs

Uses the Fraunhofer FhGenie endpoint (OpenAI-compatible) by default.
API key and base URL are read from environment variables or a .env file:
    FRAUNHOFER_API_KEY=<your-key>
    FRAUNHOFER_BASE_URL=https://fhgenie.fraunhofer.de/v1

Processes outputs from multiple models stored as JSONL files.

Input format (JSONL, one per model):
  {"doc_id": 0, "native_id": 0, "metrics": {...}, "model_output": [{"continuation": "..."}], "label": "18", ...}

Usage:
    # Minimal — reads key from .env, uses FhGenie defaults
    python annotate_reasoning_traces.py \\
        --input-dir ./model_outputs \\
        --output-dir ./annotations

    # Override model or endpoint
    python annotate_reasoning_traces.py \\
        --model MiniMaxAI/MiniMax-M2.5 \\
        --input-dir ./model_outputs

    # With original questions for richer annotation context
    python annotate_reasoning_traces.py \\
        --input-dir ./model_outputs \\
        --questions-file ./gsm8k_questions.jsonl
"""

import argparse
import json
import os
import re
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env file if available (for FRAUNHOFER_API_KEY / FRAUNHOFER_BASE_URL)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on environment variables directly

# ---------------------------------------------------------------------------
# Fraunhofer FhGenie defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get(
    "FRAUNHOFER_BASE_URL", "https://fhgenie.fraunhofer.de/v1"
)
DEFAULT_API_KEY = os.environ.get("FRAUNHOFER_API_KEY", "")
DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5"

# ---------------------------------------------------------------------------
# Try to import the openai library; fall back to raw requests if unavailable
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    USE_OPENAI_LIB = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_OPENAI_LIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ============================================================================
# Annotation prompt – reproduced from Section E of the Thought Anchors paper
# ============================================================================
SYSTEM_PROMPT = """\
You are an expert in interpreting how LLMs solve math problems \
using multi-step reasoning. Your task is to analyze a \
chain-of-thought reasoning trace, broken into discrete text \
sentences, and label each sentence with:

1. **function_tags**: One or more labels that describe what this \
sentence is *doing* functionally in the reasoning process.
2. **depends_on**: A list of earlier sentence indices that this \
sentence directly depends on, e.g., uses information, results, or \
logic introduced in earlier sentences.

This annotation will be used to build a dependency graph and \
perform causal analysis, so please be precise and conservative: \
only mark a sentence as dependent on another if its reasoning \
clearly uses a previous sentence's result or idea.

Function Tags:
1. problem_setup: Parsing or rephrasing the problem (initial \
reading or comprehension).
2. plan_generation: Stating or deciding on a plan of action \
(often meta-reasoning).
3. fact_retrieval: Recalling facts, formulas, problem details \
(without immediate computation).
4. active_computation: Performing algebra, calculations, \
manipulations toward the answer.
5. result_consolidation: Aggregating intermediate results, \
summarizing, or preparing final answer.
6. uncertainty_management: Expressing confusion, re-evaluating, \
proposing alternative plans (includes backtracking).
7. final_answer_emission: Explicit statement of the final boxed \
answer or earlier sentences that contain the final answer.
8. self_checking: Verifying previous steps, checking \
calculations, and re-confirmations.
9. unknown: Use only if the sentence does not fit any of the \
above tags or is purely stylistic or semantic.

Dependencies:
For each sentence, include a list of earlier sentence indices that \
the reasoning in this sentence *uses*. For example:
- If sentence 9 performs a computation based on a plan in sentence \
4 and a recalled rule in sentence 5, then depends_on: [4, 5]
- If sentence 24 plugs in a final answer to verify correctness \
from sentence 23, then depends_on: [23]
- If there's no clear dependency use an empty list: []
- If sentence 13 performs a computation based on information in \
sentence 11, which in turn uses information from sentence 7, then \
depends_on: [11, 7]

Important Notes:
- Make sure to include all dependencies for each sentence.
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies.
- Try to be as comprehensive as possible.
- Make sure there is a path from earlier sentences to the final answer.

Output Format:
Return ONLY a valid JSON object (no markdown fences, no preamble) \
with one entry per sentence, where each entry has:
- the sentence index (as the key, converted to a string),
- a dictionary with:
  - "function_tags": list of tag strings
  - "depends_on": list of sentence indices, converted to strings

Example:
{
  "0": {"function_tags": ["problem_setup"], "depends_on": []},
  "1": {"function_tags": ["plan_generation"], "depends_on": ["0"]},
  "2": {"function_tags": ["fact_retrieval"], "depends_on": []},
  "3": {"function_tags": ["active_computation"], "depends_on": ["1", "2"]},
  "4": {"function_tags": ["uncertainty_management"], "depends_on": ["3"]},
  "5": {"function_tags": ["final_answer_emission"], "depends_on": ["3", "4"]}
}
"""


# ============================================================================
# Sentence splitting
# ============================================================================
def split_into_sentences(text: str) -> list[str]:
    """Split a reasoning trace into sentences."""
    # Remove <think>/<think> wrapper if present
    text = re.sub(r"</?think>", "", text).strip()
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace + capital/digit
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


# ============================================================================
# Build the user prompt for a single trace
# ============================================================================
def build_user_prompt(problem: str, sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))

    if problem:
        return (
            f"Here is the math problem:\n<PROBLEM>\n{problem}\n</PROBLEM>\n\n"
            f"Here is the full chain-of-thought, broken into sentences:\n"
            f"<SENTENCES>\n{numbered}\n</SENTENCES>\n\n"
            f"Now label each sentence with function tags and dependencies. "
            f"Return ONLY the JSON object."
        )
    else:
        # No problem text available — the CoT often restates the problem
        return (
            f"Here is the full chain-of-thought reasoning trace, broken into sentences:\n"
            f"<SENTENCES>\n{numbered}\n</SENTENCES>\n\n"
            f"Now label each sentence with function tags and dependencies. "
            f"Return ONLY the JSON object."
        )


# ============================================================================
# LLM calling helpers
# ============================================================================
def call_llm_openai(client, model, system, user, temperature=0.0, max_tokens=4096):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def call_llm_raw(api_base, api_key, model, system, user, temperature=0.0, max_tokens=4096):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


# ============================================================================
# Parse JSON from LLM response
# ============================================================================
def parse_annotation_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start : brace_end + 1]
    return json.loads(text)


# ============================================================================
# Load input files (your JSONL evaluation format)
# ============================================================================
def load_model_outputs(filepath: str) -> list[dict]:
    """Load model outputs from your evaluation JSONL format.

    Each line has:
      {"doc_id": int, "native_id": int, "metrics": {...},
       "model_output": [{"continuation": "...", "model_answer": "..."}],
       "label": "...", "task_hash": "...", "model_hash": "..."}

    Returns a list of normalized dicts with:
      doc_id, response (the CoT text), label, model_answer, metrics, exact_match
    """
    path = Path(filepath)
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  Skipping malformed line {line_num}: {e}")
                continue

            # --- Extract the reasoning trace ---
            response = ""
            model_answer = ""
            if isinstance(row.get("model_output"), list) and row["model_output"]:
                first_output = row["model_output"][0]
                if isinstance(first_output, dict):
                    response = first_output.get("continuation", "")
                    model_answer = str(first_output.get("model_answer", ""))
                elif isinstance(first_output, str):
                    response = first_output
            elif isinstance(row.get("model_output"), str):
                response = row["model_output"]

            # Fall back to top-level model_answer if not in model_output
            if not model_answer:
                model_answer = str(row.get("model_answer", ""))

            items.append({
                "doc_id": row.get("doc_id", line_num - 1),
                "native_id": row.get("native_id", row.get("doc_id", line_num - 1)),
                "response": response.strip(),
                "label": str(row.get("label", "")),
                "model_answer": model_answer,
                "exact_match": row.get("metrics", {}).get("exact_match", None),
                "num_tokens": row.get("metrics", {}).get("num_tokens", None),
                "task_hash": row.get("task_hash", ""),
                "model_hash": row.get("model_hash", ""),
            })

    return items


def load_questions(filepath: str) -> dict[int, str]:
    """Load question texts keyed by doc_id/index.

    Supports JSONL with 'question'/'problem'/'prompt' field,
    or JSON list. Returns {doc_id: question_text}.
    """
    path = Path(filepath)
    questions = {}

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id", idx)
                text = (
                    row.get("question") or row.get("problem")
                    or row.get("prompt") or row.get("input") or ""
                )
                questions[doc_id] = text
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for idx, row in enumerate(data):
                if isinstance(row, dict):
                    doc_id = row.get("doc_id", idx)
                    text = (
                        row.get("question") or row.get("problem")
                        or row.get("prompt") or row.get("input") or ""
                    )
                    questions[doc_id] = text
                elif isinstance(row, str):
                    questions[idx] = row
    else:
        # Plain text — one question per line
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    questions[idx] = line.strip()

    return questions


# ============================================================================
# Incremental file writing helpers
# ============================================================================
def flush_results_to_disk(results: list[dict], out_path: Path) -> None:
    """Append a batch of results to a JSON-Lines file on disk.

    Using JSONL (one JSON object per line) for incremental writes so we
    never have to read-modify-write the whole file.  The final output is
    reassembled into a single JSON array at the end of processing.
    """
    with open(out_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_processed_doc_ids(tmp_path: Path) -> set:
    """Read an incremental JSONL temp file from a previous (possibly
    interrupted) run and return the set of doc_ids already processed.

    Robust to a truncated final line: the last write may have been cut off
    mid-line if the process was killed, so we skip any line that fails to
    parse instead of aborting.
    """
    processed: set = set()
    if not tmp_path.exists():
        return processed

    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Truncated / partial trailing line from an interrupted run.
                log.warning(f"  Skipping unparseable line in {tmp_path.name} "
                            f"(likely a partial write from a previous run)")
                continue
            if "doc_id" in row:
                processed.add(row["doc_id"])
    return processed


def finalize_output(tmp_path: Path, final_path: Path) -> list[dict]:
    """Read the incremental JSONL file, write the final JSON array, and
    return the full list for summary statistics."""
    all_results: list[dict] = []
    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"  Skipping unparseable line in {tmp_path.name} "
                            f"while finalizing output")
                continue

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Clean up the temporary JSONL file
    tmp_path.unlink(missing_ok=True)
    return all_results


# ============================================================================
# Main annotation pipeline
# ============================================================================
def annotate_trace(problem: str, response: str, call_fn, max_retries: int = 3) -> dict:
    """Annotate a single reasoning trace. Returns the annotation dict."""
    sentences = split_into_sentences(response)
    if not sentences:
        log.warning("  Empty reasoning trace, skipping.")
        return {"sentences": {}, "annotations": {}, "num_sentences": 0, "error": "empty_trace"}

    user_prompt = build_user_prompt(problem, sentences)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_fn(SYSTEM_PROMPT, user_prompt)
            annotations = parse_annotation_json(raw)
            return {
                "sentences": {str(i): s for i, s in enumerate(sentences)},
                "annotations": annotations,
                "num_sentences": len(sentences),
            }
        except Exception as e:
            log.warning(f"  Attempt {attempt}/{max_retries} failed to parse: {e}")
            if attempt < max_retries:
                time.sleep(3 ** attempt)

    return {
        "sentences": {str(i): s for i, s in enumerate(sentences)},
        "annotations": {},
        "num_sentences": len(sentences),
        "error": "parse_failed_after_retries",
    }


def run_pipeline(args):
    # ---- Validate API key ----
    if not args.api_key:
        log.error(
            "No API key provided. Set FRAUNHOFER_API_KEY in your .env file "
            "or pass --api-key on the command line."
        )
        return

    log.info(f"Using endpoint: {args.api_base}")
    log.info(f"Using model:    {args.model}")
    log.info(f"Flush interval: every {args.flush_every} traces")

    # ---- Set up LLM caller ----
    if USE_OPENAI_LIB:
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        def call_fn(system, user):
            return call_llm_openai(
                client, args.model, system, user,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
    else:
        def call_fn(system, user):
            return call_llm_raw(
                args.api_base, args.api_key, args.model,
                system, user,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )

    # ---- Load questions if provided ----
    questions: dict[int, str] = {}
    if args.questions_file:
        questions = load_questions(args.questions_file)
        log.info(f"Loaded {len(questions)} questions from {args.questions_file}")

    # ---- Discover input files ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported = {".json", ".jsonl"}
    if args.input_files:
        input_files = [Path(f) for f in args.input_files]
        missing = [f for f in input_files if not f.is_file()]
        if missing:
            log.error(f"Input file(s) not found: {[str(f) for f in missing]}")
            return
        bad_suffix = [f for f in input_files if f.suffix.lower() not in supported]
        if bad_suffix:
            log.error(f"Unsupported file type(s) (expected .json/.jsonl): {[str(f) for f in bad_suffix]}")
            return
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in supported)
        if not input_files:
            log.error(f"No .json/.jsonl files found in {input_dir}")
            return

    log.info(f"Found {len(input_files)} model output file(s): {[f.name for f in input_files]}")

    # ---- Process each model's outputs ----
    summary = {}

    for filepath in input_files:
        model_name = filepath.stem
        log.info(f"\n{'='*60}")
        log.info(f"Processing model: {model_name}")
        log.info(f"{'='*60}")

        items = load_model_outputs(str(filepath))
        log.info(f"  Loaded {len(items)} traces from {filepath.name}")

        # Paths for incremental writing
        final_path = output_dir / f"{model_name}_annotated.json"
        tmp_path = output_dir / f".{model_name}_annotated.jsonl.tmp"

        # If a final output already exists, this model is fully done — skip it.
        if final_path.exists():
            log.info(f"  Final output already exists at {final_path}, skipping model.")
            continue

        # Resume support: inspect any leftover temp file from a previous
        # interrupted run and skip the samples that were already processed.
        processed_doc_ids = load_processed_doc_ids(tmp_path)
        if processed_doc_ids:
            log.info(f"  Resuming: found {len(processed_doc_ids)} already-processed "
                     f"sample(s) in {tmp_path.name}, continuing from there.")

        buffer: list[dict] = []

        for idx, item in enumerate(items):
            doc_id = item["doc_id"]

            # Skip samples already completed in a previous run.
            if doc_id in processed_doc_ids:
                log.info(f"  [{idx+1}/{len(items)}] doc_id={doc_id}  already processed, skipping.")
                continue

            problem = questions.get(doc_id, "")
            response = item["response"]

            preview = response[:80].replace("\n", " ")
            log.info(f"  [{idx+1}/{len(items)}] doc_id={doc_id}  preview: {preview}...")

            annotation = annotate_trace(
                problem=problem,
                response=response,
                call_fn=call_fn,
                max_retries=args.max_retries,
            )

            buffer.append({
                "doc_id": doc_id,
                "native_id": item["native_id"],
                "label": item["label"],
                "model_answer": item["model_answer"],
                "exact_match": item["exact_match"],
                "num_tokens": item["num_tokens"],
                "response": response,
                **annotation,
            })

            # ---- Flush buffer to disk every N traces ----
            if len(buffer) >= args.flush_every:
                log.info(f"  Flushing {len(buffer)} results to disk...")
                flush_results_to_disk(buffer, tmp_path)
                buffer.clear()

            if args.delay > 0:
                time.sleep(args.delay)

        # ---- Flush remaining results ----
        if buffer:
            log.info(f"  Flushing final {len(buffer)} results to disk...")
            flush_results_to_disk(buffer, tmp_path)
            buffer.clear()

        # ---- Reassemble final JSON output ----
        results = finalize_output(tmp_path, final_path)
        log.info(f"  Saved {len(results)} annotations to {final_path}")

        # ---- Per-model stats ----
        tag_counts: dict[str, int] = {}
        total_sentences = 0
        correct_count = sum(1 for r in results if r.get("exact_match") == 1.0)
        for r in results:
            total_sentences += r.get("num_sentences", 0)
            for ann in r.get("annotations", {}).values():
                if isinstance(ann, dict):
                    for tag in ann.get("function_tags", []):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

        summary[model_name] = {
            "num_traces": len(results),
            "total_sentences": total_sentences,
            "correct": correct_count,
            "accuracy": round(correct_count / len(results), 4) if results else 0,
            "tag_distribution": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        }

    # ---- Save summary ----
    summary_path = output_dir / "annotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\nSummary saved to {summary_path}")
    log.info("\n--- Results per model ---")
    for model_name, stats in summary.items():
        log.info(f"  {model_name}: {stats['num_traces']} traces, "
                 f"{stats['total_sentences']} sentences, "
                 f"accuracy={stats['accuracy']}")
        for tag, count in stats["tag_distribution"].items():
            log.info(f"    {tag}: {count}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Annotate reasoning traces using the Thought Anchors taxonomy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Minimal — reads FRAUNHOFER_API_KEY from .env, uses FhGenie defaults
  python annotate_reasoning_traces.py \\
      --input-dir ./model_outputs \\
      --output-dir ./annotations

  # Override model
  python annotate_reasoning_traces.py \\
      --model MiniMaxAI/MiniMax-M2.5 \\
      --input-dir ./model_outputs

  # With original questions for richer annotation context
  python annotate_reasoning_traces.py \\
      --input-dir ./model_outputs \\
      --questions-file ./gsm8k_test.jsonl

  # Explicit API settings (overrides .env)
  python annotate_reasoning_traces.py \\
      --api-base https://fhgenie.fraunhofer.de/v1 \\
      --api-key <your-key> \\
      --model MiniMaxAI/MiniMax-M2.5 \\
      --input-dir ./model_outputs

  # Custom flush interval (write to disk every 50 traces)
  python annotate_reasoning_traces.py \\
      --input-dir ./model_outputs \\
      --flush-every 50

Environment variables (or .env file):
  FRAUNHOFER_API_KEY   — your FhGenie API key (required)
  FRAUNHOFER_BASE_URL  — API base URL (default: https://fhgenie.fraunhofer.de/v1)

Input format (JSONL, one file per model in --input-dir):
  {"doc_id": 0, "model_output": [{"continuation": "..."}], "label": "18", ...}
  {"doc_id": 1, "model_output": [{"continuation": "..."}], "label": "3", ...}
""",
    )
    parser.add_argument("--api-base", default=DEFAULT_BASE_URL,
                        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help="API key (default: from FRAUNHOFER_API_KEY env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--input-dir", default="/raid/s3/opengptx/behzad_shomali/modalities/loop_MTP_paper/thought_anchor/models_responses/",
                        help="Directory with model output JSONL files (one per model). "
                             "Ignored if --input-files is given.")
    parser.add_argument("--input-files", nargs="+", default=None,
                        help="One or more individual model output JSON/JSONL files. "
                             "Overrides --input-dir if provided.")
    parser.add_argument("--output-dir", default="/raid/s3/opengptx/behzad_shomali/modalities/loop_MTP_paper/thought_anchor/output/",
                        help="Directory to save annotation results")
    parser.add_argument("--questions-file", default=None,
                        help="Optional: JSONL/JSON with original questions (keyed by doc_id)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for annotator LLM (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for annotation response (default: 4096)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Retries per trace on parse failure (default: 5)")
    parser.add_argument("--delay", type=float, default=1,
                        help="Seconds between API calls for rate limiting")
    parser.add_argument("--flush-every", type=int, default=20,
                        help="Flush results to disk every N traces (default: 20)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()