"""Rename runs in loop_MTP_paper_evals to match their source run names in loop_MTP_paper.

Cross-reference: each eval run has a Command containing `--mapping <something>:<source_run_id>`.
We extract the source run id, look up that run in the training project, and copy its display name.
"""

import re

import wandb
from wandb_gql import gql

RENAME_MUTATION = gql(
    """
    mutation RenameRun($id: String!, $name: String!) {
        upsertBucket(input: {id: $id, displayName: $name}) {
            bucket { id displayName }
        }
    }
    """
)

ENTITY = "behzadshomali"
SOURCE_PROJECT = "loop_MTP_paper"
# SOURCE_PROJECT = "Loop_MTP_paper_expert"
# SOURCE_PROJECT = "nemotron_MATH_PartialMTP_Gating"
TARGET_PROJECT = "loop_MTP_paper_evals"

MAPPING_RE = re.compile(r"--mapping\s+\S*?:([A-Za-z0-9]+)")


def extract_command_string(run) -> str | None:
    """Look in metadata and config for anything that looks like the launch command."""
    md = run.metadata or {}
    if isinstance(md, dict):
        args = md.get("args")
        program = md.get("program") or md.get("codePath")
        if args and isinstance(args, list):
            return f"{program or ''} {' '.join(map(str, args))}"
        for key in ("command", "Command"):
            if key in md and isinstance(md[key], str):
                return md[key]

    cfg = dict(run.config) if run.config else {}
    for key in ("command", "Command", "_command"):
        v = cfg.get(key)
        if isinstance(v, str):
            return v
    return None


def extract_source_id(run) -> str | None:
    cmd = extract_command_string(run)
    if not cmd:
        return None
    m = MAPPING_RE.search(cmd)
    return m.group(1) if m else None


def main():
    api = wandb.Api()

    print(f"Fetching source runs from {ENTITY}/{SOURCE_PROJECT}...")
    source_runs = api.runs(f"{ENTITY}/{SOURCE_PROJECT}")
    id_to_name = {r.id: r.name for r in source_runs}
    print(f"  found {len(id_to_name)} source runs")

    print(f"Fetching target runs from {ENTITY}/{TARGET_PROJECT}...")
    target_runs = list(api.runs(f"{ENTITY}/{TARGET_PROJECT}"))
    print(f"  found {len(target_runs)} target runs")

    updated = 0
    skipped_no_id = 0
    skipped_no_match = 0
    skipped_same = 0

    for run in target_runs:
        source_id = extract_source_id(run)
        if not source_id:
            skipped_no_id += 1
            print(f"[no-id]    {run.id}  name={run.name!r}")
            continue
        new_name = id_to_name.get(source_id)
        if not new_name:
            skipped_no_match += 1
            print(f"[no-match] {run.id}  source_id={source_id} not in {SOURCE_PROJECT}")
            continue
        if run.name == new_name:
            skipped_same += 1
            continue
        old = run.name
        api.client.execute(
            RENAME_MUTATION,
            variable_values={"id": run.storage_id, "name": new_name},
        )
        updated += 1
        print(f"[renamed]  {run.id}  {old!r} -> {new_name!r}")

    print("\nSummary:")
    print(f"  renamed:        {updated}")
    print(f"  already correct:{skipped_same}")
    print(f"  no mapping id:  {skipped_no_id}")
    print(f"  id not found:   {skipped_no_match}")


if __name__ == "__main__":
    main()
