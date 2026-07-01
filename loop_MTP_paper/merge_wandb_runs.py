"""Scan TARGET_PROJECT for runs sharing an EXACT display name and merge them.

Behavior:
- Group runs by `run.name` (display name).
- For each group with > 1 run, pick the run with the most history rows as primary.
- Copy every row of history and every summary key from each non-primary run into
  the primary (assumes metric keys do not collide across duplicates, e.g. one
  duplicate logged `summary/*` metrics, another logged `eval_full/*` metrics).
- Rename non-primary runs with a suffix `(merged-into-<primary_id>)`.
"""

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
TARGET_PROJECT = "loop_MTP_paper_evals"

# Keys returned by run.history() that we don't want to re-log.
_HISTORY_INTERNAL_KEYS = {"_step", "_runtime", "_timestamp", "_wandb"}


def _history_row_count(run) -> int:
    """Cheap-ish history size: rely on summary `_step` when available, else scan."""
    try:
        s = dict(run.summary) if run.summary else {}
        if "_step" in s and isinstance(s["_step"], (int, float)):
            return int(s["_step"]) + 1
    except Exception:
        pass
    return sum(1 for _ in run.scan_history(page_size=1000))


def _to_plain(v):
    """Recursively convert wandb SummarySubDict / mapping / sequence into plain types."""
    if hasattr(v, "_as_dict"):
        v = v._as_dict()
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    return v


def _clean_row(row: dict) -> dict:
    return {
        k: v
        for k, v in row.items()
        if k not in _HISTORY_INTERNAL_KEYS and not k.startswith("_") and v is not None
    }


def _rename(api, run, new_name: str) -> None:
    api.client.execute(
        RENAME_MUTATION,
        variable_values={"id": run.storage_id, "name": new_name},
    )


def merge_group(api, runs: list) -> tuple[int, int]:
    """Merge a group of duplicate-named runs into the one with most history.

    Returns (rows_copied, summary_keys_copied).
    """
    sized = [(r, _history_row_count(r)) for r in runs]
    sized.sort(key=lambda x: x[1], reverse=True)
    primary, primary_rows = sized[0]
    secondaries = [r for r, _ in sized[1:]]

    print(f"\n=== Group {primary.name!r} ({len(runs)} runs) ===")
    for r, n in sized:
        tag = "PRIMARY" if r.id == primary.id else "secondary"
        print(f"  [{tag}] id={r.id}  history_rows~{n}")

    rows_copied = 0
    summary_copied = 0

    wandb_run = wandb.init(
        entity=ENTITY,
        project=TARGET_PROJECT,
        id=primary.id,
        resume="must",
    )
    try:
        existing_summary_keys = set(dict(wandb_run.summary).keys())

        # The primary's most recent step; we'll log secondary rows past this so
        # wandb's monotonic-step requirement is satisfied. Original metric keys
        # are preserved, only the `_step` index is shifted.
        next_step = primary_rows

        for sec in secondaries:
            print(f"  copying history from {sec.id} ...")
            for row in sec.scan_history(page_size=1000):
                cleaned = _clean_row(row)
                if not cleaned:
                    continue
                wandb_run.log(cleaned, step=next_step)
                next_step += 1
                rows_copied += 1

            sec_summary = dict(sec.summary) if sec.summary else {}
            for k, v in sec_summary.items():
                if k in _HISTORY_INTERNAL_KEYS or k.startswith("_"):
                    continue
                if k in existing_summary_keys:
                    continue
                wandb_run.summary[k] = _to_plain(v)
                existing_summary_keys.add(k)
                summary_copied += 1

            # Rename immediately so a later crash in this group doesn't cause
            # a re-run to double-copy this secondary's data.
            new_name = f"{sec.name} (merged-into-{primary.id})"
            try:
                _rename(api, sec, new_name)
                print(f"  renamed {sec.id}: -> {new_name!r}")
            except Exception as e:
                print(f"  WARN: failed to rename {sec.id}: {e}")
    finally:
        wandb_run.finish()

    return rows_copied, summary_copied


def main():
    api = wandb.Api()

    print(f"Fetching runs from {ENTITY}/{TARGET_PROJECT}...")
    runs = list(api.runs(f"{ENTITY}/{TARGET_PROJECT}"))
    print(f"  found {len(runs)} runs")

    groups: dict[str, list] = {}
    for r in runs:
        groups.setdefault(r.name, []).append(r)

    dup_groups = {name: rs for name, rs in groups.items() if len(rs) > 1}
    print(f"  {len(dup_groups)} name(s) have duplicates")

    total_rows = 0
    total_summary = 0
    for rs in dup_groups.values():
        rows, summ = merge_group(api, rs)
        total_rows += rows
        total_summary += summ

    print("\nSummary:")
    print(f"  duplicate groups merged:  {len(dup_groups)}")
    print(f"  history rows copied:      {total_rows}")
    print(f"  summary keys copied:      {total_summary}")


if __name__ == "__main__":
    main()
