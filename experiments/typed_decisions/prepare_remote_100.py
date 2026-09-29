from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET = "LocalLLaMA/typed-decisions"
DATASET_REVISION = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"


def parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-workflow", type=int, default=25)
    args = ap.parse_args()

    ds = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
    counts = defaultdict(int)
    rows = []

    for row in ds:
        workflow = str(row["workflow"])
        if counts[workflow] >= args.per_workflow:
            continue
        counts[workflow] += 1
        rows.append({
            "id": row["id"],
            "workflow": workflow,
            "state": parse_json(row["state"]),
            "questions": parse_json(row["questions"]),
            "gold": parse_json(row["gold"]),
        })

    if len(counts) != 4:
        raise SystemExit(f"expected 4 workflows, got {dict(counts)}")
    if any(n != args.per_workflow for n in counts.values()):
        raise SystemExit(f"unbalanced sample: {dict(counts)}")
    if len(rows) != 4 * args.per_workflow:
        raise SystemExit(f"expected {4 * args.per_workflow} cases, got {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "cases": len(rows),
        "per_workflow": dict(sorted(counts.items())),
        "decisions": sum(len(r["questions"]) for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
