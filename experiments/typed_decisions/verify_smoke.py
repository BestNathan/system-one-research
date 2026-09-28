from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--typed", required=True)
    ap.add_argument("--jev", required=True)
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--decisions", type=int, default=20)
    args = ap.parse_args()

    summaries = {
        "base_laya": load(args.base),
        "laya_typed": load(args.typed),
        "jev": load(args.jev),
    }

    failures = []
    for name, s in summaries.items():
        if s.get("cases") != args.cases:
            failures.append(f"{name}: expected {args.cases} cases, got {s.get('cases')}")
        if s.get("n") != args.decisions:
            failures.append(f"{name}: expected {args.decisions} decisions, got {s.get('n')}")
        if s.get("errors", 0) != 0:
            failures.append(f"{name}: errors={s.get('errors')}")
        if name != "jev" and s.get("dropped_questions", 0) != 0:
            failures.append(f"{name}: dropped_questions={s.get('dropped_questions')}")
        if s.get("accuracy") is None:
            failures.append(f"{name}: missing accuracy")
        if s.get("brier") is None or s.get("ece") is None:
            failures.append(f"{name}: missing calibration metrics")

    if failures:
        raise SystemExit("Smoke validation failed:\n- " + "\n- ".join(failures))

    print("typed-decisions smoke validation passed")
    for name, s in summaries.items():
        print(
            f"{name}: cases={s['cases']} decisions={s['n']} "
            f"accuracy={s['accuracy']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f}"
        )


if __name__ == "__main__":
    main()
