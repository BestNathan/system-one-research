#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slim_step(s):
    return {
        "step": s["step"],
        "event_options": s["event_options"],
        "event_probabilities": s["event_probabilities"],
        "sampled_event": s["sampled_event"],
        "event_argmax": s["event_argmax"],
        "event_draw": round(float(s["event_draw"]), 6),
        "actor_probabilities": s["actor_probabilities"],
        "sampled_action": s["sampled_action"],
        "actor_argmax": s["actor_argmax"],
        "actor_draw": round(float(s["actor_draw"]), 6),
        "speed_before_mph": round(float(s["speed_before_mph"]), 3),
        "speed_after_mph": round(float(s["speed_after_mph"]), 3),
        "front_gap_before_m": (
            None
            if s["front_gap_before_m"] is None
            else round(float(s["front_gap_before_m"]), 3)
        ),
        "front_gap_after_m": (
            None
            if s["front_gap_after_m"] is None
            else round(float(s["front_gap_after_m"]), 3)
        ),
        "near_miss": bool(s["near_miss"]),
        "critical_gap": bool(s["critical_gap"]),
        "collision_floor_hit": bool(s["collision_floor_hit"]),
    }


def slim_world(w):
    return {
        "worldline_id": w["worldline_id"],
        "reported_model": w.get("reported_model"),
        "summary": w["summary"],
        "steps": [slim_step(s) for s in w["steps"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-summary", type=Path, required=True)
    ap.add_argument("--dashscope-summary", type=Path, required=True)
    ap.add_argument("--jev-worldlines", type=Path, required=True)
    ap.add_argument("--dashscope-worldlines", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--max-worldlines", type=int, default=56)
    args = ap.parse_args()

    js = json.loads(args.jev_summary.read_text())
    ds = json.loads(args.dashscope_summary.read_text())
    jw = {
        w["worldline_id"]: w
        for w in load_jsonl(args.jev_worldlines)
        if "error" not in w
    }
    dw = {
        w["worldline_id"]: w
        for w in load_jsonl(args.dashscope_worldlines)
        if "error" not in w
    }
    ids = sorted(set(jw) & set(dw))
    tags = {i: [] for i in ids}

    def add(seq, tag, n):
        for i in seq[:n]:
            tags[i].append(tag)

    add(
        [
            i
            for i in ids
            if jw[i]["summary"]["near_miss"] and not dw[i]["summary"]["near_miss"]
        ],
        "Jev-only near miss",
        8,
    )
    add(
        [
            i
            for i in ids
            if dw[i]["summary"]["near_miss"] and not jw[i]["summary"]["near_miss"]
        ],
        "DashScope-only near miss",
        8,
    )
    add(
        [
            i
            for i in ids
            if dw[i]["summary"]["near_miss"] and jw[i]["summary"]["near_miss"]
        ],
        "Both near miss",
        6,
    )

    speed_diff = sorted(
        ids,
        key=lambda i: abs(
            float(jw[i]["summary"]["final_speed_mph"])
            - float(dw[i]["summary"]["final_speed_mph"])
        ),
        reverse=True,
    )
    add(speed_diff, "High speed divergence", 12)

    event_diff = sorted(
        ids,
        key=lambda i: sum(
            a["sampled_event"] != b["sampled_event"]
            for a, b in zip(jw[i]["steps"], dw[i]["steps"])
        ),
        reverse=True,
    )
    add(event_diff, "High event divergence", 10)

    safe = [
        i
        for i in ids
        if not jw[i]["summary"]["near_miss"] and not dw[i]["summary"]["near_miss"]
    ]
    rng = random.Random(20260930)
    rng.shuffle(safe)
    add(safe, "Low-risk baseline", 12)

    selected = [i for i in ids if tags[i]]
    selected = sorted(selected, key=lambda i: (-len(tags[i]), i))[
        : args.max_worldlines
    ]
    if 0 not in selected:
        selected = [0] + selected[:-1]
        tags[0].append("Reference worldline")
    selected = sorted(set(selected))

    pairs = [
        {
            "id": i,
            "tags": tags[i] or ["Reference"],
            "jev": slim_world(jw[i]),
            "dashscope": slim_world(dw[i]),
        }
        for i in selected
    ]

    payload = {
        "meta": {
            "title": "System One Probability World",
            "experiment": "learned-event probabilistic worlds",
            "run_id": args.run_id,
            "run_url": (
                "https://github.com/BestNathan/system-one-research/actions/runs/"
                f"{args.run_id}"
            ),
            "worldlines_total_per_model": js["worldlines_completed"],
            "worldlines_embedded": len(pairs),
            "steps_per_worldline": js["steps_per_worldline"],
        },
        "summary": {"jev": js, "dashscope": ds},
        "pairs": pairs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(pairs)} representative "
        f"pairs from {len(ids)} matched worldlines"
    )


if __name__ == "__main__":
    main()
