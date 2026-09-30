#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def slim_frame(frame):
    return {
        "f": frame["frame"],
        "t": frame["time_seconds"],
        "s": round(float(frame["speed_mph"]), 3),
        "g": (
            None
            if frame["front_gap_m"] is None
            else round(float(frame["front_gap_m"]), 3)
        ),
        "n": bool(frame["near_miss"]),
        "c": bool(frame["critical_gap"]),
        "x": bool(frame["collision_floor_hit"]),
    }


def slim_decision(row):
    return {
        "second": row["second"],
        "frame_start": row["frame_start"],
        "frame_end": row["frame_end"],
        "event_probabilities": row["event_probabilities"],
        "sampled_event": row["sampled_event"],
        "event_argmax": row["event_argmax"],
        "event_draw": round(float(row["event_draw"]), 6),
        "actor_probabilities": row["actor_probabilities"],
        "sampled_action": row["sampled_action"],
        "actor_argmax": row["actor_argmax"],
        "actor_draw": round(float(row["actor_draw"]), 6),
    }


def slim_world(world):
    return {
        "worldline_id": world["worldline_id"],
        "reported_model": world.get("reported_model"),
        "summary": world["summary"],
        "frames": [slim_frame(x) for x in world["frames"]],
        "decisions": [slim_decision(x) for x in world["decisions"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-summary", type=Path, required=True)
    ap.add_argument("--dashscope-summary", type=Path, required=True)
    ap.add_argument("--jev-worldlines", type=Path, required=True)
    ap.add_argument("--dashscope-worldlines", type=Path, required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-worldlines", type=int, default=24)
    args = ap.parse_args()

    js = json.loads(args.jev_summary.read_text(encoding="utf-8"))
    ds = json.loads(args.dashscope_summary.read_text(encoding="utf-8"))
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
            if tag not in tags[i]:
                tags[i].append(tag)

    add(
        [
            i
            for i in ids
            if jw[i]["summary"]["near_miss"] and not dw[i]["summary"]["near_miss"]
        ],
        "Jev-only near miss",
        5,
    )
    add(
        [
            i
            for i in ids
            if dw[i]["summary"]["near_miss"] and not jw[i]["summary"]["near_miss"]
        ],
        "DashScope-only near miss",
        5,
    )

    speed_diff = sorted(
        ids,
        key=lambda i: abs(
            float(jw[i]["summary"]["final_speed_mph"])
            - float(dw[i]["summary"]["final_speed_mph"])
        ),
        reverse=True,
    )
    add(speed_diff, "High speed divergence", 8)

    event_diff = sorted(
        ids,
        key=lambda i: abs(
            int(jw[i]["summary"]["events"]) - int(dw[i]["summary"]["events"])
        ),
        reverse=True,
    )
    add(event_diff, "High event-count divergence", 6)

    safe = [
        i
        for i in ids
        if not jw[i]["summary"]["near_miss"] and not dw[i]["summary"]["near_miss"]
    ]
    rng = random.Random(20260930)
    rng.shuffle(safe)
    add(safe, "Low-risk baseline", 8)

    selected = [i for i in ids if tags[i]]
    selected = sorted(selected, key=lambda i: (-len(tags[i]), i))[
        : args.max_worldlines
    ]
    if ids and ids[0] not in selected:
        selected = [ids[0]] + selected[: max(0, args.max_worldlines - 1)]
        tags[ids[0]].append("Reference worldline")
    selected = sorted(set(selected))

    payload = {
        "meta": {
            "title": "System One Probability World — 2 minute replay",
            "experiment": "two-minute learned-event probabilistic worlds",
            "run_id": args.run_id,
            "run_url": (
                "https://github.com/BestNathan/system-one-research/actions/runs/"
                f"{args.run_id}"
            ),
            "worldlines_total_per_model": js["worldlines_completed"],
            "worldlines_embedded": len(selected),
            "duration_seconds": js["duration_seconds"],
            "physics_fps": js["physics_fps"],
            "frames_per_worldline": js["frames_per_worldline"],
            "decisions_per_worldline": js["decisions_per_worldline"],
            "decision_hz": js["decision_hz"],
        },
        "summary": {"jev": js, "dashscope": ds},
        "pairs": [
            {
                "id": i,
                "tags": tags[i] or ["Reference"],
                "jev": slim_world(jw[i]),
                "dashscope": slim_world(dw[i]),
            }
            for i in selected
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(selected)} representative pairs; "
        f"{js['frames_per_worldline']} frames/worldline"
    )


if __name__ == "__main__":
    main()
