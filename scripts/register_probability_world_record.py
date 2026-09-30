#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    meta = record["meta"]
    record_id = str(meta["record_id"])

    manifest = {"schema_version": 1, "records": []}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    entry = {
        "id": record_id,
        "created_at": meta.get("created_at"),
        "label": meta.get("label") or f"Run {record_id}",
        "note": meta.get("note") or "",
        "source_commit": meta.get("commit_sha"),
        "workflow_run_id": meta.get("run_id"),
        "workflow_run_url": meta.get("run_url"),
        "seed": meta.get("seed"),
        "duration_seconds": meta.get("duration_seconds"),
        "physics_fps": meta.get("physics_fps"),
        "decision_hz": meta.get("decision_hz"),
        "runtime_variant": meta.get("runtime_variant"),
        "control": meta.get("control"),
        "models": {
            "jev": record["summary"]["jev"].get("reported_model"),
            "dashscope": record["summary"]["dashscope"].get("reported_model"),
        },
        "outcomes": {
            "jev": {
                "events": record["summary"]["jev"]["event_process"][
                    "mean_events_per_worldline"
                ],
                "near_miss": bool(
                    record["summary"]["jev"]["outcomes"]["near_miss_rate"]
                ),
                "critical_gap": bool(
                    record["summary"]["jev"]["outcomes"]["critical_gap_rate"]
                ),
                "final_speed_mph": record["summary"]["jev"]["outcomes"][
                    "final_speed_mph"
                ]["mean"],
            },
            "dashscope": {
                "events": record["summary"]["dashscope"]["event_process"][
                    "mean_events_per_worldline"
                ],
                "near_miss": bool(
                    record["summary"]["dashscope"]["outcomes"]["near_miss_rate"]
                ),
                "critical_gap": bool(
                    record["summary"]["dashscope"]["outcomes"]["critical_gap_rate"]
                ),
                "final_speed_mph": record["summary"]["dashscope"]["outcomes"][
                    "final_speed_mph"
                ]["mean"],
            },
        },
        "path": f"runs/{record_id}/record.json",
    }

    records = [
        item
        for item in manifest.get("records", [])
        if str(item.get("id")) != record_id
    ]
    records.append(entry)
    records.sort(
        key=lambda item: item.get("created_at") or "",
        reverse=True,
    )
    manifest["schema_version"] = 1
    manifest["records"] = records

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"registered probability-world record {record_id}")


if __name__ == "__main__":
    main()
