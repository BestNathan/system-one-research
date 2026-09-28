from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset

from common import (
    DATASET,
    DATASET_REVISION,
    answer_probs,
    option_keys,
    parse_json,
    record_from_probs,
    select_cases_per_workflow,
    summarize,
)


class JevBackend:
    def __init__(self, model: str):
        self.model = model
        self.reported_model = model
        key = os.environ["TYPESAFE_API_KEY"]
        self.client = httpx.Client(
            base_url="https://api.typesafe.ai",
            timeout=90,
            headers={"Authorization": f"Bearer {key}"},
        )

    def decide(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], float]:
        last = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                r = self.client.post(
                    "/v1/systemone",
                    json={"model": self.model, "state": state, "questions": questions},
                )
                elapsed = (time.perf_counter() - t0) * 1000
                if r.status_code == 429 or r.status_code >= 500:
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(min(8, 2 ** attempt))
                    continue
                r.raise_for_status()
                body = r.json()
                self.reported_model = str(body.get("model") or self.reported_model)
                return body, elapsed
            except Exception as exc:
                last = exc
                if attempt == 4:
                    raise
                time.sleep(min(8, 2 ** attempt))
        raise last or RuntimeError("Jev request failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases-per-workflow", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
    ds = select_cases_per_workflow(ds, args.cases_per_workflow)
    backend = JevBackend(args.model)

    decision_rows = []
    raw_cases = []
    case_latencies = []
    errors = []

    for case_index, row in enumerate(ds):
        state = parse_json(row["state"])
        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        try:
            body, elapsed_ms = backend.decide(state, questions)
            case_latencies.append(elapsed_ms)
            answers = body["answers"]
            raw_cases.append(
                {
                    "case_index": case_index,
                    "id": row["id"],
                    "workflow": row["workflow"],
                    "elapsed_ms": elapsed_ms,
                    "model": body.get("model"),
                    "answers": answers,
                }
            )

            for qid, qdef in questions.items():
                qtype = qdef["type"]
                keys = option_keys(qdef)
                ans = answers[qid]
                probs = answer_probs(ans, keys, qtype)
                decision_rows.append(
                    record_from_probs(
                        case_index=case_index,
                        case_id=row["id"],
                        workflow=row["workflow"],
                        question_id=qid,
                        question_type=qtype,
                        keys=keys,
                        gold=gold[qid],
                        probs=probs,
                    )
                )
        except Exception as exc:
            errors.append(
                {
                    "case_index": case_index,
                    "id": row["id"],
                    "workflow": row["workflow"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )
            print(f"ERROR case {case_index} {row['id']}: {type(exc).__name__}: {exc}", flush=True)

        if (case_index + 1) % 25 == 0:
            print(f"{args.model}: completed {case_index + 1}/{len(ds)} cases", flush=True)

    summary = {
        "backend": "jev",
        "requested_model": args.model,
        "reported_model": backend.reported_model,
        "kind": "general",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "cases": len(ds),
        "cases_ok": len(raw_cases),
        "errors": len(errors),
        "execution_mode": "hosted API, one shared-state five-question request per case",
        **summarize(decision_rows, case_latencies=case_latencies),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "decisions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in decision_rows) + "\n",
        encoding="utf-8",
    )
    (args.output / "cases.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in raw_cases) + "\n",
        encoding="utf-8",
    )
    (args.output / "errors.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in errors) + ("\n" if errors else ""),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
