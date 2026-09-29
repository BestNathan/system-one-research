from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx

from common import answer_probs, option_keys, record_from_probs, summarize


DASHSCOPE_URL = (
    "https://llm-cnwlvwjfb3z4aa29.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1/systemone"
)


def load_cases(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RemoteBackend:
    def __init__(self, backend: str):
        self.backend = backend
        if backend == "jev":
            key = os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise SystemExit("TYPESAFE_API_KEY is required")
            self.url = os.environ.get(
                "JEV_SYSTEMONE_URL", "https://api.typesafe.ai/v1/systemone"
            )
            self.model = os.environ.get("JEV_MODEL", "jev-1.13.0")
        elif backend == "dashscope":
            key = os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise SystemExit("DASHSCOPE_API_KEY is required")
            self.url = os.environ.get("DASHSCOPE_SYSTEMONE_URL", DASHSCOPE_URL)
            self.model = os.environ.get("DASHSCOPE_MODEL", "decision-model-preview")
        else:
            raise ValueError(backend)

        self.client = httpx.Client(
            timeout=90,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        self.reported_model = self.model

    def decide(self, state, questions):
        payload = {"model": self.model, "state": state, "questions": questions}
        last = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                response = self.client.post(self.url, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if response.status_code == 429 or response.status_code >= 500:
                    last = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                    time.sleep(min(8, 2 ** attempt))
                    continue
                response.raise_for_status()
                body = response.json()
                self.reported_model = str(body.get("model") or self.reported_model)
                return body, elapsed_ms
            except Exception as exc:
                last = exc
                if attempt == 4:
                    raise
                time.sleep(min(8, 2 ** attempt))
        raise last or RuntimeError("request failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "dashscope"], required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cases = load_cases(args.cases)
    backend = RemoteBackend(args.backend)
    decisions = []
    raw_cases = []
    errors = []
    latencies = []

    for case_index, case in enumerate(cases):
        try:
            body, elapsed_ms = backend.decide(case["state"], case["questions"])
            answers = body["answers"]
            latencies.append(elapsed_ms)
            raw_cases.append({
                "case_index": case_index,
                "id": case["id"],
                "workflow": case["workflow"],
                "elapsed_ms": elapsed_ms,
                "reported_model": body.get("model"),
                "usage": body.get("usage"),
                "answers": answers,
            })

            for qid, qdef in case["questions"].items():
                qtype = qdef["type"]
                keys = option_keys(qdef)
                ans = answers[qid]
                probs = answer_probs(ans, keys, qtype)
                decisions.append(record_from_probs(
                    case_index=case_index,
                    case_id=case["id"],
                    workflow=case["workflow"],
                    question_id=qid,
                    question_type=qtype,
                    keys=keys,
                    gold=case["gold"][qid],
                    probs=probs,
                ))
        except Exception as exc:
            errors.append({
                "case_index": case_index,
                "id": case["id"],
                "workflow": case["workflow"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })

        if (case_index + 1) % 10 == 0 or case_index + 1 == len(cases):
            print(f"{args.backend}: {case_index + 1}/{len(cases)}", flush=True)

    summary = {
        "backend": args.backend,
        "requested_model": backend.model,
        "reported_model": backend.reported_model,
        "endpoint": backend.url,
        "cases": len(cases),
        "cases_ok": len(raw_cases),
        "errors": len(errors),
        "execution_mode": "hosted API, one shared-state multi-question request per case",
        **summarize(decisions, case_latencies=latencies),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "decisions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in decisions)
        + ("\n" if decisions else ""),
        encoding="utf-8",
    )
    (args.output / "cases.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in raw_cases)
        + ("\n" if raw_cases else ""),
        encoding="utf-8",
    )
    (args.output / "errors.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in errors)
        + ("\n" if errors else ""),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        raise SystemExit(f"{args.backend}: {len(errors)} cases failed")


if __name__ == "__main__":
    main()
