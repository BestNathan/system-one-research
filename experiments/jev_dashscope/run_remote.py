from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import httpx


DEFAULT_DASHSCOPE_URL = (
    "https://llm-cnwlvwjfb3z4aa29.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1/systemone"
)


def load_cases(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def ece(rows, bins=10):
    if not rows:
        return None
    result = 0.0
    n = len(rows)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [
            r for r in rows
            if (lo <= r["confidence"] < hi) or (i == bins - 1 and r["confidence"] == 1.0)
        ]
        if not bucket:
            continue
        acc = sum(float(r["correct"]) for r in bucket) / len(bucket)
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        result += len(bucket) / n * abs(acc - conf)
    return result


class RemoteBackend:
    def __init__(self, backend: str):
        self.backend = backend
        if backend == "jev":
            key = os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise SystemExit("TYPESAFE_API_KEY is required")
            self.url = os.environ.get("JEV_SYSTEMONE_URL", "https://api.typesafe.ai/v1/systemone")
            self.model = os.environ.get("JEV_MODEL", "jev-1.13.0")
            authorization = f"Bearer {key}"
        elif backend == "dashscope":
            key = os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise SystemExit("DASHSCOPE_API_KEY is required")
            self.url = os.environ.get("DASHSCOPE_SYSTEMONE_URL", DEFAULT_DASHSCOPE_URL)
            self.model = os.environ.get("DASHSCOPE_MODEL", "decision-model-preview")
            authorization = f"Bearer {key}"
        else:
            raise ValueError(backend)

        self.client = httpx.Client(
            timeout=90,
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
            },
        )

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
                return response.json(), elapsed_ms
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
    rows = []
    errors = []

    for i, case in enumerate(cases):
        try:
            body, elapsed_ms = backend.decide(case["state"], case["questions"])
            answer = body["answers"]["decision"]
            question = case["questions"]["decision"]
            keys = list(question["criteria"].keys())
            gold = case["expected"]["decision"]
            predicted = str(answer.get("choice"))

            raw_probs = answer.get("probabilities") or {}
            probs = {str(k): float(v) for k, v in raw_probs.items()}
            prob_sum = sum(probs.get(k, 0.0) for k in keys)
            if prob_sum <= 0:
                raise ValueError(f"missing/invalid probabilities: {answer}")
            probs = {k: probs.get(k, 0.0) / prob_sum for k in keys}

            if predicted not in keys:
                predicted = max(keys, key=lambda k: probs[k])
            confidence = float(probs[predicted])
            onehot = {k: 1.0 if k == gold else 0.0 for k in keys}
            brier = sum((probs[k] - onehot[k]) ** 2 for k in keys)

            rows.append({
                "id": case["id"],
                "source": case.get("source"),
                "state": case["state"],
                "expected": gold,
                "predicted": predicted,
                "correct": predicted == gold,
                "confidence": confidence,
                "probabilities": probs,
                "brier": brier,
                "elapsed_ms": elapsed_ms,
                "reported_model": body.get("model"),
                "usage": body.get("usage"),
                "raw_answer": answer,
            })
        except Exception as exc:
            errors.append({
                "id": case["id"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })

        if (i + 1) % 10 == 0 or i + 1 == len(cases):
            print(f"{args.backend}: {i + 1}/{len(cases)}", flush=True)

    latencies = [r["elapsed_ms"] for r in rows]
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    summary = {
        "backend": args.backend,
        "requested_model": backend.model,
        "endpoint": backend.url,
        "cases": len(cases),
        "cases_ok": len(rows),
        "errors": len(errors),
        "accuracy": sum(float(r["correct"]) for r in rows) / len(rows) if rows else None,
        "mean_confidence": statistics.mean(r["confidence"] for r in rows) if rows else None,
        "correct_mean_confidence": statistics.mean(r["confidence"] for r in correct) if correct else None,
        "wrong_mean_confidence": statistics.mean(r["confidence"] for r in wrong) if wrong else None,
        "brier": statistics.mean(r["brier"] for r in rows) if rows else None,
        "ece": ece(rows),
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    (args.output / "errors.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in errors) + ("\n" if errors else ""),
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
