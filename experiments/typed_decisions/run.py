from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from datasets import load_dataset

DATASET = "LocalLLaMA/typed-decisions"
# Current public benchmark revision. Pinning prevents silent fixture drift.
DATASET_REVISION = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def option_keys(qdef: dict[str, Any]) -> list[str]:
    qtype = qdef["type"]
    if qtype == "choice":
        return [str(k) for k in qdef.get("criteria", {}).keys()]
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        criteria = qdef.get("criteria") or []
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unsupported question type: {qtype}")


def gold_distribution(gold: dict[str, Any], keys: list[str], qtype: str) -> np.ndarray:
    probs = gold.get("probabilities") or {}
    probs = {str(k).lower() if qtype == "noul" else str(k): float(v) for k, v in probs.items()}
    if qtype == "noul" and not probs:
        p = gold.get("noul", gold.get("probability_true"))
        if p is not None:
            probs = {"false": 1.0 - float(p), "true": float(p)}
    arr = np.asarray([probs.get(k.lower() if qtype == "noul" else k, 0.0) for k in keys], dtype=float)
    if arr.sum() <= 0:
        label = gold_label(gold, qtype)
        arr = np.zeros(len(keys), dtype=float)
        if label in keys:
            arr[keys.index(label)] = 1.0
    s = arr.sum()
    return arr / s if s > 0 else arr


def gold_label(gold: dict[str, Any], qtype: str) -> str:
    label = gold.get("label")
    if qtype == "noul":
        if isinstance(label, bool):
            return "true" if label else "false"
        return "true" if str(label).lower() in ("true", "1", "yes") else "false"
    if qtype == "score":
        try:
            return str(int(float(label)))
        except Exception:
            return str(label)
    return str(label)


def answer_distribution(answer: dict[str, Any], keys: list[str], qtype: str) -> np.ndarray:
    probs = answer.get("probabilities") or {}
    normalized = {}
    for k, v in probs.items():
        kk = str(k).lower() if qtype == "noul" else str(k)
        try:
            normalized[kk] = float(v)
        except Exception:
            pass

    if qtype == "noul" and not normalized:
        p = answer.get("noul")
        if isinstance(p, (int, float)):
            normalized = {"false": 1.0 - float(p), "true": float(p)}

    arr = np.asarray([normalized.get(k.lower() if qtype == "noul" else k, 0.0) for k in keys], dtype=float)
    s = arr.sum()
    if s <= 0:
        # Last-resort hard answer fallback. This preserves accuracy while making
        # distributional metrics intentionally harsh rather than inventing certainty.
        hard = predicted_label(answer, keys, qtype)
        if hard in keys:
            arr[keys.index(hard)] = 1.0
            s = 1.0
    return arr / s if s > 0 else arr


def predicted_label(answer: dict[str, Any], keys: list[str], qtype: str) -> str | None:
    if qtype == "choice" and answer.get("choice") is not None:
        return str(answer["choice"])
    if qtype == "noul":
        p = answer.get("noul")
        if isinstance(p, (int, float)):
            return "true" if float(p) >= 0.5 else "false"
    if qtype == "score":
        probs = answer.get("probabilities") or {}
        if probs:
            normalized = {str(k): float(v) for k, v in probs.items()}
            return max(keys, key=lambda k: normalized.get(k, -1.0))
        s = answer.get("score")
        if isinstance(s, (int, float)):
            idx = max(0, min(len(keys) - 1, int(round(float(s)))))
            return keys[idx]
    probs = answer.get("probabilities") or {}
    if probs:
        normalized = {str(k).lower() if qtype == "noul" else str(k): float(v) for k, v in probs.items()}
        return max(keys, key=lambda k: normalized.get(k.lower() if qtype == "noul" else k, -1.0))
    return None


class LayaBackend:
    def __init__(self, model: str, device: str):
        import torch
        # Laya's own CPU benchmark guidance: one model forward per call has
        # nothing useful for inter-op parallelism to overlap. Pinning inter-op
        # to 1 avoids severe oversubscription on shared CI runners.
        torch.set_num_threads(min(8, max(1, os.cpu_count() or 2)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        import laya

        self.kind = "specialist" if "typed-decisions" in model else "general"
        self.model = model
        t0 = time.perf_counter()
        self.agent = laya.load(model, device=device)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.reported_model = model

    def decide(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        result = self.agent.predict(state, questions)
        return result, (time.perf_counter() - t0) * 1000


class JevBackend:
    def __init__(self, model: str):
        self.kind = "general"
        self.model = model
        self.reported_model = model
        self.load_ms = 0.0
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
                r = self.client.post("/v1/systemone", json={"model": self.model, "state": state, "questions": questions})
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


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round((len(ys) - 1) * p))]


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    result = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [r for r in rows if lo <= r["confidence"] < hi or (i == bins - 1 and r["confidence"] == 1.0)]
        if not bucket:
            continue
        acc = sum(float(r["correct"]) for r in bucket) / len(bucket)
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        result += len(bucket) / total * abs(acc - conf)
    return result


def summarize(rows: list[dict[str, Any]], case_latencies: list[float]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("status") == "ok"]
    if not scored:
        return {"n": 0}

    by_type = defaultdict(list)
    by_workflow = defaultdict(list)
    for r in scored:
        by_type[r["question_type"]].append(r)
        by_workflow[r["workflow"]].append(r)

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        score_rows = [r for r in items if r["question_type"] == "score" and r.get("score_abs_error") is not None]
        return {
            "n": len(items),
            "accuracy": sum(float(r["correct"]) for r in items) / len(items),
            "soft_accuracy": statistics.mean(r["soft_accuracy"] for r in items),
            "kl": statistics.mean(r["kl"] for r in items),
            "tv": statistics.mean(r["tv"] for r in items),
            "brier": statistics.mean(r["brier"] for r in items),
            "ece": ece(items),
            "score_mae": statistics.mean(r["score_abs_error"] for r in score_rows) if score_rows else None,
            "within_1_level": statistics.mean(float(r["within_1"]) for r in score_rows) if score_rows else None,
        }

    return {
        **block(scored),
        "case_latency_ms": {
            "mean": statistics.mean(case_latencies) if case_latencies else None,
            "p50": percentile(case_latencies, 0.50),
            "p95": percentile(case_latencies, 0.95),
        },
        "by_question_type": {k: block(v) for k, v in sorted(by_type.items())},
        "by_workflow": {k: block(v) for k, v in sorted(by_workflow.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["laya", "jev"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    backend = LayaBackend(args.model, args.device) if args.backend == "laya" else JevBackend(args.model)

    decision_rows = []
    raw_cases = []
    case_latencies = []
    errors = []

    for case_idx, row in enumerate(ds):
        state = parse_json(row["state"])
        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        try:
            body, elapsed_ms = backend.decide(state, questions)
            case_latencies.append(elapsed_ms)
            answers = body["answers"]
            raw_cases.append({
                "case_index": case_idx,
                "id": row["id"],
                "workflow": row["workflow"],
                "elapsed_ms": elapsed_ms,
                "model": body.get("model"),
                "answers": answers,
            })

            for qid, qdef in questions.items():
                qtype = qdef["type"]
                keys = option_keys(qdef)
                g = gold[qid]
                gp = gold_distribution(g, keys, qtype)
                ans = answers[qid]
                pp = answer_distribution(ans, keys, qtype)
                if len(pp) != len(gp) or pp.sum() <= 0:
                    raise ValueError(f"invalid distribution for {row['id']}/{qid}: keys={keys}, answer={ans}")

                gold_hard = gold_label(g, qtype)
                pred_hard = keys[int(np.argmax(pp))]
                conf = float(np.max(pp))
                correct = pred_hard == gold_hard
                eps = 1e-12
                kl = float(np.sum(gp * np.log(np.clip(gp, eps, None) / np.clip(pp, eps, None))))
                tv = float(0.5 * np.abs(gp - pp).sum())
                brier = float(np.square(pp - gp).sum())
                soft_acc = float(np.dot(pp, gp))

                score_abs_error = None
                within_1 = None
                pred_score = None
                gold_score = None
                if qtype == "score":
                    pred_score = ans.get("score")
                    if not isinstance(pred_score, (int, float)):
                        pred_score = float(np.dot(np.arange(len(pp)), pp))
                    gold_score = g.get("score")
                    if not isinstance(gold_score, (int, float)):
                        gold_score = float(np.dot(np.arange(len(gp)), gp))
                    score_abs_error = abs(float(pred_score) - float(gold_score))
                    within_1 = score_abs_error <= 1.0

                decision_rows.append({
                    "status": "ok",
                    "case_index": case_idx,
                    "case_id": row["id"],
                    "workflow": row["workflow"],
                    "question_id": qid,
                    "question_type": qtype,
                    "n_options": len(keys),
                    "gold": gold_hard,
                    "predicted": pred_hard,
                    "correct": correct,
                    "confidence": conf,
                    "gold_probabilities": {k: float(v) for k, v in zip(keys, gp)},
                    "predicted_probabilities": {k: float(v) for k, v in zip(keys, pp)},
                    "soft_accuracy": soft_acc,
                    "kl": kl,
                    "tv": tv,
                    "brier": brier,
                    "gold_score": gold_score,
                    "predicted_score": pred_score,
                    "score_abs_error": score_abs_error,
                    "within_1": within_1,
                })
        except Exception as exc:
            errors.append({
                "case_index": case_idx,
                "id": row["id"],
                "workflow": row["workflow"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
            print(f"ERROR case {case_idx} {row['id']}: {type(exc).__name__}: {exc}", flush=True)

        if (case_idx + 1) % 25 == 0:
            print(f"{args.model}: completed {case_idx + 1}/{len(ds)} cases", flush=True)

    summary = {
        "backend": args.backend,
        "requested_model": args.model,
        "reported_model": backend.reported_model,
        "kind": backend.kind,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "cases": len(ds),
        "cases_ok": len(raw_cases),
        "errors": len(errors),
        "load_ms": backend.load_ms,
        **summarize(decision_rows, case_latencies),
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
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
