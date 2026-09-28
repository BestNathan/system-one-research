from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

DATASET = "LocalLLaMA/typed-decisions"
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
        criteria = qdef.get("criteria") or {}
        if isinstance(criteria, list):
            return [str(x) for x in criteria]
        return [str(k) for k in criteria.keys()]
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        criteria = qdef.get("criteria") or []
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unsupported question type: {qtype}")


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


def gold_distribution(gold: dict[str, Any], keys: list[str], qtype: str) -> np.ndarray:
    probs = gold.get("probabilities") or {}
    norm = {}
    for k, v in probs.items():
        kk = str(k).lower() if qtype == "noul" else str(k)
        norm[kk] = float(v)
    if qtype == "noul" and not norm:
        p = gold.get("noul", gold.get("probability_true"))
        if p is not None:
            norm = {"false": 1.0 - float(p), "true": float(p)}
    arr = np.asarray([norm.get(k.lower() if qtype == "noul" else k, 0.0) for k in keys], dtype=float)
    if arr.sum() <= 0:
        label = gold_label(gold, qtype)
        arr = np.zeros(len(keys), dtype=float)
        if label in keys:
            arr[keys.index(label)] = 1.0
    return arr / arr.sum() if arr.sum() > 0 else arr


def normalize_probs(values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def record_from_probs(
    *,
    case_index: int,
    case_id: str,
    workflow: str,
    question_id: str,
    question_type: str,
    keys: list[str],
    gold: dict[str, Any],
    probs: list[float] | np.ndarray,
) -> dict[str, Any]:
    pp = normalize_probs(probs)
    if len(pp) != len(keys) or pp.sum() <= 0:
        raise ValueError(f"invalid probabilities for {case_id}/{question_id}")
    gp = gold_distribution(gold, keys, question_type)
    hard = gold_label(gold, question_type)
    pred_idx = int(np.argmax(pp))
    pred = keys[pred_idx]
    gold_idx = keys.index(hard)
    correct = pred_idx == gold_idx
    conf = float(pp[pred_idx])

    onehot = np.eye(len(keys), dtype=float)[gold_idx]
    brier_hard = float(np.square(pp - onehot).sum())
    brier_soft = float(np.square(pp - gp).sum())
    soft_accuracy = float(np.dot(pp, gp))
    tv = float(0.5 * np.abs(pp - gp).sum())
    eps = 1e-12
    kl = float(np.sum(gp * np.log(np.clip(gp, eps, None) / np.clip(pp, eps, None))))

    pred_score = gold_score = score_abs_error = within_1 = None
    if question_type == "score":
        pred_score = float(np.dot(np.arange(len(pp)), pp))
        gs = gold.get("score")
        gold_score = float(gs) if isinstance(gs, (int, float)) else float(np.dot(np.arange(len(gp)), gp))
        score_abs_error = abs(pred_score - gold_score)
        within_1 = score_abs_error <= 1.0

    return {
        "status": "ok",
        "case_index": case_index,
        "case_id": case_id,
        "workflow": workflow,
        "question_id": question_id,
        "question_type": question_type,
        "n_options": len(keys),
        "gold": hard,
        "predicted": pred,
        "correct": bool(correct),
        "confidence": conf,
        "gold_probabilities": {k: float(v) for k, v in zip(keys, gp)},
        "predicted_probabilities": {k: float(v) for k, v in zip(keys, pp)},
        "soft_accuracy": soft_accuracy,
        "kl": kl,
        "tv": tv,
        "brier": brier_hard,
        "brier_hard": brier_hard,
        "brier_soft": brier_soft,
        "gold_score": gold_score,
        "predicted_score": pred_score,
        "score_abs_error": score_abs_error,
        "within_1": within_1,
    }


def answer_probs(answer: dict[str, Any], keys: list[str], qtype: str) -> np.ndarray:
    probs = answer.get("probabilities") or {}
    norm = {}
    for k, v in probs.items():
        kk = str(k).lower() if qtype == "noul" else str(k)
        try:
            norm[kk] = float(v)
        except Exception:
            pass
    if qtype == "noul" and not norm:
        p = answer.get("noul")
        if isinstance(p, (int, float)):
            norm = {"false": 1.0 - float(p), "true": float(p)}
    arr = np.asarray([norm.get(k.lower() if qtype == "noul" else k, 0.0) for k in keys], dtype=float)
    if arr.sum() <= 0:
        if qtype == "choice" and str(answer.get("choice")) in keys:
            arr[keys.index(str(answer["choice"]))] = 1.0
        elif qtype == "score" and isinstance(answer.get("score"), (int, float)):
            idx = max(0, min(len(keys) - 1, int(round(float(answer["score"])))))
            arr[idx] = 1.0
    return normalize_probs(arr)


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round((len(ys) - 1) * p))]


def ece(rows: list[dict[str, Any]], bins: int = 15) -> float | None:
    if not rows:
        return None
    conf = np.asarray([r["confidence"] for r in rows], dtype=float)
    corr = np.asarray([float(r["correct"]) for r in rows], dtype=float)
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            result += float(sel.mean() * abs(conf[sel].mean() - corr[sel].mean()))
    return result


def metric_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"n": 0}
    score_rows = [r for r in items if r["question_type"] == "score" and r.get("score_abs_error") is not None]
    return {
        "n": len(items),
        "accuracy": statistics.mean(float(r["correct"]) for r in items),
        "soft_accuracy": statistics.mean(r["soft_accuracy"] for r in items),
        "kl": statistics.mean(r["kl"] for r in items),
        "tv": statistics.mean(r["tv"] for r in items),
        "brier": statistics.mean(r["brier_hard"] for r in items),
        "brier_hard": statistics.mean(r["brier_hard"] for r in items),
        "brier_soft": statistics.mean(r["brier_soft"] for r in items),
        "ece": ece(items, 15),
        "score_mae": statistics.mean(r["score_abs_error"] for r in score_rows) if score_rows else None,
        "within_1_level": statistics.mean(float(r["within_1"]) for r in score_rows) if score_rows else None,
    }


def summarize(rows: list[dict[str, Any]], case_latencies: list[float] | None = None, scoring_seconds: float | None = None, cases: int | None = None) -> dict[str, Any]:
    by_type = defaultdict(list)
    by_workflow = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
        by_workflow[r["workflow"]].append(r)
    out = {
        **metric_block(rows),
        "by_question_type": {k: metric_block(v) for k, v in sorted(by_type.items())},
        "by_workflow": {k: metric_block(v) for k, v in sorted(by_workflow.items())},
    }
    if case_latencies is not None:
        out["case_latency_ms"] = {
            "mean": statistics.mean(case_latencies) if case_latencies else None,
            "p50": percentile(case_latencies, 0.50),
            "p95": percentile(case_latencies, 0.95),
        }
    if scoring_seconds is not None:
        out["batched_scoring_seconds"] = scoring_seconds
        out["batched_ms_per_case"] = (1000 * scoring_seconds / cases) if cases else None
        out["batched_decisions_per_second"] = len(rows) / scoring_seconds if scoring_seconds > 0 else None
    return out


def select_cases_per_workflow(ds, per_workflow: int):
    """Deterministically keep the first N cases from each workflow."""
    if per_workflow <= 0:
        return ds
    counts = defaultdict(int)
    indexes = []
    for i, row in enumerate(ds):
        workflow = str(row["workflow"])
        if counts[workflow] >= per_workflow:
            continue
        counts[workflow] += 1
        indexes.append(i)
    return ds.select(indexes)
