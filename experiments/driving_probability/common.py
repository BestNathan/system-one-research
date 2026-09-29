from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx


DASHSCOPE_URL = (
    "https://llm-cnwlvwjfb3z4aa29.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1/systemone"
)

ACTION_ACCEL_MPS2 = {
    "hard_brake": -4.5,
    "brake": -2.0,
    "keep_speed": 0.0,
    "accelerate": 1.5,
    "hard_accelerate": 3.0,
}

ACTION_ORDER = list(ACTION_ACCEL_MPS2)

QUESTION = {
    "decision": {
        "type": "choice",
        "instructions": (
            "Predict what this specific driver is most likely to do during the next one second. "
            "Model behavior, not the idealized safest action. Use the driver profile, current "
            "physical state, road context, and recent history. Collision risk should strongly "
            "increase braking probability, while an aggressive driver on a clear road may prefer "
            "acceleration. Return a calibrated probability distribution over all actions."
        ),
        "criteria": {
            "hard_brake": "strong emergency-like deceleration because immediate risk is high",
            "brake": "moderate deceleration to reduce speed or increase safety margin",
            "keep_speed": "roughly maintain current longitudinal speed",
            "accelerate": "moderate acceleration when there is useful safe space ahead",
            "hard_accelerate": "strong acceleration, plausible mainly for aggressive opportunistic driving when safe",
        },
    }
}


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

    def decide(self, state: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any], float]:
        payload = {"model": self.model, "state": state, "questions": QUESTION}
        last: Exception | None = None
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
                answer = body["answers"]["decision"]
                probs = normalize_answer_probabilities(answer)
                return probs, body, elapsed_ms
            except Exception as exc:
                last = exc
                if attempt == 4:
                    raise
                time.sleep(min(8, 2 ** attempt))
        raise last or RuntimeError("request failed")


def normalize_answer_probabilities(answer: dict[str, Any]) -> dict[str, float]:
    raw = answer.get("probabilities") or {}
    probs = {}
    for action in ACTION_ORDER:
        try:
            probs[action] = float(raw.get(action, 0.0))
        except Exception:
            probs[action] = 0.0

    total = sum(probs.values())
    if total <= 0:
        chosen = str(answer.get("choice", ""))
        if chosen not in probs:
            raise ValueError(f"missing usable probability distribution: {answer}")
        probs[chosen] = 1.0
        total = 1.0

    return {k: v / total for k, v in probs.items()}


def expected_acceleration(probs: dict[str, float]) -> float:
    return sum(probs[a] * ACTION_ACCEL_MPS2[a] for a in ACTION_ORDER)


def acceleration_probability(probs: dict[str, float]) -> float:
    return probs["accelerate"] + probs["hard_accelerate"]


def braking_probability(probs: dict[str, float]) -> float:
    return probs["brake"] + probs["hard_brake"]


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    m = {k: 0.5 * (p[k] + q[k]) for k in ACTION_ORDER}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k in ACTION_ORDER:
            if a[k] > 0:
                total += a[k] * math.log(a[k] / max(b[k], 1e-15))
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in ACTION_ORDER)


def pairwise_mean_js(distributions: list[dict[str, float]]) -> float:
    values = []
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            values.append(js_divergence(distributions[i], distributions[j]))
    return sum(values) / len(values) if values else 0.0


def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and xs[order[end]] == xs[order[pos]]:
            end += 1
        avg_rank = (pos + end - 1) / 2 + 1
        for k in range(pos, end):
            ranks[order[k]] = avg_rank
        pos = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))
