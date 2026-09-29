from __future__ import annotations

import argparse
import json
import random
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import (
    ACTION_ACCEL_MPS2,
    ACTION_ORDER,
    RemoteBackend,
    acceleration_probability,
    braking_probability,
    expected_acceleration,
    js_divergence,
    pairwise_mean_js,
    spearman,
    total_variation,
)


def base_state() -> dict[str, Any]:
    return {
        "scenario": "one-dimensional longitudinal road driving",
        "prediction_horizon_seconds": 1.0,
        "driver_profile": {
            "aggressiveness": 0.85,
            "patience": 0.20,
            "risk_tolerance": 0.80,
            "description": (
                "This driver often accelerates into available space and accepts smaller "
                "margins than an average driver, but still reacts to immediate collision risk."
            ),
        },
        "ego": {
            "speed_mph": 42.0,
            "longitudinal_acceleration_mps2": 0.0,
        },
        "road": {
            "speed_limit_mph": 65.0,
            "surface": "dry",
            "visibility": "good",
            "front_vehicle": None,
        },
        "recent_history": {
            "last_action": "keep_speed",
            "near_miss_last_10s": False,
            "hard_brake_last_10s": False,
        },
    }


def front_vehicle(distance_m: float, speed_mph: float = 32.0) -> dict[str, float]:
    return {"distance_m": distance_m, "speed_mph": speed_mph}


def result_row(label: str, state: dict[str, Any], probs: dict[str, float], elapsed_ms: float):
    return {
        "label": label,
        "state": state,
        "probabilities": probs,
        "argmax": max(ACTION_ORDER, key=lambda a: probs[a]),
        "argmax_probability": max(probs.values()),
        "p_accelerate": acceleration_probability(probs),
        "p_brake": braking_probability(probs),
        "expected_acceleration_mps2": expected_acceleration(probs),
        "elapsed_ms": elapsed_ms,
    }


def call(backend: RemoteBackend, label: str, state: dict[str, Any]):
    probs, body, elapsed = backend.decide(state)
    result = result_row(label, state, probs, elapsed)
    result["reported_model"] = body.get("model")
    result["usage"] = body.get("usage")
    return result


def sample_action(probs: dict[str, float], rng: random.Random):
    u = rng.random()
    cumulative = 0.0
    for action in ACTION_ORDER:
        cumulative += probs[action]
        if u <= cumulative:
            return action, u
    return ACTION_ORDER[-1], u


def mph_to_mps(v: float) -> float:
    return v * 0.44704


def mps_to_mph(v: float) -> float:
    return v / 0.44704


def run_closed_loop(backend: RemoteBackend, steps: int = 16):
    rng = random.Random(20260929)
    state = base_state()
    state["ego"]["speed_mph"] = 22.0
    state["driver_profile"]["aggressiveness"] = 0.90
    state["driver_profile"]["patience"] = 0.15
    state["driver_profile"]["risk_tolerance"] = 0.85
    dt = 1.0

    output = []
    previous_probs = None
    for step in range(steps):
        if step == 4:
            state["road"]["front_vehicle"] = front_vehicle(18.0, 24.0)
            state["recent_history"]["event"] = "slower vehicle just cut in"
        elif step == 8:
            state["road"]["front_vehicle"] = None
            state["recent_history"]["event"] = "lane ahead cleared"
        elif step not in (4, 8):
            state["recent_history"].pop("event", None)

        result = call(backend, f"closed-loop-{step:02d}", deepcopy(state))
        probs = result["probabilities"]
        action, draw = sample_action(probs, rng)
        accel = ACTION_ACCEL_MPS2[action]

        speed_mps = max(
            0.0,
            mph_to_mps(float(state["ego"]["speed_mph"])) + accel * dt,
        )
        old_speed_mph = float(state["ego"]["speed_mph"])
        state["ego"]["speed_mph"] = mps_to_mph(speed_mps)
        state["ego"]["longitudinal_acceleration_mps2"] = accel

        fv = state["road"]["front_vehicle"]
        if fv is not None:
            relative_mps = mph_to_mps(float(fv["speed_mph"])) - speed_mps
            fv["distance_m"] = max(1.0, float(fv["distance_m"]) + relative_mps * dt)

        state["recent_history"]["last_action"] = action
        if action == "hard_brake":
            state["recent_history"]["hard_brake_last_10s"] = True
        if fv is not None and float(fv["distance_m"]) < 7.0:
            state["recent_history"]["near_miss_last_10s"] = True

        result.update(
            {
                "step": step,
                "sample_draw": draw,
                "sampled_action": action,
                "sampled_action_probability": probs[action],
                "sampled_action_is_argmax": action == result["argmax"],
                "speed_before_mph": old_speed_mph,
                "speed_after_mph": state["ego"]["speed_mph"],
                "front_gap_after_m": None if fv is None else fv["distance_m"],
                "js_from_previous": (
                    None
                    if previous_probs is None
                    else js_divergence(previous_probs, probs)
                ),
            }
        )
        previous_probs = probs
        output.append(result)

    return output


def summarize(experiments: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    repeat = experiments["repeatability"]
    gap = experiments["front_gap_sweep"]
    aggr = experiments["aggressiveness_sweep"]
    history = experiments["history_counterfactual"]
    loop = experiments["closed_loop"]

    repeat_noise = pairwise_mean_js([r["probabilities"] for r in repeat])

    adjacent_gap_js = [
        js_divergence(gap[i]["probabilities"], gap[i + 1]["probabilities"])
        for i in range(len(gap) - 1)
    ]
    mean_gap_js = statistics.mean(adjacent_gap_js)
    gaps_numeric = [
        float(r["state"]["road"]["front_vehicle"]["distance_m"])
        for r in gap[:-1]
    ] + [250.0]
    gap_expected_accel = [r["expected_acceleration_mps2"] for r in gap]
    gap_accel_prob = [r["p_accelerate"] for r in gap]

    aggr_values = [
        float(r["state"]["driver_profile"]["aggressiveness"]) for r in aggr
    ]
    aggr_expected_accel = [r["expected_acceleration_mps2"] for r in aggr]
    aggr_accel_prob = [r["p_accelerate"] for r in aggr]

    neutral = history[0]
    near_miss = history[1]
    recent_hard_accel = history[2]

    loop_js = [
        r["js_from_previous"] for r in loop if r["js_from_previous"] is not None
    ]
    sampled = [r["sampled_action_probability"] for r in loop]
    non_argmax = [r for r in loop if not r["sampled_action_is_argmax"]]

    pre_cut = loop[3]
    cut_in = loop[4]
    clear_again = loop[8]

    all_rows = [r for rows in experiments.values() for r in rows]

    return {
        "repeatability_pairwise_mean_js": repeat_noise,
        "front_gap_adjacent_mean_js": mean_gap_js,
        "front_gap_endpoint_js": js_divergence(
            gap[0]["probabilities"], gap[-1]["probabilities"]
        ),
        "front_gap_signal_to_repeat_noise": mean_gap_js / max(repeat_noise, 1e-12),
        "front_gap_vs_expected_acceleration_spearman": spearman(
            gaps_numeric, gap_expected_accel
        ),
        "front_gap_vs_acceleration_probability_spearman": spearman(
            gaps_numeric, gap_accel_prob
        ),
        "aggressiveness_vs_expected_acceleration_spearman": spearman(
            aggr_values, aggr_expected_accel
        ),
        "aggressiveness_vs_acceleration_probability_spearman": spearman(
            aggr_values, aggr_accel_prob
        ),
        "history_near_miss_js_vs_neutral": js_divergence(
            neutral["probabilities"], near_miss["probabilities"]
        ),
        "history_near_miss_tv_vs_neutral": total_variation(
            neutral["probabilities"], near_miss["probabilities"]
        ),
        "history_recent_hard_accel_js_vs_neutral": js_divergence(
            neutral["probabilities"], recent_hard_accel["probabilities"]
        ),
        "closed_loop_mean_consecutive_js": (
            statistics.mean(loop_js) if loop_js else 0.0
        ),
        "closed_loop_max_consecutive_js": max(loop_js) if loop_js else 0.0,
        "closed_loop_non_argmax_samples": len(non_argmax),
        "closed_loop_min_sampled_probability": min(sampled) if sampled else None,
        "closed_loop_mean_sampled_probability": (
            statistics.mean(sampled) if sampled else None
        ),
        "cut_in_js": js_divergence(
            pre_cut["probabilities"], cut_in["probabilities"]
        ),
        "cut_in_delta_braking_probability": (
            cut_in["p_brake"] - pre_cut["p_brake"]
        ),
        "clear_again_js": js_divergence(
            loop[7]["probabilities"], clear_again["probabilities"]
        ),
        "clear_again_delta_acceleration_probability": (
            clear_again["p_accelerate"] - loop[7]["p_accelerate"]
        ),
        "latency_ms_mean": statistics.mean(r["elapsed_ms"] for r in all_rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "dashscope"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    backend = RemoteBackend(args.backend)
    experiments: dict[str, list[dict[str, Any]]] = {}

    reference = base_state()
    experiments["repeatability"] = [
        call(backend, f"repeat-{i}", deepcopy(reference)) for i in range(5)
    ]

    gap_rows = []
    for distance in [8.0, 15.0, 30.0, 60.0, 120.0, None]:
        s = base_state()
        s["ego"]["speed_mph"] = 45.0
        s["road"]["front_vehicle"] = (
            None if distance is None else front_vehicle(distance, 32.0)
        )
        label = "gap-clear" if distance is None else f"gap-{distance:g}m"
        gap_rows.append(call(backend, label, s))
    experiments["front_gap_sweep"] = gap_rows

    aggr_rows = []
    for value in [0.0, 0.25, 0.50, 0.75, 1.0]:
        s = base_state()
        s["driver_profile"]["aggressiveness"] = value
        s["driver_profile"]["patience"] = 1.0 - value
        s["driver_profile"]["risk_tolerance"] = 0.15 + 0.75 * value
        s["ego"]["speed_mph"] = 35.0
        s["road"]["front_vehicle"] = None
        aggr_rows.append(call(backend, f"aggressiveness-{value:.2f}", s))
    experiments["aggressiveness_sweep"] = aggr_rows

    physical = base_state()
    physical["ego"]["speed_mph"] = 42.0
    physical["road"]["front_vehicle"] = front_vehicle(55.0, 40.0)

    neutral = deepcopy(physical)
    neutral["recent_history"] = {
        "last_action": "keep_speed",
        "near_miss_last_10s": False,
        "hard_brake_last_10s": False,
    }

    near_miss = deepcopy(physical)
    near_miss["recent_history"] = {
        "last_action": "hard_brake",
        "near_miss_last_10s": True,
        "hard_brake_last_10s": True,
        "event": "a near collision happened three seconds ago",
    }

    recent_hard_accel = deepcopy(physical)
    recent_hard_accel["recent_history"] = {
        "last_action": "hard_accelerate",
        "near_miss_last_10s": False,
        "hard_brake_last_10s": False,
        "event": "the driver just accelerated strongly into an opening",
    }

    experiments["history_counterfactual"] = [
        call(backend, "history-neutral", neutral),
        call(backend, "history-near-miss", near_miss),
        call(backend, "history-recent-hard-accel", recent_hard_accel),
    ]

    experiments["closed_loop"] = run_closed_loop(backend, 16)

    result = {
        "backend": args.backend,
        "requested_model": backend.model,
        "reported_model": backend.reported_model,
        "seed": 20260929,
        "actions": ACTION_ACCEL_MPS2,
        "experiments": experiments,
    }
    result["metrics"] = summarize(experiments)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
