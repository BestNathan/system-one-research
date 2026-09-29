from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import ACTION_ACCEL_MPS2, ACTION_ORDER, RemoteBackend
from run_experiment import base_state, front_vehicle, mph_to_mps, mps_to_mph


DEFAULT_SEED = 20260929


def sample_action(probs: dict[str, float], rng: random.Random) -> tuple[str, float]:
    draw = rng.random()
    cumulative = 0.0
    for action in ACTION_ORDER:
        cumulative += probs[action]
        if draw <= cumulative:
            return action, draw
    return ACTION_ORDER[-1], draw


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def initial_state() -> dict[str, Any]:
    state = base_state()
    state["ego"]["speed_mph"] = 22.0
    state["driver_profile"]["aggressiveness"] = 0.90
    state["driver_profile"]["patience"] = 0.15
    state["driver_profile"]["risk_tolerance"] = 0.85
    return state


def apply_exogenous_event(state: dict[str, Any], step: int) -> str | None:
    if step == 4:
        state["road"]["front_vehicle"] = front_vehicle(18.0, 24.0)
        state["recent_history"]["event"] = "slower vehicle just cut in"
        return "cut_in"
    if step == 8:
        state["road"]["front_vehicle"] = None
        state["recent_history"]["event"] = "lane ahead cleared"
        return "lane_clear"
    state["recent_history"].pop("event", None)
    return None


def usage_tokens(usage: Any) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    try:
        input_tokens = int(usage.get("input_tokens") or 0)
    except Exception:
        input_tokens = 0
    try:
        output_tokens = int(usage.get("output_tokens") or 0)
    except Exception:
        output_tokens = 0
    return input_tokens, output_tokens


def run_worldline(
    backend: RemoteBackend,
    worldline_id: int,
    *,
    steps: int,
    seed_base: int,
) -> dict[str, Any]:
    seed = seed_base + worldline_id
    rng = random.Random(seed)
    state = initial_state()
    dt = 1.0

    rows: list[dict[str, Any]] = []
    min_gap = float("inf")
    max_speed = float(state["ego"]["speed_mph"])
    near_miss = False
    collision_floor_hit = False
    critical_gap = False
    input_tokens = 0
    output_tokens = 0

    for step in range(steps):
        event = apply_exogenous_event(state, step)
        gap_before = (
            None
            if state["road"]["front_vehicle"] is None
            else float(state["road"]["front_vehicle"]["distance_m"])
        )
        speed_before = float(state["ego"]["speed_mph"])

        probs, body, elapsed_ms = backend.decide(deepcopy(state))
        in_tok, out_tok = usage_tokens(body.get("usage"))
        input_tokens += in_tok
        output_tokens += out_tok

        action, draw = sample_action(probs, rng)
        accel = ACTION_ACCEL_MPS2[action]
        speed_mps = max(0.0, mph_to_mps(speed_before) + accel * dt)
        speed_after = mps_to_mph(speed_mps)
        state["ego"]["speed_mph"] = speed_after
        state["ego"]["longitudinal_acceleration_mps2"] = accel
        max_speed = max(max_speed, speed_after)

        fv = state["road"]["front_vehicle"]
        gap_after = None
        if fv is not None:
            relative_mps = mph_to_mps(float(fv["speed_mph"])) - speed_mps
            fv["distance_m"] = max(1.0, float(fv["distance_m"]) + relative_mps * dt)
            gap_after = float(fv["distance_m"])
            min_gap = min(min_gap, gap_after)
            near_miss = near_miss or gap_after < 7.0
            critical_gap = critical_gap or gap_after < 3.0
            collision_floor_hit = collision_floor_hit or gap_after <= 1.000001

        state["recent_history"]["last_action"] = action
        if action == "hard_brake":
            state["recent_history"]["hard_brake_last_10s"] = True
        if near_miss:
            state["recent_history"]["near_miss_last_10s"] = True

        rows.append(
            {
                "step": step,
                "event": event,
                "probabilities": probs,
                "sample_draw": draw,
                "sampled_action": action,
                "sampled_probability": probs[action],
                "argmax_action": max(ACTION_ORDER, key=lambda a: probs[a]),
                "speed_before_mph": speed_before,
                "speed_after_mph": speed_after,
                "front_gap_before_m": gap_before,
                "front_gap_after_m": gap_after,
                "elapsed_ms": elapsed_ms,
            }
        )

    actions = Counter(r["sampled_action"] for r in rows)
    return {
        "worldline_id": worldline_id,
        "seed": seed,
        "reported_model": backend.reported_model,
        "steps": rows,
        "summary": {
            "final_speed_mph": float(state["ego"]["speed_mph"]),
            "max_speed_mph": max_speed,
            "min_front_gap_m": None if min_gap == float("inf") else min_gap,
            "near_miss": near_miss,
            "critical_gap": critical_gap,
            "collision_floor_hit": collision_floor_hit,
            "hard_brake_count": actions["hard_brake"],
            "hard_accelerate_count": actions["hard_accelerate"],
            "non_argmax_count": sum(
                r["sampled_action"] != r["argmax_action"] for r in rows
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def aggregate(
    worldlines: list[dict[str, Any]],
    *,
    backend: str,
    requested_model: str,
) -> dict[str, Any]:
    completed = [w for w in worldlines if "error" not in w]
    if not completed:
        raise RuntimeError("no worldlines completed")

    steps = len(completed[0]["steps"])
    action_counts = Counter()
    predicted_mass = Counter()
    latencies: list[float] = []
    step_action_counts = [Counter() for _ in range(steps)]
    step_predicted_mass = [Counter() for _ in range(steps)]
    step_speed = [[] for _ in range(steps)]
    step_gap = [[] for _ in range(steps)]

    for w in completed:
        for r in w["steps"]:
            t = int(r["step"])
            action_counts[r["sampled_action"]] += 1
            step_action_counts[t][r["sampled_action"]] += 1
            latencies.append(float(r["elapsed_ms"]))
            step_speed[t].append(float(r["speed_after_mph"]))
            if r["front_gap_after_m"] is not None:
                step_gap[t].append(float(r["front_gap_after_m"]))
            for action, p in r["probabilities"].items():
                predicted_mass[action] += float(p)
                step_predicted_mass[t][action] += float(p)

    n_calls = sum(len(w["steps"]) for w in completed)
    overall_calibration = {}
    for action in ACTION_ORDER:
        empirical = action_counts[action] / n_calls
        predicted = predicted_mass[action] / n_calls
        overall_calibration[action] = {
            "empirical_frequency": empirical,
            "mean_predicted_probability": predicted,
            "difference": empirical - predicted,
        }

    by_step = []
    n_worldlines = len(completed)
    for t in range(steps):
        by_step.append(
            {
                "step": t,
                "event": "cut_in" if t == 4 else "lane_clear" if t == 8 else None,
                "action_frequency": {
                    a: step_action_counts[t][a] / n_worldlines for a in ACTION_ORDER
                },
                "mean_predicted_probability": {
                    a: step_predicted_mass[t][a] / n_worldlines for a in ACTION_ORDER
                },
                "mean_speed_after_mph": statistics.mean(step_speed[t]),
                "mean_front_gap_after_m": (
                    statistics.mean(step_gap[t]) if step_gap[t] else None
                ),
                "p10_front_gap_after_m": percentile(step_gap[t], 0.10),
            }
        )

    mins = [w["summary"]["min_front_gap_m"] for w in completed]
    mins = [float(v) for v in mins if v is not None]
    finals = [float(w["summary"]["final_speed_mph"]) for w in completed]
    maxes = [float(w["summary"]["max_speed_mph"]) for w in completed]

    input_tokens = sum(int(w["summary"]["input_tokens"]) for w in completed)
    output_tokens = sum(int(w["summary"]["output_tokens"]) for w in completed)
    near_misses = sum(bool(w["summary"]["near_miss"]) for w in completed)
    critical = sum(bool(w["summary"]["critical_gap"]) for w in completed)
    floor_hits = sum(bool(w["summary"]["collision_floor_hit"]) for w in completed)

    return {
        "backend": backend,
        "requested_model": requested_model,
        "reported_model": completed[0].get("reported_model"),
        "worldlines_requested": len(worldlines),
        "worldlines_completed": len(completed),
        "worldline_errors": len(worldlines) - len(completed),
        "steps_per_worldline": steps,
        "calls": n_calls,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "outcomes": {
            "near_miss_rate": near_misses / len(completed),
            "critical_gap_rate": critical / len(completed),
            "collision_floor_rate": floor_hits / len(completed),
            "min_gap_m": {
                "mean": statistics.mean(mins),
                "p10": percentile(mins, 0.10),
                "p50": percentile(mins, 0.50),
                "p90": percentile(mins, 0.90),
            },
            "final_speed_mph": {
                "mean": statistics.mean(finals),
                "p10": percentile(finals, 0.10),
                "p50": percentile(finals, 0.50),
                "p90": percentile(finals, 0.90),
            },
            "max_speed_mph": {
                "mean": statistics.mean(maxes),
                "p90": percentile(maxes, 0.90),
                "p99": percentile(maxes, 0.99),
            },
            "mean_hard_brakes_per_worldline": statistics.mean(
                w["summary"]["hard_brake_count"] for w in completed
            ),
            "mean_hard_accelerates_per_worldline": statistics.mean(
                w["summary"]["hard_accelerate_count"] for w in completed
            ),
            "mean_non_argmax_samples_per_worldline": statistics.mean(
                w["summary"]["non_argmax_count"] for w in completed
            ),
        },
        "sampling_calibration": {
            "overall": overall_calibration,
            "max_abs_frequency_minus_probability": max(
                abs(v["difference"]) for v in overall_calibration.values()
            ),
        },
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "by_step": by_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "dashscope"], required=True)
    ap.add_argument("--worldlines", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.worldlines <= 0 or args.steps <= 0 or args.workers <= 0:
        raise SystemExit("worldlines, steps, and workers must be positive")

    local = threading.local()

    def get_backend() -> RemoteBackend:
        if not hasattr(local, "backend"):
            local.backend = RemoteBackend(args.backend)
        return local.backend

    def run_one(worldline_id: int) -> dict[str, Any]:
        last = None
        for attempt in range(3):
            try:
                return run_worldline(
                    get_backend(),
                    worldline_id,
                    steps=args.steps,
                    seed_base=args.seed_base,
                )
            except Exception as exc:
                last = exc
                if hasattr(local, "backend"):
                    del local.backend
        return {
            "worldline_id": worldline_id,
            "seed": args.seed_base + worldline_id,
            "error": f"{type(last).__name__}: {last}" if last else "unknown error",
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, worldline_id): worldline_id
            for worldline_id in range(args.worldlines)
        }
        completed_count = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == args.worldlines:
                print(
                    f"{args.backend}: {completed_count}/{args.worldlines} worldlines",
                    flush=True,
                )

    results.sort(key=lambda w: w["worldline_id"])
    probe = get_backend()
    summary = aggregate(
        results,
        backend=args.backend,
        requested_model=probe.model,
    )
    summary["seed_base"] = args.seed_base
    summary["workers"] = args.workers

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "worldlines.jsonl").write_text(
        "\n".join(json.dumps(w, ensure_ascii=False) for w in results) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    error_rate = summary["worldline_errors"] / summary["worldlines_requested"]
    if error_rate > 0.02:
        raise SystemExit(
            f"worldline failure rate {error_rate:.2%} exceeds 2%: "
            f"{summary['worldline_errors']}/{summary['worldlines_requested']}"
        )


if __name__ == "__main__":
    main()
