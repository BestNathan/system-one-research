from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import ACTION_ACCEL_MPS2, ACTION_ORDER, QUESTION, RemoteBackend
from run_probabilistic_world import (
    EVENT_ORDER,
    age_recent_history,
    apply_event,
    event_question,
    initial_state,
    sample_distribution,
    usage_tokens,
)
from run_experiment import mph_to_mps, mps_to_mph


DEFAULT_SEED = 20260930


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def relax_lead_vehicle(state: dict[str, Any], dt: float) -> None:
    front = state["road"].get("front_vehicle")
    if front is None:
        return
    current = float(front["speed_mph"])
    desired = float(front.get("desired_speed_mph", current))
    max_delta = 2.0 * dt
    front["speed_mph"] = current + max(-max_delta, min(max_delta, desired - current))


def physics_frame(
    state: dict[str, Any],
    action: str,
    *,
    dt: float,
) -> tuple[float, float | None, bool, bool, bool]:
    speed_before = float(state["ego"]["speed_mph"])
    accel = ACTION_ACCEL_MPS2[action]
    speed_mps = max(0.0, mph_to_mps(speed_before) + accel * dt)
    speed_after = mps_to_mph(speed_mps)
    state["ego"]["speed_mph"] = speed_after
    state["ego"]["longitudinal_acceleration_mps2"] = accel

    front = state["road"].get("front_vehicle")
    gap_after = None
    near_miss = critical_gap = floor_hit = False
    if front is not None:
        relative_mps = mph_to_mps(float(front["speed_mph"])) - speed_mps
        front["distance_m"] = max(
            1.0,
            float(front["distance_m"]) + relative_mps * dt,
        )
        gap_after = float(front["distance_m"])
        near_miss = gap_after < 7.0
        critical_gap = gap_after < 3.0
        floor_hit = gap_after <= 1.000001

    relax_lead_vehicle(state, dt)
    return speed_after, gap_after, near_miss, critical_gap, floor_hit


def run_worldline(
    backend: RemoteBackend,
    worldline_id: int,
    *,
    duration_seconds: int,
    physics_fps: int,
    seed_base: int,
) -> dict[str, Any]:
    event_rng = random.Random(seed_base + worldline_id * 2)
    actor_rng = random.Random(seed_base + worldline_id * 2 + 1)
    state = initial_state()
    state["prediction_horizon_seconds"] = 1.0

    dt = 1.0 / physics_fps
    total_frames = duration_seconds * physics_fps

    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    max_speed = float(state["ego"]["speed_mph"])
    min_gap = float("inf")
    ever_near_miss = False
    ever_critical_gap = False
    ever_floor_hit = False
    in_near_miss = False
    event_count = 0
    first_event_second: int | None = None

    for second in range(duration_seconds):
        if second > 0:
            age_recent_history(state, max_age_steps=5)

        state["time_step"] = second
        state["time_seconds"] = float(second)

        event_q = event_question(state)
        event_keys = list(event_q["criteria"].keys())
        event_probs, event_body, event_ms = backend.decide_choice(
            deepcopy(state),
            "traffic_event",
            event_q,
        )
        in_tok, out_tok = usage_tokens(event_body.get("usage"))
        input_tokens += in_tok
        output_tokens += out_tok

        sampled_event, event_draw = sample_distribution(
            event_probs,
            event_keys,
            event_rng,
        )
        if sampled_event != "no_event":
            event_count += 1
            if first_event_second is None:
                first_event_second = second
        apply_event(state, sampled_event)

        actor_probs, actor_body, actor_ms = backend.decide_choice(
            deepcopy(state),
            "driver_action",
            QUESTION["decision"],
        )
        in_tok, out_tok = usage_tokens(actor_body.get("usage"))
        input_tokens += in_tok
        output_tokens += out_tok

        sampled_action, actor_draw = sample_distribution(
            actor_probs,
            ACTION_ORDER,
            actor_rng,
        )
        if sampled_action == "hard_brake":
            state["recent_history"]["hard_brakes"].append(
                {"kind": "hard_brake", "age_steps": 0}
            )

        frame_start = len(frames)
        speed_before = float(state["ego"]["speed_mph"])
        gap_before = (
            None
            if state["road"].get("front_vehicle") is None
            else float(state["road"]["front_vehicle"]["distance_m"])
        )

        decision_near_miss = False
        decision_critical = False
        decision_floor = False

        for subframe in range(physics_fps):
            frame_index = second * physics_fps + subframe
            speed_after, gap_after, near_miss, critical_gap, floor_hit = physics_frame(
                state,
                sampled_action,
                dt=dt,
            )

            if near_miss and not in_near_miss:
                state["recent_history"]["near_misses"].append(
                    {
                        "kind": "near_miss",
                        "age_steps": 0,
                        "gap_m": gap_after,
                    }
                )
            in_near_miss = near_miss

            ever_near_miss = ever_near_miss or near_miss
            ever_critical_gap = ever_critical_gap or critical_gap
            ever_floor_hit = ever_floor_hit or floor_hit
            decision_near_miss = decision_near_miss or near_miss
            decision_critical = decision_critical or critical_gap
            decision_floor = decision_floor or floor_hit
            max_speed = max(max_speed, speed_after)
            if gap_after is not None:
                min_gap = min(min_gap, gap_after)

            frames.append(
                {
                    "frame": frame_index,
                    "time_seconds": round((frame_index + 1) * dt, 6),
                    "decision_second": second,
                    "subframe": subframe,
                    "speed_mph": round(speed_after, 4),
                    "front_gap_m": (
                        None if gap_after is None else round(gap_after, 4)
                    ),
                    "near_miss": near_miss,
                    "critical_gap": critical_gap,
                    "collision_floor_hit": floor_hit,
                }
            )

        state["recent_history"]["last_action"] = sampled_action

        decisions.append(
            {
                "second": second,
                "frame_start": frame_start,
                "frame_end": len(frames) - 1,
                "event_options": event_keys,
                "event_probabilities": event_probs,
                "sampled_event": sampled_event,
                "event_draw": event_draw,
                "event_argmax": max(event_keys, key=lambda k: event_probs[k]),
                "event_elapsed_ms": event_ms,
                "actor_probabilities": actor_probs,
                "sampled_action": sampled_action,
                "actor_draw": actor_draw,
                "actor_argmax": max(ACTION_ORDER, key=lambda k: actor_probs[k]),
                "actor_elapsed_ms": actor_ms,
                "speed_before_mph": speed_before,
                "speed_after_mph": float(state["ego"]["speed_mph"]),
                "front_gap_before_m": gap_before,
                "front_gap_after_m": (
                    None
                    if state["road"].get("front_vehicle") is None
                    else float(state["road"]["front_vehicle"]["distance_m"])
                ),
                "near_miss": decision_near_miss,
                "critical_gap": decision_critical,
                "collision_floor_hit": decision_floor,
            }
        )

    assert len(frames) == total_frames
    event_counts = Counter(r["sampled_event"] for r in decisions)
    action_counts = Counter(r["sampled_action"] for r in decisions)

    return {
        "worldline_id": worldline_id,
        "event_seed": seed_base + worldline_id * 2,
        "actor_seed": seed_base + worldline_id * 2 + 1,
        "reported_model": backend.reported_model,
        "duration_seconds": duration_seconds,
        "physics_fps": physics_fps,
        "frames": frames,
        "decisions": decisions,
        "summary": {
            "events": event_count,
            "first_event_second": first_event_second,
            "eventful": event_count > 0,
            "final_speed_mph": float(state["ego"]["speed_mph"]),
            "max_speed_mph": max_speed,
            "min_front_gap_m": None if min_gap == float("inf") else min_gap,
            "near_miss": ever_near_miss,
            "critical_gap": ever_critical_gap,
            "collision_floor_hit": ever_floor_hit,
            "hard_brake_count": action_counts["hard_brake"],
            "hard_accelerate_count": action_counts["hard_accelerate"],
            "cut_in_count": event_counts["vehicle_cut_in"],
            "lead_hard_brake_count": event_counts["lead_vehicle_hard_brake"],
            "lead_accelerate_count": event_counts["lead_vehicle_accelerate"],
            "lead_exit_count": event_counts["lead_vehicle_exit"],
            "event_non_argmax_count": sum(
                r["sampled_event"] != r["event_argmax"] for r in decisions
            ),
            "actor_non_argmax_count": sum(
                r["sampled_action"] != r["actor_argmax"] for r in decisions
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

    n = len(completed)
    duration_seconds = int(completed[0]["duration_seconds"])
    physics_fps = int(completed[0]["physics_fps"])
    decisions = duration_seconds
    frames = duration_seconds * physics_fps

    event_samples = Counter()
    event_eligible = Counter()
    event_predicted_mass = Counter()
    actor_samples = Counter()
    actor_predicted_mass = Counter()
    event_latency: list[float] = []
    actor_latency: list[float] = []

    for w in completed:
        for row in w["decisions"]:
            event = row["sampled_event"]
            action = row["sampled_action"]
            event_samples[event] += 1
            actor_samples[action] += 1
            event_latency.append(float(row["event_elapsed_ms"]))
            actor_latency.append(float(row["actor_elapsed_ms"]))

            for key in row["event_options"]:
                event_eligible[key] += 1
                event_predicted_mass[key] += float(row["event_probabilities"][key])
            for key in ACTION_ORDER:
                actor_predicted_mass[key] += float(row["actor_probabilities"][key])

    event_calibration = {}
    for event in EVENT_ORDER:
        eligible = event_eligible[event]
        if eligible <= 0:
            continue
        empirical = event_samples[event] / eligible
        predicted = event_predicted_mass[event] / eligible
        event_calibration[event] = {
            "eligible_calls": eligible,
            "empirical_frequency_when_eligible": empirical,
            "mean_predicted_probability": predicted,
            "difference": empirical - predicted,
        }

    actor_calls = sum(actor_samples.values())
    actor_calibration = {}
    for action in ACTION_ORDER:
        empirical = actor_samples[action] / actor_calls
        predicted = actor_predicted_mass[action] / actor_calls
        actor_calibration[action] = {
            "empirical_frequency": empirical,
            "mean_predicted_probability": predicted,
            "difference": empirical - predicted,
        }

    mins = [
        float(w["summary"]["min_front_gap_m"])
        for w in completed
        if w["summary"]["min_front_gap_m"] is not None
    ]
    finals = [float(w["summary"]["final_speed_mph"]) for w in completed]
    maxes = [float(w["summary"]["max_speed_mph"]) for w in completed]
    events = [int(w["summary"]["events"]) for w in completed]

    return {
        "backend": backend,
        "requested_model": requested_model,
        "reported_model": completed[0].get("reported_model"),
        "worldlines_requested": len(worldlines),
        "worldlines_completed": n,
        "worldline_errors": len(worldlines) - n,
        "duration_seconds": duration_seconds,
        "physics_fps": physics_fps,
        "frames_per_worldline": frames,
        "decision_hz": 1,
        "decisions_per_worldline": decisions,
        "system_one_calls": n * decisions * 2,
        "usage": {
            "input_tokens": sum(int(w["summary"]["input_tokens"]) for w in completed),
            "output_tokens": sum(int(w["summary"]["output_tokens"]) for w in completed),
        },
        "event_process": {
            "eventful_worldline_rate": sum(
                bool(w["summary"]["eventful"]) for w in completed
            ) / n,
            "mean_events_per_worldline": statistics.mean(events),
            "counts": dict(event_samples),
            "sampling_self_consistency": event_calibration,
            "max_abs_frequency_minus_probability": max(
                abs(v["difference"]) for v in event_calibration.values()
            ),
        },
        "actor_process": {
            "counts": dict(actor_samples),
            "sampling_self_consistency": actor_calibration,
            "max_abs_frequency_minus_probability": max(
                abs(v["difference"]) for v in actor_calibration.values()
            ),
        },
        "outcomes": {
            "near_miss_rate": sum(bool(w["summary"]["near_miss"]) for w in completed) / n,
            "critical_gap_rate": sum(
                bool(w["summary"]["critical_gap"]) for w in completed
            ) / n,
            "collision_floor_rate": sum(
                bool(w["summary"]["collision_floor_hit"]) for w in completed
            ) / n,
            "min_gap_m": {
                "mean": statistics.mean(mins) if mins else None,
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
        },
        "latency_ms": {
            "event_mean": statistics.mean(event_latency),
            "event_p95": percentile(event_latency, 0.95),
            "actor_mean": statistics.mean(actor_latency),
            "actor_p95": percentile(actor_latency, 0.95),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "dashscope"], required=True)
    ap.add_argument("--worldlines", type=int, default=1)
    ap.add_argument("--duration-seconds", type=int, default=120)
    ap.add_argument("--physics-fps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if min(args.worldlines, args.duration_seconds, args.physics_fps, args.workers) <= 0:
        raise SystemExit("worldlines, duration-seconds, physics-fps and workers must be positive")

    local = threading.local()

    def get_backend() -> RemoteBackend:
        if not hasattr(local, "backend"):
            local.backend = RemoteBackend(args.backend)
        return local.backend

    def run_one(worldline_id: int) -> dict[str, Any]:
        last = None
        for _ in range(3):
            try:
                return run_worldline(
                    get_backend(),
                    worldline_id,
                    duration_seconds=args.duration_seconds,
                    physics_fps=args.physics_fps,
                    seed_base=args.seed_base,
                )
            except Exception as exc:
                last = exc
                if hasattr(local, "backend"):
                    del local.backend
        return {
            "worldline_id": worldline_id,
            "error": f"{type(last).__name__}: {last}" if last else "unknown error",
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, worldline_id): worldline_id
            for worldline_id in range(args.worldlines)
        }
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 10 == 0 or done == args.worldlines:
                print(f"{args.backend}: {done}/{args.worldlines} worlds", flush=True)

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
            f"worldline failure rate {error_rate:.2%} exceeds 2%"
        )


if __name__ == "__main__":
    main()
