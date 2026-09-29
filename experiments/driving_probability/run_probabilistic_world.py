from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import ACTION_ACCEL_MPS2, ACTION_ORDER, QUESTION, RemoteBackend
from run_experiment import mph_to_mps, mps_to_mph


DEFAULT_SEED = 20260930
EVENT_ORDER = [
    "no_event",
    "vehicle_cut_in",
    "lead_vehicle_hard_brake",
    "lead_vehicle_accelerate",
    "lead_vehicle_exit",
]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


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


def sample_distribution(
    probs: dict[str, float],
    keys: list[str],
    rng: random.Random,
) -> tuple[str, float]:
    draw = rng.random()
    cumulative = 0.0
    for key in keys:
        cumulative += probs.get(key, 0.0)
        if draw <= cumulative:
            return key, draw
    return keys[-1], draw


def initial_state() -> dict[str, Any]:
    return {
        "scenario": "probabilistic one-dimensional highway driving",
        "time_step": 0,
        "prediction_horizon_seconds": 1.0,
        "driver_profile": {
            "aggressiveness": 0.82,
            "patience": 0.25,
            "risk_tolerance": 0.78,
            "description": (
                "The ego driver likes to use available space and accepts moderately small "
                "margins, but reacts to visible immediate collision risk."
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
        "traffic_context": {
            "traffic_density": 0.68,
            "adjacent_lane_activity": 0.78,
            "merge_pressure": 0.66,
            "traffic_flow_speed_mph": 46.0,
            "description": (
                "Moderately dense highway traffic with active adjacent-lane merging. "
                "Most one-second intervals contain no abrupt event, but cut-ins and "
                "lead-vehicle speed changes are plausible."
            ),
        },
        "recent_history": {
            "last_action": "keep_speed",
            "events": [],
            "near_misses": [],
            "hard_brakes": [],
        },
    }


def age_recent_history(state: dict[str, Any], max_age_steps: int = 5) -> None:
    history = state["recent_history"]
    for key in ("events", "near_misses", "hard_brakes"):
        aged = []
        for item in history.get(key, []):
            next_item = dict(item)
            next_item["age_steps"] = int(next_item.get("age_steps", 0)) + 1
            if next_item["age_steps"] <= max_age_steps:
                aged.append(next_item)
        history[key] = aged


def event_question(state: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "no_event": (
            "no abrupt external traffic event occurs during the next second; "
            "nearby traffic continues approximately normally"
        ),
    }
    front = state["road"].get("front_vehicle")
    if front is None:
        criteria["vehicle_cut_in"] = (
            "a vehicle from an adjacent lane merges into the ego lane ahead, "
            "creating a new lead vehicle at a relatively short but plausible gap"
        )
    else:
        criteria["lead_vehicle_hard_brake"] = (
            "the current lead vehicle suddenly brakes hard and loses substantial speed"
        )
        criteria["lead_vehicle_accelerate"] = (
            "the current lead vehicle accelerates noticeably and opens the gap"
        )
        criteria["lead_vehicle_exit"] = (
            "the current lead vehicle leaves the ego lane, removing the immediate lead vehicle"
        )

    return {
        "type": "choice",
        "instructions": (
            "Predict the most likely EXOGENOUS traffic event during the next one second. "
            "Do not choose the ego driver's action. Use current traffic state, traffic context, "
            "lead-vehicle state when present, and recent event history. Ordinary no-event intervals "
            "should remain possible and often common; abrupt events should receive probability only "
            "when the supplied state makes them plausible. Return a calibrated probability "
            "distribution over every available event."
        ),
        "criteria": criteria,
    }


def apply_event(state: dict[str, Any], event: str) -> None:
    ego_speed = float(state["ego"]["speed_mph"])
    road = state["road"]
    front = road.get("front_vehicle")

    if event == "vehicle_cut_in":
        road["front_vehicle"] = {
            "distance_m": 22.0,
            "speed_mph": max(18.0, ego_speed - 9.0),
            "desired_speed_mph": max(30.0, ego_speed - 2.0),
            "source": "cut_in",
        }
    elif event == "lead_vehicle_hard_brake" and front is not None:
        front["speed_mph"] = max(5.0, float(front["speed_mph"]) - 14.0)
    elif event == "lead_vehicle_accelerate" and front is not None:
        front["speed_mph"] = min(72.0, float(front["speed_mph"]) + 9.0)
    elif event == "lead_vehicle_exit" and front is not None:
        road["front_vehicle"] = None

    if event != "no_event":
        state["recent_history"]["events"].append(
            {"kind": event, "age_steps": 0}
        )


def relax_lead_vehicle(state: dict[str, Any]) -> None:
    front = state["road"].get("front_vehicle")
    if front is None:
        return
    current = float(front["speed_mph"])
    desired = float(front.get("desired_speed_mph", current))
    delta = max(-2.0, min(2.0, desired - current))
    front["speed_mph"] = current + delta


def update_ego_and_gap(
    state: dict[str, Any],
    action: str,
    dt: float = 1.0,
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

    return speed_after, gap_after, near_miss, critical_gap, floor_hit


def run_worldline(
    backend: RemoteBackend,
    worldline_id: int,
    *,
    steps: int,
    seed_base: int,
) -> dict[str, Any]:
    event_rng = random.Random(seed_base + worldline_id * 2)
    actor_rng = random.Random(seed_base + worldline_id * 2 + 1)
    state = initial_state()

    rows: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    max_speed = float(state["ego"]["speed_mph"])
    min_gap = float("inf")
    ever_near_miss = False
    ever_critical_gap = False
    ever_floor_hit = False
    event_count = 0
    first_event_step: int | None = None

    for step in range(steps):
        if step > 0:
            age_recent_history(state)
        state["time_step"] = step

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
            if first_event_step is None:
                first_event_step = step
        apply_event(state, sampled_event)

        speed_before = float(state["ego"]["speed_mph"])
        gap_before = (
            None
            if state["road"].get("front_vehicle") is None
            else float(state["road"]["front_vehicle"]["distance_m"])
        )

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
        speed_after, gap_after, near_miss, critical_gap, floor_hit = update_ego_and_gap(
            state,
            sampled_action,
        )
        relax_lead_vehicle(state)

        max_speed = max(max_speed, speed_after)
        if gap_after is not None:
            min_gap = min(min_gap, gap_after)

        if near_miss and not ever_near_miss:
            state["recent_history"]["near_misses"].append(
                {"kind": "near_miss", "age_steps": 0, "gap_m": gap_after}
            )
        if sampled_action == "hard_brake":
            state["recent_history"]["hard_brakes"].append(
                {"kind": "hard_brake", "age_steps": 0}
            )

        ever_near_miss = ever_near_miss or near_miss
        ever_critical_gap = ever_critical_gap or critical_gap
        ever_floor_hit = ever_floor_hit or floor_hit
        state["recent_history"]["last_action"] = sampled_action

        rows.append(
            {
                "step": step,
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
                "speed_after_mph": speed_after,
                "front_gap_before_m": gap_before,
                "front_gap_after_m": gap_after,
                "near_miss": near_miss,
                "critical_gap": critical_gap,
                "collision_floor_hit": floor_hit,
            }
        )

    event_counts = Counter(r["sampled_event"] for r in rows)
    action_counts = Counter(r["sampled_action"] for r in rows)
    return {
        "worldline_id": worldline_id,
        "event_seed": seed_base + worldline_id * 2,
        "actor_seed": seed_base + worldline_id * 2 + 1,
        "reported_model": backend.reported_model,
        "steps": rows,
        "summary": {
            "events": event_count,
            "first_event_step": first_event_step,
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
                r["sampled_event"] != r["event_argmax"] for r in rows
            ),
            "actor_non_argmax_count": sum(
                r["sampled_action"] != r["actor_argmax"] for r in rows
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

    event_samples = Counter()
    event_eligible = Counter()
    event_predicted_mass = Counter()
    actor_samples = Counter()
    actor_predicted_mass = Counter()
    event_latencies: list[float] = []
    actor_latencies: list[float] = []
    response_after_event = defaultdict(lambda: Counter(total=0, brake=0, accel=0))
    event_to_near_miss_2step = defaultdict(lambda: Counter(total=0, hit=0))

    steps_per_worldline = len(completed[0]["steps"])
    by_step = [
        {
            "event_samples": Counter(),
            "actor_samples": Counter(),
            "event_probability": Counter(),
            "actor_probability": Counter(),
        }
        for _ in range(steps_per_worldline)
    ]

    for w in completed:
        rows = w["steps"]
        for idx, row in enumerate(rows):
            event = row["sampled_event"]
            action = row["sampled_action"]
            event_samples[event] += 1
            actor_samples[action] += 1
            event_latencies.append(float(row["event_elapsed_ms"]))
            actor_latencies.append(float(row["actor_elapsed_ms"]))

            for key in row["event_options"]:
                event_eligible[key] += 1
                event_predicted_mass[key] += float(row["event_probabilities"][key])
            for key in ACTION_ORDER:
                actor_predicted_mass[key] += float(row["actor_probabilities"][key])

            t = int(row["step"])
            by_step[t]["event_samples"][event] += 1
            by_step[t]["actor_samples"][action] += 1
            for key in row["event_options"]:
                by_step[t]["event_probability"][key] += float(
                    row["event_probabilities"][key]
                )
            for key in ACTION_ORDER:
                by_step[t]["actor_probability"][key] += float(
                    row["actor_probabilities"][key]
                )

            if event != "no_event":
                response_after_event[event]["total"] += 1
                if action in ("hard_brake", "brake"):
                    response_after_event[event]["brake"] += 1
                if action in ("accelerate", "hard_accelerate"):
                    response_after_event[event]["accel"] += 1

                event_to_near_miss_2step[event]["total"] += 1
                horizon = rows[idx : min(len(rows), idx + 3)]
                if any(bool(x["near_miss"]) for x in horizon):
                    event_to_near_miss_2step[event]["hit"] += 1

    event_calibration = {}
    for event in EVENT_ORDER:
        eligible = event_eligible[event]
        if eligible == 0:
            continue
        event_calibration[event] = {
            "eligible_calls": eligible,
            "mean_predicted_probability": event_predicted_mass[event] / eligible,
            "empirical_frequency_when_eligible": event_samples[event] / eligible,
            "difference": (
                event_samples[event] / eligible
                - event_predicted_mass[event] / eligible
            ),
        }

    actor_calls = sum(actor_samples.values())
    actor_calibration = {}
    for action in ACTION_ORDER:
        actor_calibration[action] = {
            "mean_predicted_probability": actor_predicted_mass[action] / actor_calls,
            "empirical_frequency": actor_samples[action] / actor_calls,
            "difference": (
                actor_samples[action] / actor_calls
                - actor_predicted_mass[action] / actor_calls
            ),
        }

    n = len(completed)
    step_summary = []
    for t, item in enumerate(by_step):
        step_summary.append(
            {
                "step": t,
                "event_frequency": {
                    event: item["event_samples"][event] / n
                    for event in EVENT_ORDER
                },
                "actor_frequency": {
                    action: item["actor_samples"][action] / n
                    for action in ACTION_ORDER
                },
                "mean_event_probability": {
                    event: item["event_probability"][event] / max(
                        1,
                        sum(
                            1
                            for w in completed
                            if event in w["steps"][t]["event_options"]
                        ),
                    )
                    for event in EVENT_ORDER
                },
                "mean_actor_probability": {
                    action: item["actor_probability"][action] / n
                    for action in ACTION_ORDER
                },
            }
        )

    mins = [
        float(w["summary"]["min_front_gap_m"])
        for w in completed
        if w["summary"]["min_front_gap_m"] is not None
    ]
    finals = [float(w["summary"]["final_speed_mph"]) for w in completed]
    maxes = [float(w["summary"]["max_speed_mph"]) for w in completed]
    events_per_world = [int(w["summary"]["events"]) for w in completed]
    first_events = [
        int(w["summary"]["first_event_step"])
        for w in completed
        if w["summary"]["first_event_step"] is not None
    ]

    calls = sum(len(w["steps"]) * 2 for w in completed)
    input_tokens = sum(int(w["summary"]["input_tokens"]) for w in completed)
    output_tokens = sum(int(w["summary"]["output_tokens"]) for w in completed)

    response_summary = {}
    for event, counts in response_after_event.items():
        total = counts["total"]
        response_summary[event] = {
            "samples": total,
            "brake_rate_same_step": counts["brake"] / total if total else None,
            "acceleration_rate_same_step": counts["accel"] / total if total else None,
        }

    risk_summary = {}
    for event, counts in event_to_near_miss_2step.items():
        total = counts["total"]
        risk_summary[event] = {
            "samples": total,
            "near_miss_within_2_steps": counts["hit"] / total if total else None,
        }

    return {
        "backend": backend,
        "requested_model": requested_model,
        "reported_model": completed[0].get("reported_model"),
        "worldlines_requested": len(worldlines),
        "worldlines_completed": n,
        "worldline_errors": len(worldlines) - n,
        "steps_per_worldline": steps_per_worldline,
        "system_one_calls": calls,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "event_process": {
            "eventful_worldline_rate": sum(
                bool(w["summary"]["eventful"]) for w in completed
            ) / n,
            "mean_events_per_worldline": statistics.mean(events_per_world),
            "first_event_step": {
                "mean": statistics.mean(first_events) if first_events else None,
                "p50": percentile(first_events, 0.50),
                "p90": percentile(first_events, 0.90),
            },
            "counts": dict(event_samples),
            "sampling_self_consistency": event_calibration,
            "max_abs_frequency_minus_probability": max(
                abs(x["difference"]) for x in event_calibration.values()
            ),
        },
        "actor_process": {
            "counts": dict(actor_samples),
            "sampling_self_consistency": actor_calibration,
            "max_abs_frequency_minus_probability": max(
                abs(x["difference"]) for x in actor_calibration.values()
            ),
        },
        "outcomes": {
            "near_miss_rate": sum(
                bool(w["summary"]["near_miss"]) for w in completed
            ) / n,
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
        "event_response": response_summary,
        "event_risk": risk_summary,
        "latency_ms": {
            "event_mean": statistics.mean(event_latencies),
            "event_p95": percentile(event_latencies, 0.95),
            "actor_mean": statistics.mean(actor_latencies),
            "actor_p95": percentile(actor_latencies, 0.95),
        },
        "by_step": step_summary,
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
        for _ in range(3):
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
            if done % 50 == 0 or done == args.worldlines:
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
