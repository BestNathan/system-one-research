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

from common import RemoteBackend
from run_experiment import mph_to_mps, mps_to_mph
from run_probabilistic_world import sample_distribution, usage_tokens


DEFAULT_SEED = 20260930
RUNTIME_VARIANT = "v3c-safe-cruise-following"

ACTION_ACCEL_MPS2 = {
    "hard_brake": -4.5,
    "brake": -2.0,
    "coast": -0.7,
    "keep_speed": 0.0,
    "accelerate": 0.9,
    "hard_accelerate": 1.6,
}
ACTION_ORDER = list(ACTION_ACCEL_MPS2)

LOW_SPEED_BRAKE_MASK_MPH = 0.5
ACTION_PREDICTION_HORIZON_SECONDS = 2.0
RECENT_ACTION_MAX_AGE_SECONDS = 2
RECENT_EVENT_MAX_AGE_SECONDS = 5
LEAD_RELAX_MPH_PER_SECOND = 1.5
LEAD_HARD_BRAKE_DELTA_MPH = 12.0
MAX_RELEVANT_EXTRA_GAP_M = 80.0

ACTION_CRITERIA = {
    "hard_brake": "strong emergency-like deceleration when the safe following envelope is closing quickly",
    "brake": "moderate deceleration to rebuild following margin or reduce overspeed",
    "coast": "gentle deceleration without active braking, useful for smooth gap or speed correction",
    "keep_speed": "hold approximately the current longitudinal speed",
    "accelerate": "gentle acceleration toward the cruise target when safe space is available",
    "hard_accelerate": "stronger acceleration used only when well below target speed with ample safe space",
}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def initial_state(
    *,
    target_speed_mph: float,
    min_gap_m: float,
    physics_fps: int,
) -> dict[str, Any]:
    target_gap_m = min_gap_m + 20.0
    free_gap_m = min_gap_m + 50.0
    return {
        "scenario": "probabilistic adaptive cruise control on a one-dimensional highway",
        "time_step": 0,
        "time_seconds": 0.0,
        "prediction_horizon_seconds": 1.0,
        "controller_profile": {
            "description": (
                "A smooth adaptive-cruise controller. Safety distance is a hard runtime "
                "constraint. Within the safe region it prefers low-jerk corrections, stable "
                "speed near the cruise target, and a comfortable following gap."
            ),
            "speed_stability_priority": 0.9,
            "smoothness_priority": 0.85,
            "following_stability_priority": 0.95,
        },
        "ego": {
            "speed_mph": target_speed_mph,
            "longitudinal_acceleration_mps2": 0.0,
        },
        "road": {
            "speed_limit_mph": max(90.0, target_speed_mph + 10.0),
            "surface": "dry",
            "visibility": "good",
            "front_vehicle": {
                "distance_m": target_gap_m + 10.0,
                "speed_mph": max(45.0, target_speed_mph - 8.0),
                "desired_speed_mph": max(55.0, target_speed_mph - 4.0),
                "source": "initial_following_baseline",
            },
        },
        "traffic_context": {
            "traffic_density": 0.42,
            "adjacent_lane_activity": 0.48,
            "merge_pressure": 0.34,
            "traffic_flow_speed_mph": target_speed_mph - 2.0,
            "description": (
                "Steady highway traffic with an initial slower lead vehicle so the adaptive "
                "cruise controller must demonstrate following-distance regulation from the "
                "start. Most one-second intervals are uneventful; lead speed changes or lane "
                "exit remain plausible."
            ),
        },
        "cruise": {
            "target_speed_mph": target_speed_mph,
            "speed_deadband_mph": 2.0,
            "min_gap_m": min_gap_m,
            "target_gap_m": target_gap_m,
            "free_gap_m": free_gap_m,
        },
        "safety_runtime": {
            "variant": RUNTIME_VARIANT,
            "physics_fps": physics_fps,
            "action_prediction_horizon_seconds": ACTION_PREDICTION_HORIZON_SECONDS,
            "hard_invariant": "front_gap_m >= min_gap_m whenever a front vehicle exists",
            "shield": "per-frame control barrier",
            "speed_envelope": (
                "dynamic action-space shaping that restores the cruise deadband "
                "without weakening the minimum-gap invariant"
            ),
        },
        "recent_history": {
            "events": [],
            "actions": [],
            "safety_overrides": [],
        },
        "control_state": {},
    }


def age_history(state: dict[str, Any]) -> None:
    history = state["recent_history"]
    for key, max_age in (
        ("events", RECENT_EVENT_MAX_AGE_SECONDS),
        ("actions", RECENT_ACTION_MAX_AGE_SECONDS),
        ("safety_overrides", RECENT_EVENT_MAX_AGE_SECONDS),
    ):
        aged = []
        for item in history.get(key, []):
            next_item = dict(item)
            next_item["age_seconds"] = int(next_item.get("age_seconds", 0)) + 1
            if next_item["age_seconds"] <= max_age:
                aged.append(next_item)
        history[key] = aged


def relax_lead_vehicle(state: dict[str, Any], dt: float) -> None:
    front = state["road"].get("front_vehicle")
    if front is None:
        return
    current = float(front["speed_mph"])
    desired = float(front.get("desired_speed_mph", current))
    max_delta = LEAD_RELAX_MPH_PER_SECOND * dt
    front["speed_mph"] = current + max(-max_delta, min(max_delta, desired - current))


def predict_min_gap(
    state: dict[str, Any],
    *,
    accel_mps2: float,
    horizon_seconds: float,
    physics_fps: int,
    lead_speed_override_mph: float | None = None,
) -> float:
    front = state["road"].get("front_vehicle")
    if front is None:
        return float("inf")

    dt = 1.0 / physics_fps
    steps = max(1, round(horizon_seconds * physics_fps))
    gap = float(front["distance_m"])
    ego_mps = mph_to_mps(float(state["ego"]["speed_mph"]))
    lead_mps = mph_to_mps(
        float(front["speed_mph"])
        if lead_speed_override_mph is None
        else lead_speed_override_mph
    )
    minimum = gap

    for _ in range(steps):
        ego_mps = max(0.0, ego_mps + accel_mps2 * dt)
        gap += (lead_mps - ego_mps) * dt
        minimum = min(minimum, gap)

    return minimum


def refresh_control_state(state: dict[str, Any], physics_fps: int) -> None:
    ego_speed = float(state["ego"]["speed_mph"])
    cruise = state["cruise"]
    front = state["road"].get("front_vehicle")

    control: dict[str, Any] = {
        "speed_error_mph": float(cruise["target_speed_mph"]) - ego_speed,
        "speed_in_deadband": abs(
            ego_speed - float(cruise["target_speed_mph"])
        ) <= float(cruise["speed_deadband_mph"]),
        "front_vehicle_present": front is not None,
    }

    if front is None:
        control.update(
            {
                "gap_m": None,
                "gap_margin_m": None,
                "target_gap_error_m": None,
                "relative_speed_mph": None,
                "closing_speed_mph": 0.0,
                "ttc_seconds": None,
                "keep_speed_predicted_min_gap_2s_m": None,
            }
        )
    else:
        gap = float(front["distance_m"])
        lead_speed = float(front["speed_mph"])
        closing_mph = max(0.0, ego_speed - lead_speed)
        closing_mps = mph_to_mps(closing_mph)
        ttc = gap / closing_mps if closing_mps > 1e-6 else None
        control.update(
            {
                "gap_m": gap,
                "gap_margin_m": gap - float(cruise["min_gap_m"]),
                "target_gap_error_m": gap - float(cruise["target_gap_m"]),
                "relative_speed_mph": lead_speed - ego_speed,
                "closing_speed_mph": closing_mph,
                "ttc_seconds": ttc,
                "keep_speed_predicted_min_gap_2s_m": predict_min_gap(
                    state,
                    accel_mps2=0.0,
                    horizon_seconds=ACTION_PREDICTION_HORIZON_SECONDS,
                    physics_fps=physics_fps,
                ),
            }
        )

    state["control_state"] = control


def hard_brake_event_feasible(state: dict[str, Any], physics_fps: int) -> bool:
    front = state["road"].get("front_vehicle")
    if front is None:
        return False

    min_gap = float(state["cruise"]["min_gap_m"])
    gap = float(front["distance_m"])
    if gap < min_gap + 12.0:
        return False

    shocked_lead_speed = max(
        20.0,
        float(front["speed_mph"]) - LEAD_HARD_BRAKE_DELTA_MPH,
    )
    predicted = predict_min_gap(
        state,
        accel_mps2=ACTION_ACCEL_MPS2["hard_brake"],
        horizon_seconds=1.5,
        physics_fps=physics_fps,
        lead_speed_override_mph=shocked_lead_speed,
    )
    return predicted >= min_gap + 0.5


def event_question(state: dict[str, Any], physics_fps: int) -> dict[str, Any]:
    criteria = {
        "no_event": (
            "no abrupt external event occurs during the next second; surrounding traffic "
            "continues approximately normally"
        )
    }

    front = state["road"].get("front_vehicle")
    if front is None:
        criteria["vehicle_cut_in"] = (
            "an adjacent-lane vehicle merges ahead at a runtime-constrained safe insertion "
            "gap, becoming the new lead vehicle"
        )
    else:
        if hard_brake_event_feasible(state, physics_fps):
            criteria["lead_vehicle_hard_brake"] = (
                "the lead vehicle brakes sharply; this option is exposed only while the "
                "runtime can still preserve the hard minimum-gap invariant"
            )
        criteria["lead_vehicle_accelerate"] = (
            "the lead vehicle accelerates noticeably and opens the following gap"
        )
        criteria["lead_vehicle_exit"] = (
            "the lead vehicle leaves the ego lane, making the road immediately clear"
        )

    return {
        "type": "choice",
        "instructions": (
            "Predict the most likely EXOGENOUS highway traffic event during the next one "
            "second. This is a steady cruise scenario, so ordinary no-event intervals should "
            "usually be common. Do not choose the ego controller action. Use traffic state, "
            "lead-vehicle state, and recent event history. Return a calibrated probability "
            "distribution over every currently available event."
        ),
        "criteria": criteria,
    }


def apply_event(state: dict[str, Any], event: str) -> None:
    road = state["road"]
    front = road.get("front_vehicle")
    cruise = state["cruise"]
    target_speed = float(cruise["target_speed_mph"])
    min_gap = float(cruise["min_gap_m"])
    target_gap = float(cruise["target_gap_m"])

    if event == "vehicle_cut_in":
        road["front_vehicle"] = {
            "distance_m": max(target_gap + 5.0, min_gap + 25.0),
            "speed_mph": max(45.0, target_speed - 12.0),
            "desired_speed_mph": max(55.0, target_speed - 5.0),
            "source": "cut_in",
        }
    elif event == "lead_vehicle_hard_brake" and front is not None:
        front["speed_mph"] = max(
            20.0,
            float(front["speed_mph"]) - LEAD_HARD_BRAKE_DELTA_MPH,
        )
    elif event == "lead_vehicle_accelerate" and front is not None:
        front["speed_mph"] = min(
            target_speed + 10.0,
            float(front["speed_mph"]) + 8.0,
        )
    elif event == "lead_vehicle_exit" and front is not None:
        road["front_vehicle"] = None

    if event != "no_event":
        state["recent_history"]["events"].append(
            {"kind": event, "age_seconds": 0}
        )


def one_second_speed_mph(speed_mph: float, action: str) -> float:
    speed_mps = max(
        0.0,
        mph_to_mps(speed_mph) + ACTION_ACCEL_MPS2[action],
    )
    return mps_to_mph(speed_mps)


def apply_speed_envelope(
    state: dict[str, Any],
    gap_safe_actions: list[str],
) -> list[str]:
    """Keep the stochastic policy inside a cruise-speed restoring envelope.

    Distance safety remains the hard constraint. This second envelope is a control
    objective: when the road is clear it does not let the probabilistic policy keep
    accelerating away from the target or remain parked far below it.
    """
    speed = float(state["ego"]["speed_mph"])
    target = float(state["cruise"]["target_speed_mph"])
    deadband = float(state["cruise"]["speed_deadband_mph"])
    front = state["road"].get("front_vehicle")

    lower = target - deadband
    upper = target + deadband
    selected = list(gap_safe_actions)
    mode = "following-safety-first" if front is not None else "cruise-deadband"

    if speed > upper:
        restoring = [
            a for a in selected if ACTION_ACCEL_MPS2[a] < 0.0
        ]
        if restoring:
            selected = restoring
            mode = "overspeed-restore"
    elif front is None and speed < lower:
        restoring = [
            a for a in selected if ACTION_ACCEL_MPS2[a] > 0.0
        ]
        if restoring:
            selected = restoring
            mode = "underspeed-restore"
    elif front is None:
        band_preserving = [
            a
            for a in selected
            if lower - 1e-9
            <= one_second_speed_mph(speed, a)
            <= upper + 1e-9
        ]
        if band_preserving:
            selected = band_preserving
            mode = "cruise-band-hold"
    elif speed > target:
        non_accelerating = [
            a for a in selected if ACTION_ACCEL_MPS2[a] <= 0.0
        ]
        if non_accelerating:
            selected = non_accelerating
            mode = "following-no-overspeed"

    state["control_state"]["gap_safe_actions"] = list(gap_safe_actions)
    state["control_state"]["speed_envelope_actions"] = list(selected)
    state["control_state"]["speed_envelope_mode"] = mode
    return selected


def safe_action_space(
    state: dict[str, Any],
    physics_fps: int,
) -> tuple[list[str], dict[str, float | None]]:
    min_gap = float(state["cruise"]["min_gap_m"])
    speed = float(state["ego"]["speed_mph"])
    predicted: dict[str, float | None] = {}
    gap_safe: list[str] = []

    for action in ACTION_ORDER:
        if speed <= LOW_SPEED_BRAKE_MASK_MPH and action in ("hard_brake", "brake"):
            predicted[action] = None
            continue

        min_predicted = predict_min_gap(
            state,
            accel_mps2=ACTION_ACCEL_MPS2[action],
            horizon_seconds=ACTION_PREDICTION_HORIZON_SECONDS,
            physics_fps=physics_fps,
        )
        predicted[action] = None if min_predicted == float("inf") else min_predicted
        if min_predicted >= min_gap - 1e-9:
            gap_safe.append(action)

    if not gap_safe:
        gap_safe = ["hard_brake"]

    return apply_speed_envelope(state, gap_safe), predicted


def actor_question(
    state: dict[str, Any],
    safe_actions: list[str],
) -> dict[str, Any]:
    cruise = state["cruise"]
    target_speed = float(cruise["target_speed_mph"])
    target_gap = float(cruise["target_gap_m"])
    min_gap = float(cruise["min_gap_m"])
    deadband = float(cruise["speed_deadband_mph"])

    return {
        "type": "choice",
        "instructions": (
            "Choose the ADAPTIVE CRUISE CONTROL action most appropriate for the next one "
            "second. This is not free-form human driving. The runtime has already removed "
            "actions predicted to violate the hard following-distance constraint. Within the "
            f"remaining safe action space, prefer smooth control: keep speed near {target_speed:.1f} "
            f"mph, treat +/-{deadband:.1f} mph as a stable cruise deadband, and when following "
            f"a lead vehicle prefer a comfortable gap near {target_gap:.1f} m while never "
            f"depending on crossing the hard {min_gap:.1f} m boundary. Avoid unnecessary "
            "brake/accelerate oscillation. Return a calibrated probability distribution over "
            "every available action."
        ),
        "criteria": {key: ACTION_CRITERIA[key] for key in safe_actions},
    }


def prune_irrelevant_front_vehicle(state: dict[str, Any]) -> bool:
    front = state["road"].get("front_vehicle")
    if front is None:
        return False
    max_relevant = (
        float(state["cruise"]["free_gap_m"]) + MAX_RELEVANT_EXTRA_GAP_M
    )
    if float(front["distance_m"]) <= max_relevant:
        return False
    state["road"]["front_vehicle"] = None
    return True


def physics_frame(
    state: dict[str, Any],
    requested_action: str,
    *,
    dt: float,
) -> tuple[float, float | None, float | None, bool, float]:
    speed_before_mps = mph_to_mps(float(state["ego"]["speed_mph"]))
    requested_accel = ACTION_ACCEL_MPS2[requested_action]
    candidate_speed_mps = max(0.0, speed_before_mps + requested_accel * dt)

    front = state["road"].get("front_vehicle")
    override = False
    executed_speed_mps = candidate_speed_mps
    gap_after: float | None = None
    gap_margin: float | None = None

    if front is not None:
        min_gap = float(state["cruise"]["min_gap_m"])
        gap_before = float(front["distance_m"])
        lead_speed_mps = mph_to_mps(float(front["speed_mph"]))

        # Discrete control-barrier condition:
        # gap_next = gap + (v_lead - v_ego_next) * dt >= min_gap
        barrier_max_ego_mps = lead_speed_mps + max(0.0, gap_before - min_gap) / dt
        if executed_speed_mps > barrier_max_ego_mps:
            executed_speed_mps = max(0.0, barrier_max_ego_mps)
            override = True

        gap_after = gap_before + (lead_speed_mps - executed_speed_mps) * dt
        if gap_after < min_gap - 1e-7:
            raise AssertionError(
                f"safety invariant violated: gap={gap_after:.9f} < min={min_gap:.9f}"
            )
        if gap_after < min_gap:
            gap_after = min_gap
        front["distance_m"] = gap_after
        gap_margin = gap_after - min_gap

    executed_accel = (executed_speed_mps - speed_before_mps) / dt
    state["ego"]["speed_mph"] = mps_to_mph(executed_speed_mps)
    state["ego"]["longitudinal_acceleration_mps2"] = executed_accel

    relax_lead_vehicle(state, dt)
    if prune_irrelevant_front_vehicle(state):
        gap_after = None
        gap_margin = None

    return (
        float(state["ego"]["speed_mph"]),
        gap_after,
        gap_margin,
        override,
        executed_accel,
    )


def run_worldline(
    backend: RemoteBackend,
    worldline_id: int,
    *,
    duration_seconds: int,
    physics_fps: int,
    seed_base: int,
    target_speed_mph: float,
    min_gap_m: float,
) -> dict[str, Any]:
    event_rng = random.Random(seed_base + worldline_id * 2)
    actor_rng = random.Random(seed_base + worldline_id * 2 + 1)
    state = initial_state(
        target_speed_mph=target_speed_mph,
        min_gap_m=min_gap_m,
        physics_fps=physics_fps,
    )

    dt = 1.0 / physics_fps
    total_frames = duration_seconds * physics_fps

    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    event_count = 0
    first_event_second: int | None = None
    max_speed = float(state["ego"]["speed_mph"])
    min_gap = float("inf")
    invariant_violations = 0
    shield_override_frames = 0
    shield_override_decisions = 0
    free_speed_abs_errors: list[float] = []
    free_band_frames = 0
    free_frames = 0
    following_gap_abs_errors: list[float] = []

    for second in range(duration_seconds):
        if second > 0:
            age_history(state)

        state["time_step"] = second
        state["time_seconds"] = float(second)
        refresh_control_state(state, physics_fps)

        event_q = event_question(state, physics_fps)
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
        refresh_control_state(state, physics_fps)

        safe_actions, predicted_min_gaps = safe_action_space(state, physics_fps)
        actor_q = actor_question(state, safe_actions)
        actor_probs, actor_body, actor_ms = backend.decide_choice(
            deepcopy(state),
            "driver_action",
            actor_q,
        )
        in_tok, out_tok = usage_tokens(actor_body.get("usage"))
        input_tokens += in_tok
        output_tokens += out_tok

        sampled_action, actor_draw = sample_distribution(
            actor_probs,
            safe_actions,
            actor_rng,
        )

        frame_start = len(frames)
        speed_before = float(state["ego"]["speed_mph"])
        gap_before = (
            None
            if state["road"].get("front_vehicle") is None
            else float(state["road"]["front_vehicle"]["distance_m"])
        )
        decision_override_frames = 0
        decision_executed_accels: list[float] = []

        for subframe in range(physics_fps):
            frame_index = second * physics_fps + subframe
            (
                speed_after,
                gap_after,
                gap_margin,
                override,
                executed_accel,
            ) = physics_frame(
                state,
                sampled_action,
                dt=dt,
            )

            if override:
                shield_override_frames += 1
                decision_override_frames += 1

            if gap_after is not None:
                min_gap = min(min_gap, gap_after)
                if gap_after < min_gap_m - 1e-7:
                    invariant_violations += 1
                following_gap_abs_errors.append(
                    abs(gap_after - float(state["cruise"]["target_gap_m"]))
                )
            else:
                free_frames += 1
                speed_error = abs(speed_after - target_speed_mph)
                free_speed_abs_errors.append(speed_error)
                if speed_error <= float(state["cruise"]["speed_deadband_mph"]):
                    free_band_frames += 1

            max_speed = max(max_speed, speed_after)
            decision_executed_accels.append(executed_accel)

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
                    "gap_margin_m": (
                        None if gap_margin is None else round(gap_margin, 4)
                    ),
                    "safety_override": override,
                    "requested_action": sampled_action,
                    "executed_acceleration_mps2": round(executed_accel, 6),
                    "near_miss": False,
                    "critical_gap": False,
                    "collision_floor_hit": False,
                }
            )

        if decision_override_frames:
            shield_override_decisions += 1
            state["recent_history"]["safety_overrides"].append(
                {
                    "kind": "min_gap_barrier",
                    "frames": decision_override_frames,
                    "age_seconds": 0,
                }
            )

        mean_executed_accel = statistics.mean(decision_executed_accels)
        state["recent_history"]["actions"].append(
            {
                "mean_longitudinal_acceleration_mps2": mean_executed_accel,
                "age_seconds": 0,
            }
        )
        refresh_control_state(state, physics_fps)

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
                "actor_options": safe_actions,
                "actor_probabilities": actor_probs,
                "sampled_action": sampled_action,
                "actor_draw": actor_draw,
                "actor_argmax": max(safe_actions, key=lambda k: actor_probs[k]),
                "actor_elapsed_ms": actor_ms,
                "predicted_min_gap_by_action_m": predicted_min_gaps,
                "shield_override_frames": decision_override_frames,
                "mean_executed_acceleration_mps2": mean_executed_accel,
                "speed_before_mph": speed_before,
                "speed_after_mph": float(state["ego"]["speed_mph"]),
                "front_gap_before_m": gap_before,
                "front_gap_after_m": (
                    None
                    if state["road"].get("front_vehicle") is None
                    else float(state["road"]["front_vehicle"]["distance_m"])
                ),
            }
        )

    assert len(frames) == total_frames
    assert invariant_violations == 0

    event_counts = Counter(r["sampled_event"] for r in decisions)
    action_counts = Counter(r["sampled_action"] for r in decisions)

    return {
        "worldline_id": worldline_id,
        "event_seed": seed_base + worldline_id * 2,
        "actor_seed": seed_base + worldline_id * 2 + 1,
        "reported_model": backend.reported_model,
        "runtime_variant": RUNTIME_VARIANT,
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
            "near_miss": False,
            "critical_gap": False,
            "collision_floor_hit": False,
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
            "safe_cruise": {
                "target_speed_mph": target_speed_mph,
                "min_gap_m": min_gap_m,
                "target_gap_m": float(state["cruise"]["target_gap_m"]),
                "speed_deadband_mph": float(state["cruise"]["speed_deadband_mph"]),
                "invariant_violation_frames": invariant_violations,
                "shield_override_frames": shield_override_frames,
                "shield_override_decisions": shield_override_decisions,
                "free_frames": free_frames,
                "free_cruise_band_rate": (
                    free_band_frames / free_frames if free_frames else None
                ),
                "free_speed_mae_mph": (
                    statistics.mean(free_speed_abs_errors)
                    if free_speed_abs_errors
                    else None
                ),
                "following_gap_mae_m": (
                    statistics.mean(following_gap_abs_errors)
                    if following_gap_abs_errors
                    else None
                ),
                "min_gap_margin_m": (
                    None if min_gap == float("inf") else min_gap - min_gap_m
                ),
            },
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
    event_samples = Counter()
    event_eligible = Counter()
    event_predicted_mass = Counter()
    actor_samples = Counter()
    actor_eligible = Counter()
    actor_predicted_mass = Counter()
    event_latency: list[float] = []
    actor_latency: list[float] = []

    for world in completed:
        for row in world["decisions"]:
            event_samples[row["sampled_event"]] += 1
            actor_samples[row["sampled_action"]] += 1
            event_latency.append(float(row["event_elapsed_ms"]))
            actor_latency.append(float(row["actor_elapsed_ms"]))
            for key in row["event_options"]:
                event_eligible[key] += 1
                event_predicted_mass[key] += float(row["event_probabilities"][key])
            for key in row["actor_options"]:
                actor_eligible[key] += 1
                actor_predicted_mass[key] += float(row["actor_probabilities"][key])

    event_calibration = {}
    for event, eligible in event_eligible.items():
        if eligible:
            empirical = event_samples[event] / eligible
            predicted = event_predicted_mass[event] / eligible
            event_calibration[event] = {
                "eligible_calls": eligible,
                "empirical_frequency_when_eligible": empirical,
                "mean_predicted_probability": predicted,
                "difference": empirical - predicted,
            }

    actor_calibration = {}
    for action, eligible in actor_eligible.items():
        if eligible:
            empirical = actor_samples[action] / eligible
            predicted = actor_predicted_mass[action] / eligible
            actor_calibration[action] = {
                "eligible_calls": eligible,
                "empirical_frequency_when_eligible": empirical,
                "mean_predicted_probability": predicted,
                "difference": empirical - predicted,
            }

    summaries = [w["summary"] for w in completed]
    safe = [s["safe_cruise"] for s in summaries]
    mins = [
        float(s["min_front_gap_m"])
        for s in summaries
        if s["min_front_gap_m"] is not None
    ]
    finals = [float(s["final_speed_mph"]) for s in summaries]
    maxes = [float(s["max_speed_mph"]) for s in summaries]
    events = [int(s["events"]) for s in summaries]

    def mean_optional(key: str) -> float | None:
        values = [float(x[key]) for x in safe if x.get(key) is not None]
        return statistics.mean(values) if values else None

    return {
        "backend": backend,
        "requested_model": requested_model,
        "reported_model": completed[0].get("reported_model"),
        "runtime_variant": RUNTIME_VARIANT,
        "worldlines_requested": len(worldlines),
        "worldlines_completed": n,
        "worldline_errors": len(worldlines) - n,
        "duration_seconds": duration_seconds,
        "physics_fps": physics_fps,
        "frames_per_worldline": duration_seconds * physics_fps,
        "decision_hz": 1,
        "decisions_per_worldline": duration_seconds,
        "system_one_calls": n * duration_seconds * 2,
        "usage": {
            "input_tokens": sum(int(s["input_tokens"]) for s in summaries),
            "output_tokens": sum(int(s["output_tokens"]) for s in summaries),
        },
        "event_process": {
            "eventful_worldline_rate": sum(bool(s["eventful"]) for s in summaries) / n,
            "mean_events_per_worldline": statistics.mean(events),
            "counts": dict(event_samples),
            "sampling_self_consistency": event_calibration,
            "max_abs_frequency_minus_probability": max(
                (abs(v["difference"]) for v in event_calibration.values()),
                default=0.0,
            ),
        },
        "actor_process": {
            "counts": dict(actor_samples),
            "sampling_self_consistency": actor_calibration,
            "max_abs_frequency_minus_probability": max(
                (abs(v["difference"]) for v in actor_calibration.values()),
                default=0.0,
            ),
        },
        "outcomes": {
            "near_miss_rate": 0.0,
            "critical_gap_rate": 0.0,
            "collision_floor_rate": 0.0,
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
        "safe_cruise": {
            "target_speed_mph": safe[0]["target_speed_mph"],
            "min_gap_m": safe[0]["min_gap_m"],
            "target_gap_m": safe[0]["target_gap_m"],
            "speed_deadband_mph": safe[0]["speed_deadband_mph"],
            "invariant_violation_frames": sum(
                int(x["invariant_violation_frames"]) for x in safe
            ),
            "shield_override_frames": sum(
                int(x["shield_override_frames"]) for x in safe
            ),
            "shield_override_decisions": sum(
                int(x["shield_override_decisions"]) for x in safe
            ),
            "free_cruise_band_rate": mean_optional("free_cruise_band_rate"),
            "free_speed_mae_mph": mean_optional("free_speed_mae_mph"),
            "following_gap_mae_m": mean_optional("following_gap_mae_m"),
            "min_gap_margin_m": mean_optional("min_gap_margin_m"),
        },
        "latency_ms": {
            "event_mean": statistics.mean(event_latency),
            "event_p95": percentile(event_latency, 0.95),
            "actor_mean": statistics.mean(actor_latency),
            "actor_p95": percentile(actor_latency, 0.95),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "dashscope"], required=True)
    ap.add_argument("--worldlines", type=int, default=1)
    ap.add_argument("--duration-seconds", type=int, default=120)
    ap.add_argument("--physics-fps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target-speed-mph", type=float, default=80.0)
    ap.add_argument("--min-gap-m", type=float, default=30.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if min(args.worldlines, args.duration_seconds, args.physics_fps, args.workers) <= 0:
        raise SystemExit("worldlines, duration-seconds, physics-fps and workers must be positive")
    if args.target_speed_mph <= 0:
        raise SystemExit("target-speed-mph must be positive")
    if args.min_gap_m <= 0:
        raise SystemExit("min-gap-m must be positive")

    local = threading.local()

    def get_backend() -> RemoteBackend:
        if not hasattr(local, "backend"):
            local.backend = RemoteBackend(args.backend)
        return local.backend

    def run_one(worldline_id: int) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(3):
            try:
                return run_worldline(
                    get_backend(),
                    worldline_id,
                    duration_seconds=args.duration_seconds,
                    physics_fps=args.physics_fps,
                    seed_base=args.seed_base,
                    target_speed_mph=args.target_speed_mph,
                    min_gap_m=args.min_gap_m,
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
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
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
    if summary["safe_cruise"]["invariant_violation_frames"] != 0:
        raise SystemExit("safe-cruise minimum-gap invariant was violated")


if __name__ == "__main__":
    main()
