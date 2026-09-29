from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(v, digits=4):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def pct(v):
    if v is None:
        return "-"
    return f"{100*v:.1f}%"


def row_for_label(data, experiment, label):
    for r in data["experiments"][experiment]:
        if r["label"] == label:
            return r
    raise KeyError(label)


def distribution_cells(row):
    p = row["probabilities"]
    return [
        pct(p["hard_brake"]),
        pct(p["brake"]),
        pct(p["keep_speed"]),
        pct(p["accelerate"]),
        pct(p["hard_accelerate"]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", required=True)
    ap.add_argument("--dashscope", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    providers = [("Jev", load(args.jev)), ("DashScope", load(args.dashscope))]
    lines = [
        "# Dynamic probability distribution experiment",
        "",
        "## Research question",
        "",
        "When the environment state changes, does a System One decision model move the entire action probability distribution in a structured way, rather than merely changing an argmax label?",
        "",
        "The experiment uses a deliberately small 1D longitudinal-driving world. It is not a driving-safety benchmark and it does not claim the models are production vehicle controllers. The purpose is to probe probability dynamics under controlled state interventions.",
        "",
        "Action space: hard_brake, brake, keep_speed, accelerate, hard_accelerate.",
        "",
        "Four probes are used:",
        "",
        "1. repeat the exact same state five times to estimate provider/model output noise;",
        "2. change only front-vehicle distance and measure distribution movement;",
        "3. change only driver aggressiveness and measure distribution movement;",
        "4. run a seeded 16-step closed loop with a sudden cut-in and later lane clearing, sampling actions from the returned probability distribution.",
        "",
        "A same-physical-state history counterfactual additionally checks whether recent near-miss history changes the distribution.",
        "",
        "## Summary metrics",
        "",
        "| Metric | Jev | DashScope |",
        "|---|---:|---:|",
    ]
    keys = [
        ("Repeat identical-state pairwise mean JS", "repeatability_pairwise_mean_js"),
        ("Front-gap adjacent mean JS", "front_gap_adjacent_mean_js"),
        ("Front-gap endpoint JS", "front_gap_endpoint_js"),
        ("Gap signal / repeat noise", "front_gap_signal_to_repeat_noise"),
        ("Gap vs expected acceleration Spearman", "front_gap_vs_expected_acceleration_spearman"),
        ("Aggressiveness vs expected acceleration Spearman", "aggressiveness_vs_expected_acceleration_spearman"),
        ("Near-miss history JS vs neutral", "history_near_miss_js_vs_neutral"),
        ("Cut-in JS", "cut_in_js"),
        ("Cut-in delta braking probability", "cut_in_delta_braking_probability"),
        ("Lane-clear delta acceleration probability", "clear_again_delta_acceleration_probability"),
        ("Closed-loop non-argmax sampled actions", "closed_loop_non_argmax_samples"),
        ("Closed-loop minimum sampled-action probability", "closed_loop_min_sampled_probability"),
    ]
    j = providers[0][1]["metrics"]
    d = providers[1][1]["metrics"]
    for title, key in keys:
        if "probability" in title.lower():
            a, b = pct(j.get(key)), pct(d.get(key))
        elif key == "closed_loop_non_argmax_samples":
            a, b = str(j.get(key)), str(d.get(key))
        else:
            a, b = f(j.get(key)), f(d.get(key))
        lines.append(f"| {title} | {a} | {b} |")

    lines += [
        "",
        "JS divergence is measured in natural-log nats and is bounded by ln(2). Repeat-state JS is the empirical noise floor for this experiment.",
        "",
        "## Controlled front-gap sweep",
        "",
    ]
    for name, data in providers:
        lines += [
            f"### {name}",
            "",
            "| Gap | hard brake | brake | keep | accelerate | hard accelerate | E[a] m/s2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in data["experiments"]["front_gap_sweep"]:
            fv = r["state"]["road"]["front_vehicle"]
            gap = "clear" if fv is None else f"{fv['distance_m']:.0f} m"
            cells = distribution_cells(r)
            lines.append(
                f"| {gap} | {' | '.join(cells)} | {r['expected_acceleration_mps2']:.3f} |"
            )
        lines.append("")

    lines += ["## Aggressiveness intervention", ""]
    for name, data in providers:
        lines += [
            f"### {name}",
            "",
            "| Aggressiveness | P(accelerate + hard accelerate) | Expected acceleration m/s2 | Argmax |",
            "|---:|---:|---:|---|",
        ]
        for r in data["experiments"]["aggressiveness_sweep"]:
            value = r["state"]["driver_profile"]["aggressiveness"]
            lines.append(
                f"| {value:.2f} | {pct(r['p_accelerate'])} | {r['expected_acceleration_mps2']:.3f} | {r['argmax']} |"
            )
        lines.append("")

    lines += ["## Same physical state, different recent history", ""]
    for name, data in providers:
        neutral = row_for_label(data, "history_counterfactual", "history-neutral")
        near = row_for_label(data, "history_counterfactual", "history-near-miss")
        accel = row_for_label(data, "history_counterfactual", "history-recent-hard-accel")
        lines += [
            f"### {name}",
            "",
            "| History | P(brake) | P(accelerate) | Expected acceleration | Argmax |",
            "|---|---:|---:|---:|---|",
            f"| neutral | {pct(neutral['p_brake'])} | {pct(neutral['p_accelerate'])} | {neutral['expected_acceleration_mps2']:.3f} | {neutral['argmax']} |",
            f"| near miss 3s ago | {pct(near['p_brake'])} | {pct(near['p_accelerate'])} | {near['expected_acceleration_mps2']:.3f} | {near['argmax']} |",
            f"| just hard-accelerated | {pct(accel['p_brake'])} | {pct(accel['p_accelerate'])} | {accel['expected_acceleration_mps2']:.3f} | {accel['argmax']} |",
            "",
        ]

    lines += ["## Closed-loop event trace", ""]
    for name, data in providers:
        lines += [
            f"### {name}",
            "",
            "| t | speed before->after mph | front gap after | P(brake) | P(accel) | argmax | sampled action | sampled p | JS from prev |",
            "|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
        for r in data["experiments"]["closed_loop"]:
            gap = "-" if r["front_gap_after_m"] is None else f"{r['front_gap_after_m']:.1f}m"
            js = "-" if r["js_from_previous"] is None else f"{r['js_from_previous']:.4f}"
            lines.append(
                f"| {r['step']} | {r['speed_before_mph']:.1f}->{r['speed_after_mph']:.1f} | {gap} | "
                f"{pct(r['p_brake'])} | {pct(r['p_accelerate'])} | {r['argmax']} | {r['sampled_action']} | "
                f"{pct(r['sampled_action_probability'])} | {js} |"
            )
        lines.append("")

    lines += [
        "## Interpretation guide",
        "",
        "The strongest evidence for genuine state-conditioned probability movement is a front-gap intervention JS that is materially larger than the identical-state repeat JS noise floor, combined with directionally sensible movement in braking and acceleration mass.",
        "",
        "The aggressiveness sweep tests a different dimension: whether a stable latent-style variable changes the distribution even when the road is clear. A positive rank correlation is evidence that the model is using that state variable rather than treating it as inert prompt text.",
        "",
        "The closed loop demonstrates a separate point: sampling from the full distribution can choose a non-argmax action. That is expected probabilistic behavior, not a model inconsistency. The random source in this experiment is a fixed-seed PRNG so the trace is reproducible; System One supplies the conditional distribution, not physical true randomness.",
        "",
        "The near-miss counterfactual should be read as a memory/state test. The physical state is held fixed; only recent history changes. Any distribution shift therefore comes from the model conditioning on the supplied history.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
