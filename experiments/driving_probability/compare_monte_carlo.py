from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def num(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def paired_outcome(a, b, key):
    ids = sorted(set(a) & set(b))
    both = a_only = b_only = neither = 0
    for worldline_id in ids:
        av = bool(a[worldline_id]["summary"][key])
        bv = bool(b[worldline_id]["summary"][key])
        if av and bv:
            both += 1
        elif av:
            a_only += 1
        elif bv:
            b_only += 1
        else:
            neither += 1
    return {
        "n": len(ids),
        "both": both,
        "jev_only": a_only,
        "dashscope_only": b_only,
        "neither": neither,
    }


def paired_mean_difference(a, b, key):
    ids = sorted(set(a) & set(b))
    diffs = [
        float(a[i]["summary"][key]) - float(b[i]["summary"][key])
        for i in ids
    ]
    return statistics.mean(diffs) if diffs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-summary", required=True)
    ap.add_argument("--dashscope-summary", required=True)
    ap.add_argument("--jev-worldlines", required=True)
    ap.add_argument("--dashscope-worldlines", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    j = load_json(args.jev_summary)
    d = load_json(args.dashscope_summary)
    jw = {
        w["worldline_id"]: w
        for w in load_jsonl(args.jev_worldlines)
        if "error" not in w
    }
    dw = {
        w["worldline_id"]: w
        for w in load_jsonl(args.dashscope_worldlines)
        if "error" not in w
    }

    near = paired_outcome(jw, dw, "near_miss")
    critical = paired_outcome(jw, dw, "critical_gap")
    floor = paired_outcome(jw, dw, "collision_floor_hit")

    lines = [
        "# Monte Carlo System One driving worldlines",
        "",
        "## Design",
        "",
        f"Each provider runs **{j['worldlines_completed']} independent closed-loop worldlines** of {j['steps_per_worldline']} steps.",
        "",
        "The providers do not share state. A Jev trajectory evolves only from Jev probabilities and sampled Jev actions; a DashScope trajectory evolves only from DashScope probabilities and sampled DashScope actions.",
        "",
        "Worldline IDs use the same deterministic PRNG seeds across providers. This is a matched-random-number design: the random draws are paired, but the state trajectories are model-specific and diverge immediately when the returned distributions differ.",
        "",
        "Every worldline receives the same exogenous event schedule: a slower vehicle cuts in at step 4 and the lane clears at step 8.",
        "",
        "## Aggregate outcomes",
        "",
        "| Metric | Jev | DashScope |",
        "|---|---:|---:|",
        f"| Worldlines completed | {j['worldlines_completed']} | {d['worldlines_completed']} |",
        f"| System One calls | {j['calls']} | {d['calls']} |",
        f"| Near-miss rate | {pct(j['outcomes']['near_miss_rate'])} | {pct(d['outcomes']['near_miss_rate'])} |",
        f"| Critical-gap rate (<3m) | {pct(j['outcomes']['critical_gap_rate'])} | {pct(d['outcomes']['critical_gap_rate'])} |",
        f"| 1m simulator-floor rate | {pct(j['outcomes']['collision_floor_rate'])} | {pct(d['outcomes']['collision_floor_rate'])} |",
        f"| Mean minimum gap | {num(j['outcomes']['min_gap_m']['mean'])} m | {num(d['outcomes']['min_gap_m']['mean'])} m |",
        f"| Median minimum gap | {num(j['outcomes']['min_gap_m']['p50'])} m | {num(d['outcomes']['min_gap_m']['p50'])} m |",
        f"| Mean final speed | {num(j['outcomes']['final_speed_mph']['mean'])} mph | {num(d['outcomes']['final_speed_mph']['mean'])} mph |",
        f"| Mean max speed | {num(j['outcomes']['max_speed_mph']['mean'])} mph | {num(d['outcomes']['max_speed_mph']['mean'])} mph |",
        f"| Mean hard brakes / worldline | {num(j['outcomes']['mean_hard_brakes_per_worldline'])} | {num(d['outcomes']['mean_hard_brakes_per_worldline'])} |",
        f"| Mean hard accelerations / worldline | {num(j['outcomes']['mean_hard_accelerates_per_worldline'])} | {num(d['outcomes']['mean_hard_accelerates_per_worldline'])} |",
        f"| Mean non-argmax samples / worldline | {num(j['outcomes']['mean_non_argmax_samples_per_worldline'])} | {num(d['outcomes']['mean_non_argmax_samples_per_worldline'])} |",
        f"| Sampling max absolute calibration gap | {pct(j['sampling_calibration']['max_abs_frequency_minus_probability'])} | {pct(d['sampling_calibration']['max_abs_frequency_minus_probability'])} |",
        f"| Input tokens | {j['usage']['input_tokens']:,} | {d['usage']['input_tokens']:,} |",
        f"| Output tokens | {j['usage']['output_tokens']:,} | {d['usage']['output_tokens']:,} |",
        f"| Mean latency | {num(j['latency_ms']['mean'], 1)} ms | {num(d['latency_ms']['mean'], 1)} ms |",
        "",
        "## Paired worldline outcomes",
        "",
        "Because the same PRNG seed is assigned to the same worldline ID, paired outcomes show how much the model-specific probability field changes the resulting world even under matched random draws.",
        "",
        "| Outcome | Both | Jev only | DashScope only | Neither |",
        "|---|---:|---:|---:|---:|",
        f"| Near miss | {near['both']} | {near['jev_only']} | {near['dashscope_only']} | {near['neither']} |",
        f"| Critical gap | {critical['both']} | {critical['jev_only']} | {critical['dashscope_only']} | {critical['neither']} |",
        f"| 1m floor | {floor['both']} | {floor['jev_only']} | {floor['dashscope_only']} | {floor['neither']} |",
        "",
        "Paired mean Jev-minus-DashScope differences:",
        "",
        f"- final speed: **{num(paired_mean_difference(jw, dw, 'final_speed_mph'))} mph**",
        f"- max speed: **{num(paired_mean_difference(jw, dw, 'max_speed_mph'))} mph**",
        "",
        "## Sampling self-consistency",
        "",
        "The action sampler draws directly from each model's returned distribution. Therefore empirical action frequency should converge to the average reported probability as the number of worldlines grows. This is a test of the simulation/sampling pipeline, not external behavioral calibration.",
        "",
        "| Action | Jev predicted | Jev sampled | DashScope predicted | DashScope sampled |",
        "|---|---:|---:|---:|---:|",
    ]

    for action in j["sampling_calibration"]["overall"]:
        jc = j["sampling_calibration"]["overall"][action]
        dc = d["sampling_calibration"]["overall"][action]
        lines.append(
            f"| {action} | {pct(jc['mean_predicted_probability'])} | {pct(jc['empirical_frequency'])} | "
            f"{pct(dc['mean_predicted_probability'])} | {pct(dc['empirical_frequency'])} |"
        )

    lines += [
        "",
        "## Event-window comparison",
        "",
        "| Step | Event | Jev mean gap | DashScope mean gap | Jev brake sampled | DashScope brake sampled | Jev accel sampled | DashScope accel sampled |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for step in [3, 4, 5, 6, 7, 8, 9]:
        js = j["by_step"][step]
        ds = d["by_step"][step]
        jbr = (
            js["action_frequency"]["hard_brake"]
            + js["action_frequency"]["brake"]
        )
        dbr = (
            ds["action_frequency"]["hard_brake"]
            + ds["action_frequency"]["brake"]
        )
        jac = (
            js["action_frequency"]["accelerate"]
            + js["action_frequency"]["hard_accelerate"]
        )
        dac = (
            ds["action_frequency"]["accelerate"]
            + ds["action_frequency"]["hard_accelerate"]
        )
        event = js.get("event") or "-"
        lines.append(
            f"| {step} | {event} | {num(js['mean_front_gap_after_m'])} | {num(ds['mean_front_gap_after_m'])} | "
            f"{pct(jbr)} | {pct(dbr)} | {pct(jac)} | {pct(dac)} |"
        )

    lines += [
        "",
        "## Interpretation constraint",
        "",
        "These worldlines compare the emergent behavior of two generic System One probability fields inside the same toy dynamics. They do not measure real-world driving correctness. A lower near-miss rate in this synthetic world is an environment-specific behavioral outcome, not a general safety ranking of the models.",
        "",
        "The experiment's key object is the distribution of trajectories: repeated sampling turns small per-step probability differences into measurable differences in world-state distributions over time.",
    ]

    Path(args.output).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
