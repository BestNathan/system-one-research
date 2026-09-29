from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EVENT_ORDER = [
    "no_event",
    "vehicle_cut_in",
    "lead_vehicle_hard_brake",
    "lead_vehicle_accelerate",
    "lead_vehicle_exit",
]


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


def paired_boolean(jw, dw, key):
    ids = sorted(set(jw) & set(dw))
    both = j_only = d_only = neither = 0
    for worldline_id in ids:
        j = bool(jw[worldline_id]["summary"][key])
        d = bool(dw[worldline_id]["summary"][key])
        if j and d:
            both += 1
        elif j:
            j_only += 1
        elif d:
            d_only += 1
        else:
            neither += 1
    return {
        "n": len(ids),
        "both": both,
        "jev_only": j_only,
        "dashscope_only": d_only,
        "neither": neither,
    }


def paired_mean(jw, dw, key):
    ids = sorted(set(jw) & set(dw))
    if not ids:
        return None
    return statistics.mean(
        float(jw[i]["summary"][key]) - float(dw[i]["summary"][key])
        for i in ids
    )


def event_count(summary, event):
    return int(summary["event_process"]["counts"].get(event, 0))


def response(summary, event, field):
    return summary["event_response"].get(event, {}).get(field)


def risk(summary, event):
    return summary["event_risk"].get(event, {}).get(
        "near_miss_within_2_steps"
    )


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
        x["worldline_id"]: x
        for x in load_jsonl(args.jev_worldlines)
        if "error" not in x
    }
    dw = {
        x["worldline_id"]: x
        for x in load_jsonl(args.dashscope_worldlines)
        if "error" not in x
    }

    eventful = paired_boolean(jw, dw, "eventful")
    near = paired_boolean(jw, dw, "near_miss")
    critical = paired_boolean(jw, dw, "critical_gap")
    floor = paired_boolean(jw, dw, "collision_floor_hit")

    same_event_by_step = []
    same_action_by_step = []
    paired_speed_mae = []
    ids = sorted(set(jw) & set(dw))
    for step in range(j["steps_per_worldline"]):
        same_event = 0
        same_action = 0
        speed_diff = []
        for worldline_id in ids:
            jr = jw[worldline_id]["steps"][step]
            dr = dw[worldline_id]["steps"][step]
            same_event += jr["sampled_event"] == dr["sampled_event"]
            same_action += jr["sampled_action"] == dr["sampled_action"]
            speed_diff.append(
                abs(float(jr["speed_after_mph"]) - float(dr["speed_after_mph"]))
            )
        same_event_by_step.append(same_event / len(ids))
        same_action_by_step.append(same_action / len(ids))
        paired_speed_mae.append(statistics.mean(speed_diff))

    lines = [
        "# Learned-event probabilistic worlds",
        "",
        "## Research question",
        "",
        "Can the fixed exogenous event schedule be replaced by a second System One probability head, so that both actor actions and world events are state-conditioned probability distributions?",
        "",
        "Each provider runs a fully separate world. At every one-second step:",
        "",
        "~~~text",
        "state_t",
        "  -> event System One -> P(exogenous event | state_t)",
        "  -> sample event and mutate world",
        "  -> actor System One -> P(driver action | updated state_t)",
        "  -> sample driver action",
        "  -> physics",
        "  -> state_t+1",
        "~~~",
        "",
        "There is no hard-coded cut-in time in this experiment.",
        "",
        "Recent event memory is explicitly aged and removed after five steps; the model itself remains stateless.",
        "",
        "## Main result",
        "",
        "| Metric | Jev world | DashScope world |",
        "|---|---:|---:|",
        f"| Worldlines completed | {j['worldlines_completed']} | {d['worldlines_completed']} |",
        f"| System One calls | {j['system_one_calls']:,} | {d['system_one_calls']:,} |",
        f"| Eventful worldlines | {pct(j['event_process']['eventful_worldline_rate'])} | {pct(d['event_process']['eventful_worldline_rate'])} |",
        f"| Mean exogenous events/worldline | {num(j['event_process']['mean_events_per_worldline'])} | {num(d['event_process']['mean_events_per_worldline'])} |",
        f"| Near-miss rate | {pct(j['outcomes']['near_miss_rate'])} | {pct(d['outcomes']['near_miss_rate'])} |",
        f"| Critical-gap rate | {pct(j['outcomes']['critical_gap_rate'])} | {pct(d['outcomes']['critical_gap_rate'])} |",
        f"| 1m simulator-floor rate | {pct(j['outcomes']['collision_floor_rate'])} | {pct(d['outcomes']['collision_floor_rate'])} |",
        f"| Mean minimum gap | {num(j['outcomes']['min_gap_m']['mean'])} m | {num(d['outcomes']['min_gap_m']['mean'])} m |",
        f"| Mean final speed | {num(j['outcomes']['final_speed_mph']['mean'])} mph | {num(d['outcomes']['final_speed_mph']['mean'])} mph |",
        f"| Event sampler max abs gap | {pct(j['event_process']['max_abs_frequency_minus_probability'])} | {pct(d['event_process']['max_abs_frequency_minus_probability'])} |",
        f"| Actor sampler max abs gap | {pct(j['actor_process']['max_abs_frequency_minus_probability'])} | {pct(d['actor_process']['max_abs_frequency_minus_probability'])} |",
        f"| Input tokens | {j['usage']['input_tokens']:,} | {d['usage']['input_tokens']:,} |",
        f"| Output tokens | {j['usage']['output_tokens']:,} | {d['usage']['output_tokens']:,} |",
        "",
        "## Learned event distribution",
        "",
        "| Event | Jev sampled | DashScope sampled | Jev count | DashScope count |",
        "|---|---:|---:|---:|---:|",
    ]

    for event in EVENT_ORDER:
        jc = j["event_process"]["sampling_self_consistency"].get(event)
        dc = d["event_process"]["sampling_self_consistency"].get(event)
        lines.append(
            f"| {event} | "
            f"{pct(None if jc is None else jc['empirical_frequency_when_eligible'])} | "
            f"{pct(None if dc is None else dc['empirical_frequency_when_eligible'])} | "
            f"{event_count(j, event)} | {event_count(d, event)} |"
        )

    lines += [
        "",
        "Event frequencies above use eligible calls as their denominator because the action space is state-dependent: a cut-in is offered only when no lead vehicle exists, while lead-vehicle actions are offered only when a lead vehicle exists.",
        "",
        "## Actor reaction to sampled external events",
        "",
        "| Event | Jev same-step brake | DashScope same-step brake | Jev same-step accel | DashScope same-step accel | Jev near miss <=2 steps | DashScope near miss <=2 steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for event in EVENT_ORDER[1:]:
        lines.append(
            f"| {event} | "
            f"{pct(response(j, event, 'brake_rate_same_step'))} | "
            f"{pct(response(d, event, 'brake_rate_same_step'))} | "
            f"{pct(response(j, event, 'acceleration_rate_same_step'))} | "
            f"{pct(response(d, event, 'acceleration_rate_same_step'))} | "
            f"{pct(risk(j, event))} | {pct(risk(d, event))} |"
        )

    lines += [
        "",
        "## Paired world outcomes",
        "",
        "The same worldline ID uses matched actor/event random-number streams across providers, while every state transition remains provider-specific.",
        "",
        "| Outcome | Both | Jev only | DashScope only | Neither |",
        "|---|---:|---:|---:|---:|",
        f"| Any learned event | {eventful['both']} | {eventful['jev_only']} | {eventful['dashscope_only']} | {eventful['neither']} |",
        f"| Near miss | {near['both']} | {near['jev_only']} | {near['dashscope_only']} | {near['neither']} |",
        f"| Critical gap | {critical['both']} | {critical['jev_only']} | {critical['dashscope_only']} | {critical['neither']} |",
        f"| 1m floor | {floor['both']} | {floor['jev_only']} | {floor['dashscope_only']} | {floor['neither']} |",
        "",
        f"Paired mean Jev-minus-DashScope final-speed difference: **{num(paired_mean(jw, dw, 'final_speed_mph'))} mph**.",
        "",
        "## World divergence over time",
        "",
        "| Step | Same sampled event | Same sampled actor action | Mean paired speed difference |",
        "|---:|---:|---:|---:|",
    ]

    for step in range(j["steps_per_worldline"]):
        lines.append(
            f"| {step} | {pct(same_event_by_step[step])} | "
            f"{pct(same_action_by_step[step])} | {num(paired_speed_mae[step])} mph |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This experiment removes the strongest hand-authored stochastic element from the previous simulator. The timing and type of abrupt traffic events are now sampled from a learned conditional distribution and feed back into the next state.",
        "",
        "The event model is still not grounded to real traffic statistics. Its probabilities are zero-shot semantic judgments produced from the supplied state and profile. Therefore this experiment measures whether a learned probability field can drive a coherent stochastic world, not whether the generated traffic-event rates are realistic.",
        "",
        "The useful systems abstraction is now two coupled stochastic policies:",
        "",
        "~~~text",
        "P(event_t | state_t, traffic_profile, recent_history)",
        "P(action_t | state_t after event_t, driver_profile, recent_history)",
        "~~~",
        "",
        "Sampling both distributions, applying physics, and feeding the result back creates an endogenous probability world. Differences between provider probability fields can compound through both the event process and the actor process.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
