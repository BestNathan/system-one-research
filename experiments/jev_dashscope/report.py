import argparse
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def pct(v):
    return "-" if v is None else f"{100*v:.1f}%"


def num(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def ms(v):
    return "-" if v is None else f"{v:.0f}"


def exact_two_sided(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-summary", required=True)
    ap.add_argument("--dashscope-summary", required=True)
    ap.add_argument("--jev-raw", required=True)
    ap.add_argument("--dashscope-raw", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    jev = load_json(args.jev_summary)
    dash = load_json(args.dashscope_summary)
    jr = {r["id"]: r for r in load_jsonl(args.jev_raw)}
    dr = {r["id"]: r for r in load_jsonl(args.dashscope_raw)}
    ids = sorted(set(jr) & set(dr))

    both_correct = jev_only = dash_only = both_wrong = 0
    disagreements = []
    for case_id in ids:
        j, d = jr[case_id], dr[case_id]
        if j["correct"] and d["correct"]:
            both_correct += 1
        elif j["correct"] and not d["correct"]:
            jev_only += 1
        elif not j["correct"] and d["correct"]:
            dash_only += 1
        else:
            both_wrong += 1
        if j["predicted"] != d["predicted"]:
            disagreements.append({
                "id": case_id,
                "state": j["state"],
                "expected": j["expected"],
                "jev": j["predicted"],
                "jev_conf": j["confidence"],
                "dash": d["predicted"],
                "dash_conf": d["confidence"],
            })

    p = exact_two_sided(jev_only, dash_only)

    lines = [
        "# Jev vs DashScope decision-model-preview — BANKING77",
        "",
        "## Scope",
        "",
        "A compact paired benchmark using exactly 77 BANKING77 test cases: one example for each intent. Every request exposes the full 77-way choice space.",
        "",
        "Both providers receive the exact same state, question instructions, option keys, and option descriptions.",
        "",
        "## Main result",
        "",
        "| Metric | TypeSafe Jev | DashScope decision-model-preview |",
        "|---|---:|---:|",
        f"| Cases OK | {jev['cases_ok']}/{jev['cases']} | {dash['cases_ok']}/{dash['cases']} |",
        f"| Accuracy | {pct(jev.get('accuracy'))} | {pct(dash.get('accuracy'))} |",
        f"| Brier ↓ | {num(jev.get('brier'))} | {num(dash.get('brier'))} |",
        f"| ECE ↓ | {num(jev.get('ece'))} | {num(dash.get('ece'))} |",
        f"| Mean confidence | {pct(jev.get('mean_confidence'))} | {pct(dash.get('mean_confidence'))} |",
        f"| Confidence when correct | {pct(jev.get('correct_mean_confidence'))} | {pct(dash.get('correct_mean_confidence'))} |",
        f"| Confidence when wrong | {pct(jev.get('wrong_mean_confidence'))} | {pct(dash.get('wrong_mean_confidence'))} |",
        f"| Mean latency | {ms(jev.get('latency_ms',{}).get('mean'))} ms | {ms(dash.get('latency_ms',{}).get('mean'))} ms |",
        f"| p50 latency | {ms(jev.get('latency_ms',{}).get('p50'))} ms | {ms(dash.get('latency_ms',{}).get('p50'))} ms |",
        f"| p95 latency | {ms(jev.get('latency_ms',{}).get('p95'))} ms | {ms(dash.get('latency_ms',{}).get('p95'))} ms |",
        "",
        "Latency is end-to-end from the GitHub runner to each hosted service; region/network path is part of the measurement, so it is not a pure model-compute comparison.",
        "",
        "## Paired correctness",
        "",
        f"- both correct: **{both_correct}/{len(ids)}**",
        f"- Jev correct / DashScope wrong: **{jev_only}**",
        f"- DashScope correct / Jev wrong: **{dash_only}**",
        f"- both wrong: **{both_wrong}**",
        f"- exact two-sided discordant-pair p: **{p:.4g}**",
        f"- prediction disagreements: **{len(disagreements)}/{len(ids)} ({100*len(disagreements)/len(ids):.1f}%)**",
        "",
        "## Disagreements",
        "",
        "| Case | Gold | Jev | DashScope |",
        "|---|---|---|---|",
    ]
    for x in disagreements:
        lines.append(
            f"| {x['id']} | {x['expected']} | {x['jev']} ({x['jev_conf']:.2f}) | "
            f"{x['dash']} ({x['dash_conf']:.2f}) |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This benchmark intentionally optimizes for low evaluation cost, not statistical power.",
        "- The 77 cases cover every BANKING77 intent exactly once, but one example per intent is too small for strong claims about small accuracy differences.",
        "- Because both systems are tested on identical cases, large paired differences or systematic disagreement patterns are still informative.",
        "- BANKING77 is English and uses a very large 77-way action space; it does not measure Chinese decision quality or small local action-space behavior.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
