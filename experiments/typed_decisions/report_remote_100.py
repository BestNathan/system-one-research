from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def key(row):
    return (row["case_id"], row["question_id"])


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


def paired(a_rows, b_rows):
    a = {key(r): r for r in a_rows if r.get("status") == "ok"}
    b = {key(r): r for r in b_rows if r.get("status") == "ok"}
    common = sorted(set(a) & set(b))
    both = a_only = b_only = neither = disagreements = 0

    by_type = defaultdict(lambda: [0, 0, 0, 0])
    by_workflow = defaultdict(lambda: [0, 0, 0, 0])

    for k in common:
        ar, br = a[k], b[k]
        av, bv = bool(ar["correct"]), bool(br["correct"])
        if av and bv:
            both += 1
            slot = 0
        elif av and not bv:
            a_only += 1
            slot = 1
        elif not av and bv:
            b_only += 1
            slot = 2
        else:
            neither += 1
            slot = 3
        if ar["predicted"] != br["predicted"]:
            disagreements += 1
        by_type[ar["question_type"]][slot] += 1
        by_workflow[ar["workflow"]][slot] += 1

    return {
        "n": len(common),
        "both_correct": both,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": neither,
        "disagreements": disagreements,
        "p_exact": exact_two_sided(a_only, b_only),
        "by_type": dict(by_type),
        "by_workflow": dict(by_workflow),
    }


def case_exact(rows):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(bool(row["correct"]))
    return {
        case_id: bool(values) and all(values)
        for case_id, values in by_case.items()
    }


def case_pair(a_rows, b_rows):
    a = case_exact(a_rows)
    b = case_exact(b_rows)
    ids = sorted(set(a) & set(b))
    both = a_only = b_only = neither = 0
    for cid in ids:
        if a[cid] and b[cid]:
            both += 1
        elif a[cid] and not b[cid]:
            a_only += 1
        elif not a[cid] and b[cid]:
            b_only += 1
        else:
            neither += 1
    return {
        "n": len(ids),
        "both_correct": both,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": neither,
        "p_exact": exact_two_sided(a_only, b_only),
        "jev_exact_accuracy": sum(a[cid] for cid in ids) / len(ids) if ids else None,
        "dashscope_exact_accuracy": sum(b[cid] for cid in ids) / len(ids) if ids else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-summary", required=True)
    ap.add_argument("--dashscope-summary", required=True)
    ap.add_argument("--jev-decisions", required=True)
    ap.add_argument("--dashscope-decisions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    jev = load_json(args.jev_summary)
    dash = load_json(args.dashscope_summary)
    jr = load_jsonl(args.jev_decisions)
    dr = load_jsonl(args.dashscope_decisions)

    pair = paired(jr, dr)
    cases = case_pair(jr, dr)

    lines = [
        "# Jev vs DashScope — typed-decisions 100-case stratified sample",
        "",
        "## Scope",
        "",
        "This experiment uses 100 cases from the pinned LocalLLaMA/typed-decisions official test split.",
        "",
        "- 4 workflow families",
        "- exactly 25 cases per workflow",
        "- 5 typed questions per case",
        "- 500 paired decisions total",
        "- both providers receive identical state and question objects",
        "- TypeSafe Jev is evaluated as a hosted general System One backend",
        "- DashScope uses decision-model-preview through its System One-compatible endpoint",
        "",
        "## Main result",
        "",
        "| Metric | TypeSafe Jev | DashScope decision-model-preview |",
        "|---|---:|---:|",
        f"| Cases OK | {jev['cases_ok']}/{jev['cases']} | {dash['cases_ok']}/{dash['cases']} |",
        f"| Decisions | {jev.get('n',0)} | {dash.get('n',0)} |",
        f"| Accuracy | {pct(jev.get('accuracy'))} | {pct(dash.get('accuracy'))} |",
        f"| Soft accuracy | {num(jev.get('soft_accuracy'))} | {num(dash.get('soft_accuracy'))} |",
        f"| KL from gold ↓ | {num(jev.get('kl'))} | {num(dash.get('kl'))} |",
        f"| TV ↓ | {num(jev.get('tv'))} | {num(dash.get('tv'))} |",
        f"| Hard-label Brier ↓ | {num(jev.get('brier'))} | {num(dash.get('brier'))} |",
        f"| Soft-distribution Brier ↓ | {num(jev.get('brier_soft'))} | {num(dash.get('brier_soft'))} |",
        f"| ECE ↓ | {num(jev.get('ece'))} | {num(dash.get('ece'))} |",
        f"| Score MAE ↓ | {num(jev.get('score_mae'))} | {num(dash.get('score_mae'))} |",
        f"| Within 1 score level | {pct(jev.get('within_1_level'))} | {pct(dash.get('within_1_level'))} |",
        f"| Case p50 latency | {ms(jev.get('case_latency_ms',{}).get('p50'))} ms | {ms(dash.get('case_latency_ms',{}).get('p50'))} ms |",
        f"| Case p95 latency | {ms(jev.get('case_latency_ms',{}).get('p95'))} ms | {ms(dash.get('case_latency_ms',{}).get('p95'))} ms |",
        "",
        "Latency is end-to-end from GitHub-hosted runners to different hosted services and regions; it is not pure model-compute latency.",
        "",
        "## Paired hard-decision comparison",
        "",
        f"- both correct: **{pair['both_correct']}/{pair['n']}**",
        f"- Jev correct / DashScope wrong: **{pair['a_only']}**",
        f"- DashScope correct / Jev wrong: **{pair['b_only']}**",
        f"- both wrong: **{pair['both_wrong']}**",
        f"- different predicted labels: **{pair['disagreements']}/{pair['n']}**",
        f"- exact two-sided discordant-pair p: **{pair['p_exact']:.4g}**",
        "",
        "## Whole-case exact match",
        "",
        "A case counts as exact-match correct only when all five typed questions are correct.",
        "",
        f"- Jev exact-case accuracy: **{pct(cases['jev_exact_accuracy'])}**",
        f"- DashScope exact-case accuracy: **{pct(cases['dashscope_exact_accuracy'])}**",
        f"- both cases exact: **{cases['both_correct']}/{cases['n']}**",
        f"- Jev-only exact cases: **{cases['a_only']}**",
        f"- DashScope-only exact cases: **{cases['b_only']}**",
        f"- neither exact: **{cases['both_wrong']}**",
        f"- exact-case discordant p: **{cases['p_exact']:.4g}**",
        "",
        "## By question type",
        "",
        "| Type | Jev accuracy | DashScope accuracy | Jev Brier ↓ | DashScope Brier ↓ | Jev ECE ↓ | DashScope ECE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    qtypes = sorted(
        set(jev.get("by_question_type", {}))
        | set(dash.get("by_question_type", {}))
    )
    for qt in qtypes:
        j = jev["by_question_type"].get(qt, {})
        d = dash["by_question_type"].get(qt, {})
        lines.append(
            f"| {qt} | {pct(j.get('accuracy'))} | {pct(d.get('accuracy'))} | "
            f"{num(j.get('brier'))} | {num(d.get('brier'))} | "
            f"{num(j.get('ece'))} | {num(d.get('ece'))} |"
        )

    lines += [
        "",
        "## By workflow",
        "",
        "| Workflow | Jev accuracy | DashScope accuracy | Jev Brier ↓ | DashScope Brier ↓ |",
        "|---|---:|---:|---:|---:|",
    ]

    workflows = sorted(
        set(jev.get("by_workflow", {}))
        | set(dash.get("by_workflow", {}))
    )
    for wf in workflows:
        j = jev["by_workflow"].get(wf, {})
        d = dash["by_workflow"].get(wf, {})
        lines.append(
            f"| {wf} | {pct(j.get('accuracy'))} | {pct(d.get('accuracy'))} | "
            f"{num(j.get('brier'))} | {num(d.get('brier'))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This sample is larger and structurally richer than the 77-case BANKING77 slice because each request contains multiple typed decisions.",
        "- Stratification prevents one workflow family from dominating the result.",
        "- Hard accuracy and distributional metrics answer different questions; higher argmax accuracy does not automatically imply better probability calibration.",
        "- The test split is held out from the typed-decisions training split, but it shares the same four workflow families.",
        "- This experiment compares two hosted general-purpose decision services; it does not include the Laya typed specialist.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
