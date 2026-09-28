import argparse
import json
import math
from pathlib import Path

OFFICIAL = {
    "laya_typed": {"accuracy": 0.766, "soft_accuracy": 0.471, "brier": 0.062, "ece": 0.213, "score_mae": 0.242},
    "jev": {"accuracy": 0.727, "soft_accuracy": 0.580, "kl": 1.442, "tv": 0.251, "brier": 0.148, "ece": 0.144, "score_mae": 0.391, "within_1_level": 0.952},
}

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def pct(v):
    return "-" if v is None else f"{v*100:.1f}%"

def num(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"

def ms(v):
    return "-" if v is None else f"{v:.0f}"

def key(row):
    return (row["case_id"], row["question_id"])

def exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)

def paired(a_rows, b_rows):
    a = {key(r): bool(r["correct"]) for r in a_rows if r.get("status") == "ok"}
    b = {key(r): bool(r["correct"]) for r in b_rows if r.get("status") == "ok"}
    common = sorted(set(a) & set(b))
    both = a_only = b_only = neither = 0
    for k in common:
        av, bv = a[k], b[k]
        if av and bv:
            both += 1
        elif av and not bv:
            a_only += 1
        elif not av and bv:
            b_only += 1
        else:
            neither += 1
    return {"n": len(common), "both_correct": both, "a_only": a_only, "b_only": b_only, "both_wrong": neither, "p_exact": exact_two_sided(a_only, b_only)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-summary", required=True)
    ap.add_argument("--typed-summary", required=True)
    ap.add_argument("--jev-summary", required=True)
    ap.add_argument("--base-decisions", required=True)
    ap.add_argument("--typed-decisions", required=True)
    ap.add_argument("--jev-decisions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = load_json(args.base_summary)
    typed = load_json(args.typed_summary)
    jev = load_json(args.jev_summary)
    base_rows = load_jsonl(args.base_decisions)
    typed_rows = load_jsonl(args.typed_decisions)
    jev_rows = load_jsonl(args.jev_decisions)

    bt = paired(base_rows, typed_rows)
    tj = paired(typed_rows, jev_rows)
    bj = paired(base_rows, jev_rows)

    lines = [
        "# Official typed-decisions — independent paired replication",
        "",
        "## Scope",
        "",
        "All three systems are evaluated on the exact same pinned LocalLLaMA/typed-decisions all:test split: 400 cases, 2,000 typed decisions, five questions per case.",
        "",
        "- Base Laya: general English checkpoint, zero-shot on this benchmark.",
        "- Laya typed-decisions: specialist checkpoint fine-tuned on the train split of these four workflow families.",
        "- TypeSafe Jev: general zero-shot System One service.",
        "",
        "**The specialist and generalist rows answer different product questions.** The paired numbers below are useful, but a specialist-vs-generalist accuracy gap is not a universal model ranking.",
        "",
        "## Main result",
        "",
        "| Metric | Base Laya | Laya typed specialist | TypeSafe Jev |",
        "|---|---:|---:|---:|",
        f"| Cases OK | {base['cases_ok']}/{base['cases']} | {typed['cases_ok']}/{typed['cases']} | {jev['cases_ok']}/{jev['cases']} |",
        f"| Decisions | {base.get('n',0)} | {typed.get('n',0)} | {jev.get('n',0)} |",
        f"| Accuracy | {pct(base.get('accuracy'))} | **{pct(typed.get('accuracy'))}** | {pct(jev.get('accuracy'))} |",
        f"| Soft accuracy | {num(base.get('soft_accuracy'))} | {num(typed.get('soft_accuracy'))} | {num(jev.get('soft_accuracy'))} |",
        f"| KL from gold ↓ | {num(base.get('kl'))} | {num(typed.get('kl'))} | {num(jev.get('kl'))} |",
        f"| TV ↓ | {num(base.get('tv'))} | {num(typed.get('tv'))} | {num(jev.get('tv'))} |",
        f"| Brier ↓ | {num(base.get('brier'))} | {num(typed.get('brier'))} | {num(jev.get('brier'))} |",
        f"| ECE ↓ | {num(base.get('ece'))} | {num(typed.get('ece'))} | {num(jev.get('ece'))} |",
        f"| Score MAE ↓ | {num(base.get('score_mae'))} | {num(typed.get('score_mae'))} | {num(jev.get('score_mae'))} |",
        f"| Within 1 score level | {pct(base.get('within_1_level'))} | {pct(typed.get('within_1_level'))} | {pct(jev.get('within_1_level'))} |",
        f"| Case latency p50 ms | {ms(base.get('case_latency_ms',{}).get('p50'))} | {ms(typed.get('case_latency_ms',{}).get('p50'))} | {ms(jev.get('case_latency_ms',{}).get('p50'))} |",
        f"| Case latency p95 ms | {ms(base.get('case_latency_ms',{}).get('p95'))} | {ms(typed.get('case_latency_ms',{}).get('p95'))} | {ms(jev.get('case_latency_ms',{}).get('p95'))} |",
        "",
        "Latency is not hardware-equivalent: both Laya checkpoints run locally on the GitHub-hosted CPU runner, while Jev is a hosted API.",
        "",
        "## Reproduction against published reference values",
        "",
        "| Metric | Laya typed: this run | Laya typed: published | Jev: this run | Jev: dataset-card measurement |",
        "|---|---:|---:|---:|---:|",
        f"| Accuracy | {num(typed.get('accuracy'))} | {OFFICIAL['laya_typed']['accuracy']:.3f} | {num(jev.get('accuracy'))} | {OFFICIAL['jev']['accuracy']:.3f} |",
        f"| Soft accuracy | {num(typed.get('soft_accuracy'))} | {OFFICIAL['laya_typed']['soft_accuracy']:.3f} | {num(jev.get('soft_accuracy'))} | {OFFICIAL['jev']['soft_accuracy']:.3f} |",
        f"| Brier ↓ | {num(typed.get('brier'))} | {OFFICIAL['laya_typed']['brier']:.3f} | {num(jev.get('brier'))} | {OFFICIAL['jev']['brier']:.3f} |",
        f"| ECE ↓ | {num(typed.get('ece'))} | {OFFICIAL['laya_typed']['ece']:.3f} | {num(jev.get('ece'))} | {OFFICIAL['jev']['ece']:.3f} |",
        f"| Score MAE ↓ | {num(typed.get('score_mae'))} | {OFFICIAL['laya_typed']['score_mae']:.3f} | {num(jev.get('score_mae'))} | {OFFICIAL['jev']['score_mae']:.3f} |",
        "",
        "## Paired hard-decision comparison",
        "",
        "Every paired table uses the same (case_id, question_id) decisions.",
        "",
        "### Typed specialist vs Jev",
        "",
        f"- both correct: {tj['both_correct']}/{tj['n']}",
        f"- typed specialist correct / Jev wrong: **{tj['a_only']}**",
        f"- Jev correct / typed specialist wrong: **{tj['b_only']}**",
        f"- both wrong: {tj['both_wrong']}",
        f"- exact two-sided discordant-pair p: {tj['p_exact']:.4g}",
        "",
        "### Base Laya vs typed specialist",
        "",
        f"- both correct: {bt['both_correct']}/{bt['n']}",
        f"- base only: {bt['a_only']}",
        f"- typed specialist only: {bt['b_only']}",
        f"- both wrong: {bt['both_wrong']}",
        f"- exact two-sided discordant-pair p: {bt['p_exact']:.4g}",
        "",
        "### Base Laya vs Jev",
        "",
        f"- both correct: {bj['both_correct']}/{bj['n']}",
        f"- base only: {bj['a_only']}",
        f"- Jev only: {bj['b_only']}",
        f"- both wrong: {bj['both_wrong']}",
        f"- exact two-sided discordant-pair p: {bj['p_exact']:.4g}",
        "",
        "## By question type",
        "",
        "| Type | Base Laya | Typed specialist | Jev |",
        "|---|---:|---:|---:|",
    ]

    qtypes = sorted(set(base.get("by_question_type",{})) | set(typed.get("by_question_type",{})) | set(jev.get("by_question_type",{})))
    for qt in qtypes:
        lines.append(f"| {qt} | {pct(base.get('by_question_type',{}).get(qt,{}).get('accuracy'))} | {pct(typed.get('by_question_type',{}).get(qt,{}).get('accuracy'))} | {pct(jev.get('by_question_type',{}).get(qt,{}).get('accuracy'))} |")

    lines += ["", "## By workflow", "", "| Workflow | Base Laya | Typed specialist | Jev |", "|---|---:|---:|---:|"]
    workflows = sorted(set(base.get("by_workflow",{})) | set(typed.get("by_workflow",{})) | set(jev.get("by_workflow",{})))
    for wf in workflows:
        lines.append(f"| {wf} | {pct(base.get('by_workflow',{}).get(wf,{}).get('accuracy'))} | {pct(typed.get('by_workflow',{}).get(wf,{}).get('accuracy'))} | {pct(jev.get('by_workflow',{}).get(wf,{}).get('accuracy'))} |")

    lines += [
        "",
        "## Interpretation rules",
        "",
        "1. If the typed checkpoint reproduces roughly the published 0.766 accuracy, its advertised in-domain result is independently supported.",
        "2. If it also exceeds Jev on the same 2,000 decisions, the statement 'the Laya typed specialist beats Jev on this benchmark's hard-label accuracy' is supported for this dataset.",
        "3. That does not mean the specialist is a stronger general System One model: it was trained on these workflow families, while Jev is evaluated zero-shot.",
        "4. Distributional metrics matter separately. A model can have higher argmax accuracy while being much worse at reproducing the teacher's uncertainty.",
        "5. The benchmark gold is a synthetic teacher signal rather than real-world ground truth, so the result measures agreement with that decision process.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
