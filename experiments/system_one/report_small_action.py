import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def pred(row):
    return row["answers"]["decision"]["predicted"]

def conf(row):
    d = row["answers"]["decision"]
    if isinstance(d.get("confidence"), (int, float)):
        return float(d["confidence"])
    raw = d.get("raw") or {}
    for key in ("confidence", "answer_confidence", "probability"):
        if isinstance(raw.get(key), (int, float)):
            return float(raw[key])
    probs = raw.get("probabilities")
    p = d.get("predicted")
    if isinstance(probs, dict) and p in probs and isinstance(probs[p], (int, float)):
        return float(probs[p])
    return None

def percentile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round((len(ys) - 1) * p))]

def stats(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": 0, "accuracy": None, "mean_ms": None, "p95_ms": None, "mean_conf": None, "high_conf_acc": None}
    cs = [c for r in ok if (c := conf(r)) is not None]
    hc = [r for r in ok if (conf(r) is not None and conf(r) >= .9)]
    return {
        "n": len(ok),
        "accuracy": sum(r["correct"] for r in ok) / len(ok),
        "mean_ms": statistics.mean(r["elapsed_ms"] for r in ok),
        "p95_ms": percentile([r["elapsed_ms"] for r in ok], .95),
        "mean_conf": statistics.mean(cs) if cs else None,
        "high_conf_acc": (sum(r["correct"] for r in hc) / len(hc)) if hc else None,
        "high_conf_n": len(hc),
    }

def fmt(x, pct=False, digits=1):
    if x is None:
        return "-"
    return f"{x*100:.{digits}f}%" if pct else f"{x:.{digits}f}"

def action_rows(rows):
    out = {}
    for k in (3,5,10,20):
        out[k] = stats([r for r in rows if r.get("family") == f"small-k{k}"])
    return out

def transition(rows, a=3, b=20):
    by = defaultdict(dict)
    for r in rows:
        fam = r.get("family", "")
        if fam not in (f"small-k{a}", f"small-k{b}"):
            continue
        base = r["id"].split("-", 2)[2]
        by[base][int(fam.replace("small-k",""))] = bool(r.get("correct"))
    z = {"both_correct":0, "correct_to_wrong":0, "wrong_to_correct":0, "both_wrong":0}
    for v in by.values():
        if a not in v or b not in v:
            continue
        if v[a] and v[b]: z["both_correct"] += 1
        elif v[a] and not v[b]: z["correct_to_wrong"] += 1
        elif not v[a] and v[b]: z["wrong_to_correct"] += 1
        else: z["both_wrong"] += 1
    return z

def order_stats(rows):
    variants = {}
    for name in ("original","reversed","rotated"):
        variants[name] = stats([r for r in rows if r.get("family") == f"small-order-{name}"])
    by = defaultdict(list)
    for r in rows:
        fam = r.get("family", "")
        if not fam.startswith("small-order-") or r.get("status") != "ok":
            continue
        base = r["id"].split("-", 3)[3]
        by[base].append(pred(r))
    complete = [xs for xs in by.values() if len(xs) == 3]
    unstable = sum(len(set(xs)) > 1 for xs in complete)
    return variants, (unstable / len(complete) if complete else None), len(complete)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--laya-small", required=True)
    ap.add_argument("--jev-small", required=True)
    ap.add_argument("--laya-baseline", required=True)
    ap.add_argument("--jev-baseline", required=True)
    ap.add_argument("--output", required=True)
    a=ap.parse_args()

    models = {
        "Laya": load_jsonl(a.laya_small),
        "Jev": load_jsonl(a.jev_small),
    }
    baseline = {
        "Laya": load_json(a.laya_baseline)["by_family"]["intent-77"],
        "Jev": load_json(a.jev_baseline)["by_family"]["intent-77"],
    }

    curves = {m: action_rows(rows) for m, rows in models.items()}
    transitions = {m: transition(rows) for m, rows in models.items()}
    orders = {m: order_stats(rows) for m, rows in models.items()}

    lines = [
        "# Small Action-Space Experiment",
        "",
        "## Design",
        "The same BANKING77 examples are evaluated against nested candidate sets. The selected 20 intents are deliberately confusable and come from card-payment, cash-withdrawal, transfer, and top-up domains.",
        "",
        "- 100 base examples: 20 intents × 5 examples each.",
        "- Action-space sweep: 3 / 5 / 10 / 20 choices for the same base examples.",
        "- 5-way order test: original / reversed / deterministic rotation of the same options.",
        "- The existing 77-way BANKING77 run is included as an external cardinality reference, but it uses 154 examples rather than this exact 100-example subset.",
        "",
        "## Accuracy vs action-space size",
        "",
        "| Actions | Laya accuracy | Jev accuracy | Laya mean ms | Jev mean ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (3,5,10,20):
        l,j=curves["Laya"][k],curves["Jev"][k]
        lines.append(f"| {k} | {fmt(l['accuracy'],True)} | {fmt(j['accuracy'],True)} | {fmt(l['mean_ms'])} | {fmt(j['mean_ms'])} |")
    lines.append(f"| 77* | {fmt(baseline['Laya']['accuracy'],True)} | {fmt(baseline['Jev']['accuracy'],True)} | - | - |")
    lines += [
        "",
        "\* 77-way is the existing 154-case BANKING77 baseline, not the exact same 100-case subset.",
        "",
        "## Confidence",
        "",
        "| Actions | Laya mean conf | Laya acc @ conf≥.90 | Jev mean conf | Jev acc @ conf≥.90 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (3,5,10,20):
        l,j=curves["Laya"][k],curves["Jev"][k]
        lines.append(f"| {k} | {fmt(l['mean_conf'],True)} | {fmt(l['high_conf_acc'],True)} ({l.get('high_conf_n',0)}) | {fmt(j['mean_conf'],True)} | {fmt(j['high_conf_acc'],True)} ({j.get('high_conf_n',0)}) |")

    lines += ["", "## Paired 3-way → 20-way transitions", ""]
    for m,z in transitions.items():
        n=sum(z.values())
        lines += [
            f"### {m}",
            f"- both correct: {z['both_correct']}/{n}",
            f"- correct at 3-way → wrong at 20-way: {z['correct_to_wrong']}/{n}",
            f"- wrong at 3-way → correct at 20-way: {z['wrong_to_correct']}/{n}",
            f"- both wrong: {z['both_wrong']}/{n}",
            "",
        ]

    lines += [
        "## 5-way option-order robustness",
        "",
        "| Model | Original | Reversed | Rotated | Prediction changes across orderings |",
        "|---|---:|---:|---:|---:|",
    ]
    for m,(v,unstable,n) in orders.items():
        lines.append(f"| {m} | {fmt(v['original']['accuracy'],True)} | {fmt(v['reversed']['accuracy'],True)} | {fmt(v['rotated']['accuracy'],True)} | {fmt(unstable,True)} ({n} bases) |")

    lines += [
        "",
        "## Interpretation",
        "- If accuracy falls monotonically as 3→5→10→20 grows while the input examples stay fixed, action-space cardinality is a causal contributor rather than merely a dataset-difficulty artifact.",
        "- A large 3-way→20-way correct-to-wrong count is stronger evidence of cardinality sensitivity than comparing unrelated aggregate datasets.",
        "- Order instability indicates the decision head is sensitive to presentation order even when the semantic action set is unchanged.",
        "- Confidence should be treated separately from accuracy; high confidence on incorrect decisions is unsafe for autonomous execution gates.",
    ]
    Path(a.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
