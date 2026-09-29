from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return "-"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", required=True)
    ap.add_argument("--dashscope", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    jev = load(args.jev)
    dash = load(args.dashscope)
    case = load(args.case)

    lines = [
        "# Jev vs DashScope — System One protocol smoke",
        "",
        f"Case: {case.get('id')}",
        "",
        "Both backends received the exact same state and questions payload.",
        "",
        "| Backend | Requested model | Reported model | Latency ms | Auth |",
        "|---|---|---|---:|---|",
        f"| Jev | {jev.get('model_requested')} | {jev.get('model_reported') or '-'} | {jev.get('elapsed_ms', 0):.1f} | {jev.get('auth_variant')} |",
        f"| DashScope | {dash.get('model_requested')} | {dash.get('model_reported') or '-'} | {dash.get('elapsed_ms', 0):.1f} | {dash.get('auth_variant')} |",
        "",
        "## Decisions",
        "",
        "| Question | Type | Jev hard/value | Jev confidence | DashScope hard/value | DashScope confidence |",
        "|---|---|---|---:|---|---:|",
    ]

    ja = jev.get("answers") or {}
    da = dash.get("answers") or {}
    for qid, qdef in case["questions"].items():
        typ = qdef["type"]
        j = ja.get(qid, {})
        d = da.get(qid, {})
        if typ == "choice":
            jv, dv = j.get("choice"), d.get("choice")
        elif typ == "noul":
            jv, dv = j.get("noul"), d.get("noul")
        else:
            jv, dv = j.get("score"), d.get("score")
        lines.append(
            f"| {qid} | {typ} | {fmt(jv)} | {fmt(j.get('confidence'))} | "
            f"{fmt(dv)} | {fmt(d.get('confidence'))} |"
        )

    lines += ["", "## Probability distributions", ""]
    for qid in case["questions"]:
        lines += [
            f"### {qid}",
            "",
            "Jev",
            "",
            "~~~json",
            json.dumps(ja.get(qid), ensure_ascii=False, indent=2),
            "~~~",
            "",
            "DashScope",
            "",
            "~~~json",
            json.dumps(da.get(qid), ensure_ascii=False, indent=2),
            "~~~",
            "",
        ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
