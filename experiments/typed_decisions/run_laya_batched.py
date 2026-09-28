from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from common import (
    DATASET,
    DATASET_REVISION,
    option_keys,
    parse_json,
    record_from_probs,
    select_cases_per_workflow,
    summarize,
)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket


def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


@torch.no_grad()
def score_cases(agent, cases, max_tokens=8192, max_seqs=64):
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0

    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, mk = build_sequence(agent.tok, state, q, max_len, hml)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0))
                items.append(None)
                dropped += 1
                continue
            if len(mk) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0))
                items.append(None)
                dropped += 1
                continue
            items.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(mk)))

    order = sorted(
        [i for i, it in enumerate(items) if it is not None],
        key=lambda i: len(items[i]["ids"]),
    )
    out = [None] * len(items)
    t0 = time.perf_counter()
    i = 0

    while i < len(order):
        j, longest = i, 0
        while (
            j < len(order)
            and j - i < max_seqs
            and max(longest, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens
        ):
            longest = max(longest, len(items[order[j]]["ids"]))
            j += 1
        j = max(j, i + 1)

        sel = [items[order[t]] for t in range(i, j)]
        batch = collate_items([sel], agent.tok.pad_token_id)
        logits, _ = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
        logits = logits.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = logits[r, : len(sel[r]["markers"])]
        i = j

    return out, index, time.perf_counter() - t0, dropped


def softmax_t(z, t=1.0):
    z = np.asarray(z, dtype=float) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_for(agent, qt, k):
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cases-per-workflow", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(min(8, max(1, os.cpu_count() or 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    import laya

    t0 = time.perf_counter()
    agent = laya.load(args.model, device=args.device)
    load_ms = (time.perf_counter() - t0) * 1000
    agent.model.eval()

    ds = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
    ds = select_cases_per_workflow(ds, args.cases_per_workflow)

    cases = []
    meta = []
    for i, row in enumerate(ds):
        state = parse_json(row["state"])
        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        cases.append((state, questions))
        meta.append(
            {
                "case_index": i,
                "case_id": row["id"],
                "workflow": row["workflow"],
                "questions": questions,
                "gold": gold,
            }
        )

    logits, index, seconds, dropped = score_cases(agent, cases)

    rows = []
    errors = []
    for (ci, qid, qt, k), z in zip(index, logits):
        m = meta[ci]
        qdef = m["questions"][qid]
        qtype = qdef["type"]
        keys = option_keys(qdef)
        if z is None:
            errors.append(
                {
                    "case_index": ci,
                    "case_id": m["case_id"],
                    "workflow": m["workflow"],
                    "question_id": qid,
                    "error": "Laya scoring dropped this question",
                }
            )
            continue
        p = softmax_t(z, temp_for(agent, qt, k))
        rows.append(
            record_from_probs(
                case_index=ci,
                case_id=m["case_id"],
                workflow=m["workflow"],
                question_id=qid,
                question_type=qtype,
                keys=keys,
                gold=m["gold"][qid],
                probs=p,
            )
        )

    summary = {
        "backend": "laya",
        "requested_model": args.model,
        "reported_model": args.model,
        "kind": "specialist" if "typed-decisions" in args.model else "general",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "cases": len(ds),
        "cases_ok": len(ds) if not errors else len({r["case_index"] for r in rows}),
        "errors": len(errors),
        "dropped_questions": dropped,
        "load_ms": load_ms,
        "execution_mode": "official-style batched raw-logit scoring",
        **summarize(rows, scoring_seconds=seconds, cases=len(ds)),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "decisions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
        encoding="utf-8",
    )
    (args.output / "errors.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in errors) + ("\n" if errors else ""),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
