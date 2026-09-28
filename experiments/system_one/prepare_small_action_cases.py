import argparse, json
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from prepare_cases import LABELS

FAMILIES = {
    "card-payment": [
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "declined_card_payment",
        "pending_card_payment",
    ],
    "cash-withdrawal": [
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal",
        "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
    ],
    "transfer": [
        "cancel_transfer",
        "declined_transfer",
        "failed_transfer",
        "pending_transfer",
        "transfer_fee_charged",
    ],
    "top-up": [
        "top_up_by_card_charge",
        "top_up_failed",
        "pending_top_up",
        "top_up_reverted",
        "top_up_limits",
    ],
}

PAIR = {
    "card-payment": "cash-withdrawal",
    "cash-withdrawal": "card-payment",
    "transfer": "top-up",
    "top-up": "transfer",
}

def question(options):
    return {
        "decision": {
            "type": "choice",
            "instructions": "Classify the customer's banking intent. Choose exactly one intent.",
            "criteria": {x: x.replace("_", " ") for x in options},
        }
    }

def rotate(xs, n):
    n %= len(xs)
    return xs[n:] + xs[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--examples-per-intent", type=int, default=5)
    args = ap.parse_args()

    ds = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/PolyAI/banking77/resolve/a00bafa8d1db84baced2c9032e3f227f9c486cb2/data/test-00000-of-00001.parquet",
        split="train",
    )
    label_to_id = {name: i for i, name in enumerate(LABELS)}
    selected = {x for group in FAMILIES.values() for x in group}
    samples = defaultdict(list)
    for row in ds:
        name = LABELS[int(row["label"])]
        if name in selected and len(samples[name]) < args.examples_per_intent:
            samples[name].append(row["text"])
        if all(len(samples[x]) >= args.examples_per_intent for x in selected):
            break

    rows = []
    family_names = list(FAMILIES)
    all20 = [x for fam in family_names for x in FAMILIES[fam]]

    for fam_idx, fam in enumerate(family_names):
        group5 = FAMILIES[fam]
        group10 = group5 + FAMILIES[PAIR[fam]]
        for label_idx, expected in enumerate(group5):
            peers = [x for x in group5 if x != expected]
            options3 = [expected, peers[label_idx % len(peers)], peers[(label_idx + 1) % len(peers)]]
            option_sets = {3: options3, 5: group5, 10: group10, 20: all20}
            for sample_idx, state in enumerate(samples[expected]):
                base = f"{fam}-{label_idx}-{sample_idx}"
                for k, options in option_sets.items():
                    # Rotate deterministically so the gold label is not pinned to one position.
                    ordered = rotate(list(options), (sample_idx + label_idx + fam_idx) % len(options))
                    rows.append({
                        "id": f"small-k{k}-{base}",
                        "source": "PolyAI/banking77:test",
                        "family": f"small-k{k}",
                        "scenario": "banking-small-action-space",
                        "domain": fam,
                        "action_space_size": k,
                        "state": state,
                        "questions": question(ordered),
                        "expected": {"decision": expected},
                    })

                # Order sensitivity: same 5-way choice, three deterministic orderings.
                orders = {
                    "original": list(group5),
                    "reversed": list(reversed(group5)),
                    "rotated": rotate(list(group5), (sample_idx + label_idx + 1) % 5),
                }
                for variant, ordered in orders.items():
                    rows.append({
                        "id": f"small-order-{variant}-{base}",
                        "source": "PolyAI/banking77:test",
                        "family": f"small-order-{variant}",
                        "scenario": "banking-small-action-order",
                        "domain": fam,
                        "action_space_size": 5,
                        "state": state,
                        "questions": question(ordered),
                        "expected": {"decision": expected},
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cases ({len(rows)//7} base examples x 7 variants) to {args.output}")

if __name__ == "__main__":
    main()
