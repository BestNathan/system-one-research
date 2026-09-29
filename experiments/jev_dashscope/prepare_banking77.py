import argparse
import json
from pathlib import Path

from datasets import load_dataset

LABELS = [
    "activate_my_card","age_limit","apple_pay_or_google_pay","atm_support","automatic_top_up",
    "balance_not_updated_after_bank_transfer","balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed","cancel_transfer","card_about_to_expire","card_acceptance","card_arrival",
    "card_delivery_estimate","card_linking","card_not_working","card_payment_fee_charged",
    "card_payment_not_recognised","card_payment_wrong_exchange_rate","card_swallowed",
    "cash_withdrawal_charge","cash_withdrawal_not_recognised","change_pin","compromised_card",
    "contactless_not_working","country_support","declined_card_payment","declined_cash_withdrawal",
    "declined_transfer","direct_debit_payment_not_recognised","disposable_card_limits",
    "edit_personal_details","exchange_charge","exchange_rate","exchange_via_app",
    "extra_charge_on_statement","failed_transfer","fiat_currency_support",
    "get_disposable_virtual_card","get_physical_card","getting_spare_card","getting_virtual_card",
    "lost_or_stolen_card","lost_or_stolen_phone","order_physical_card","passcode_forgotten",
    "pending_card_payment","pending_cash_withdrawal","pending_top_up","pending_transfer","pin_blocked",
    "receiving_money","Refund_not_showing_up","request_refund","reverted_card_payment?",
    "supported_cards_and_currencies","terminate_account","top_up_by_bank_transfer_charge",
    "top_up_by_card_charge","top_up_by_cash_or_cheque","top_up_failed","top_up_limits",
    "top_up_reverted","topping_up_by_card","transaction_charged_twice","transfer_fee_charged",
    "transfer_into_account","transfer_not_received_by_recipient","transfer_timing",
    "unable_to_verify_identity","verify_my_identity","verify_source_of_funds","verify_top_up",
    "virtual_card_not_working","visa_or_mastercard","why_verify_identity",
    "wrong_amount_of_cash_received","wrong_exchange_rate_for_cash_withdrawal",
]

DATA_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/a00bafa8d1db84baced2c9032e3f227f9c486cb2/data/test-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files=DATA_URL, split="train")
    by_label = {}
    for row in ds:
        label = int(row["label"])
        by_label.setdefault(label, row)
        if len(by_label) == len(LABELS):
            break

    missing = [i for i in range(len(LABELS)) if i not in by_label]
    if missing:
        raise SystemExit(f"missing BANKING77 labels: {missing}")

    criteria = {name: name.replace("_", " ") for name in LABELS}
    rows = []
    for label, name in enumerate(LABELS):
        source = by_label[label]
        rows.append({
            "id": f"banking77-one-per-intent-{label:02d}",
            "source": "PolyAI/banking77:test",
            "family": "intent-77",
            "state": source["text"],
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Classify the customer's banking intent.",
                    "criteria": criteria,
                }
            },
            "expected": {"decision": name},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} BANKING77 cases: one case per intent")


if __name__ == "__main__":
    main()
