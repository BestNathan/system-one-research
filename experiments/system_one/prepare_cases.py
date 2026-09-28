import argparse, json, random
from pathlib import Path
from datasets import load_dataset

LABELS=[
"activate_my_card","age_limit","apple_pay_or_google_pay","atm_support","automatic_top_up",
"balance_not_updated_after_bank_transfer","balance_not_updated_after_cheque_or_cash_deposit","beneficiary_not_allowed",
"cancel_transfer","card_about_to_expire","card_acceptance","card_arrival","card_delivery_estimate","card_linking",
"card_not_working","card_payment_fee_charged","card_payment_not_recognised","card_payment_wrong_exchange_rate",
"card_swallowed","cash_withdrawal_charge","cash_withdrawal_not_recognised","change_pin","compromised_card",
"contactless_not_working","country_support","declined_card_payment","declined_cash_withdrawal","declined_transfer",
"direct_debit_payment_not_recognised","disposable_card_limits","edit_personal_details","exchange_charge","exchange_rate",
"exchange_via_app","extra_charge_on_statement","failed_transfer","fiat_currency_support","get_disposable_virtual_card",
"get_physical_card","getting_spare_card","getting_virtual_card","lost_or_stolen_card","lost_or_stolen_phone",
"order_physical_card","passcode_forgotten","pending_card_payment","pending_cash_withdrawal","pending_top_up",
"pending_transfer","pin_blocked","receiving_money","Refund_not_showing_up","request_refund","reverted_card_payment?",
"supported_cards_and_currencies","terminate_account","top_up_by_bank_transfer_charge","top_up_by_card_charge",
"top_up_by_cash_or_cheque","top_up_failed","top_up_limits","top_up_reverted","topping_up_by_card",
"transaction_charged_twice","transfer_fee_charged","transfer_into_account","transfer_not_received_by_recipient",
"transfer_timing","unable_to_verify_identity","verify_my_identity","verify_source_of_funds","verify_top_up",
"virtual_card_not_working","visa_or_mastercard","why_verify_identity","wrong_amount_of_cash_received",
"wrong_exchange_rate_for_cash_withdrawal"]

def synthetic():
    rows=[]
    examples=[
      ("prod outage","Production checkout is unavailable for every customer.","high",["low","medium","high"]),
      ("minor typo","There is a typo in an internal help page.","low",["low","medium","high"]),
      ("refund","I was charged twice and need the duplicate refunded.","billing",["billing","technical","other"]),
      ("crash","The application crashes when Settings opens.","technical",["billing","technical","other"]),
    ]
    for i,(title,state,expected,opts) in enumerate(examples):
        rows.append({"id":f"synthetic-{i}","source":"synthetic","family":"workflow-routing","state":state,
          "questions":{"decision":{"type":"choice","instructions":"Select the best action/category.","criteria":{x:x.replace("_"," ") for x in opts}}},
          "expected":{"decision":expected}})
    return rows

def banking(limit=154):
    ds=load_dataset("PolyAI/banking77",split="test")
    # deterministic balanced-ish sample: first two examples per label, then fill.
    seen={}; chosen=[]
    for row in ds:
        y=int(row["label"]); n=seen.get(y,0)
        if n<2: chosen.append(row); seen[y]=n+1
        if len(chosen)>=limit: break
    criteria={x:x.replace("_"," ") for x in LABELS}
    return [{"id":f"banking77-{i}","source":"PolyAI/banking77:test","family":"intent-77","state":r["text"],
      "questions":{"decision":{"type":"choice","instructions":"Classify the customer's banking intent.","criteria":criteria}},
      "expected":{"decision":LABELS[int(r["label"])]}} for i,r in enumerate(chosen)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--banking-limit",type=int,default=154); a=ap.parse_args()
    rows=synthetic()+banking(a.banking_limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
    print(f"wrote {len(rows)} cases to {a.output}")
if __name__=="__main__": main()
