import argparse, json, os, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_cases(suite):
    rows=[json.loads(x) for x in (ROOT/"cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if suite=="smoke":
        return rows[:3]
    if suite=="core":
        return rows
    if suite=="feishu":
        # Seed suite remains local and deterministic. A later phase can vendor the frozen
        # 64-case Feishu diagnostic after its license/provenance files are copied verbatim.
        return [r for r in rows if "zh" in r.get("tags",[])]
    raise ValueError(suite)

def laya_backend():
    from laya import Router
    router=Router()
    def call(row):
        q={"decision":row["question"]}
        t=time.perf_counter()
        result=router.predict(row["state"], q)
        ms=(time.perf_counter()-t)*1000
        a=result["answers"]["decision"]
        return {"choice":a.get("choice"),"confidence":a.get("confidence",a.get("answer_confidence")),"raw":a}, ms
    return call

def jev_backend():
    import httpx
    key=os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is required for --backend jev")
    client=httpx.Client(timeout=60, headers={"Authorization":"Bearer "+key})
    model=os.environ.get("JEV_MODEL","jev-1.13.0")
    def call(row):
        payload={"model":model,"state":row["state"],"questions":{"decision":row["question"]}}
        t=time.perf_counter()
        r=client.post("https://api.typesafe.ai/v1/systemone",json=payload)
        ms=(time.perf_counter()-t)*1000
        r.raise_for_status()
        body=r.json()
        a=body["answers"]["decision"]
        return {"choice":a.get("choice"),"confidence":a.get("confidence",a.get("answer_confidence")),"raw":a,"model":body.get("model"),"usage":body.get("usage")}, ms
    return call

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["laya","jev"],required=True)
    ap.add_argument("--suite",choices=["smoke","core","feishu"],default="smoke")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    rows=load_cases(args.suite)
    args.output.mkdir(parents=True,exist_ok=True)
    call=laya_backend() if args.backend=="laya" else jev_backend()
    out=[]
    for row in rows:
        pred,ms=call(row)
        out.append({**row,"prediction":pred,"elapsed_ms":ms,"correct":pred.get("choice")==row["expected"]})
    lat=[x["elapsed_ms"] for x in out]
    summary={
        "backend":args.backend,
        "suite":args.suite,
        "n":len(out),
        "accuracy":sum(x["correct"] for x in out)/len(out) if out else None,
        "latency_ms":{"p50":statistics.median(lat) if lat else None,"mean":statistics.mean(lat) if lat else None},
        "notes":[
            "This harness compares identical state/question/options case-by-case.",
            "Latency is end-to-end for the chosen deployment path; local Laya and remote Jev are not hardware-equivalent.",
            "Do not combine quality and latency into one score."
        ],
    }
    (args.output/"raw.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in out)+"\n",encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
