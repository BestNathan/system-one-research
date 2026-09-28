import argparse, json, math, statistics
from pathlib import Path
from backends import create_backend

def confidence(a):
    for k in ("confidence","answer_confidence","probability"):
        if isinstance(a,dict) and isinstance(a.get(k),(int,float)): return float(a[k])
    return None

def predicted(a):
    if not isinstance(a,dict): return a
    for k in ("choice","answer","value","noul","score"):
        if k in a: return a[k]
    return None

def percentile(xs,p):
    if not xs:return None
    ys=sorted(xs); return ys[min(len(ys)-1,round((len(ys)-1)*p))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--backend",choices=["laya","jev"],required=True); ap.add_argument("--cases",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in a.cases.read_text(encoding="utf-8").splitlines() if x.strip()]
    b=create_backend(a.backend); out=[]
    for case in rows:
        try:
            body,ms=b.decide(case["state"],case["questions"])
            answers=body["answers"]; ok=True; details={}
            for qid,expected in case["expected"].items():
                ans=answers[qid]; pred=predicted(ans); good=pred==expected; ok &= good
                details[qid]={"expected":expected,"predicted":pred,"confidence":confidence(ans),"correct":good,"raw":ans}
            out.append({**case,"status":"ok","elapsed_ms":ms,"correct":ok,"answers":details,"usage":body.get("usage")})
        except Exception as e:
            out.append({**case,"status":"error","error_type":type(e).__name__,"error":str(e)[:500]})
    good=[x for x in out if x["status"]=="ok"]; lat=[x["elapsed_ms"] for x in good]
    fam={}
    for x in good:
        z=fam.setdefault(x["family"],[0,0]); z[1]+=1; z[0]+=int(x["correct"])
    summary={"backend":a.backend,"model":getattr(b,"model",None),"load_ms":getattr(b,"load_ms",None),"n":len(out),"ok":len(good),
      "accuracy":sum(int(x["correct"]) for x in good)/len(good) if good else None,
      "latency_ms":{"mean":statistics.mean(lat) if lat else None,"p50":percentile(lat,.5),"p95":percentile(lat,.95)},
      "by_family":{k:{"accuracy":v[0]/v[1],"n":v[1]} for k,v in fam.items()}}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"raw.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in out)+"\n",encoding="utf-8")
    (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
