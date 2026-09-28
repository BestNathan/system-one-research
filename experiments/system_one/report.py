import argparse,json
from pathlib import Path
def load(p): return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--laya",required=True);p.add_argument("--jev",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 l,j=load(a.laya),load(a.jev)
 lines=["# Laya vs TypeSafe Jev — System One Experiment","",
 "## Scope","Both backends received the same canonical SystemOne state/questions. Quality and systems latency are reported separately; local CPU Laya latency is not hardware-equivalent to hosted Jev latency.","",
 "## Results","| Metric | Laya | Jev |","|---|---:|---:|",
 f"| Cases completed | {l['ok']}/{l['n']} | {j['ok']}/{j['n']} |",
 f"| Overall accuracy | {l['accuracy']:.3f} | {j['accuracy']:.3f} |",
 f"| Mean latency ms | {l['latency_ms']['mean']:.1f} | {j['latency_ms']['mean']:.1f} |",
 f"| p50 latency ms | {l['latency_ms']['p50']:.1f} | {j['latency_ms']['p50']:.1f} |",
 f"| p95 latency ms | {l['latency_ms']['p95']:.1f} | {j['latency_ms']['p95']:.1f} |","",
 "## Per-family accuracy","| Family | Laya | Jev |","|---|---:|---:|"]
 for fam in sorted(set(l["by_family"])|set(j["by_family"])):
  lv=l["by_family"].get(fam,{}).get("accuracy"); jv=j["by_family"].get(fam,{}).get("accuracy")
  lines.append(f"| {fam} | {lv:.3f} | {jv:.3f} |")
 lines += ["","## Interpretation constraints",
 "- BANKING77 is a 77-way intent-routing stress test, not a complete Agent benchmark.",
 "- Synthetic workflow cases probe action selection but should not be treated as external ground truth.",
 "- Hosted Jev latency includes network/service time; Laya latency here is GitHub-hosted CPU inference.",
 "- A later run should add option-order permutations, confidence calibration/ECE/Brier, N-way vs binary decomposition, multilingual slices, and batch throughput."]
 Path(a.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
