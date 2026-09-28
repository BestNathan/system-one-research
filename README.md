# System One Research

Reproducible experiments for comparing fast typed-decision / System-1 model interfaces.

## Laya vs TypeSafe Jev

The first experiment intentionally separates **decision quality** from **systems performance**.

### Research questions

1. Given byte-identical state, instructions and options, how often do Laya and Jev make the same correct typed decision?
2. How calibrated are returned probabilities/confidences, especially on ambiguous cases?
3. How sensitive is each model to option order and to decomposing one multi-way choice into multiple binary decisions?
4. What are the deployment trade-offs: cold start, p50/p95 latency, batch throughput, cost, memory and local/offline capability?
5. Do results change materially by language, especially Chinese vs English?

### Experimental design

**Track A — quality**

- identical cases and typed schemas for both backends
- choice accuracy
- calibration: Brier score / ECE once the dataset is large enough
- confidence-vs-correctness curves
- option-order perturbation
- decision-decomposition ablation: one N-way choice vs N binary/yes-no decisions
- English + Chinese slices
- raw response archive for every case

**Track B — systems**

- cold model/load time
- warm single-request p50/p95
- batch throughput at 1/8/32/128
- peak RSS / GPU memory where measurable
- API usage/cost for Jev
- local/offline capability and model artifact size for Laya

Do **not** collapse Track A and Track B into one overall score: local Laya latency and hosted Jev latency are different deployment paths.

### Why rerun instead of quoting Laya's published Jev comparison?

Laya's repository contains useful existing comparisons, but its benchmark notes explicitly say some Jev headline figures are third-party/published rather than measured in the same run. Our experiment therefore treats those as prior evidence, not as the result.

### GitHub Actions

Run **Actions → system-one benchmark → Run workflow**.

- `smoke`: fastest validation.
- `core`: all currently committed cases.
- `feishu`: Chinese workflow slice; this is currently only a seed. The next phase should vendor the upstream frozen 64-case fixture with its provenance/license files unchanged.
- `run_jev=true`: adds the Jev job. Configure repository secret `TYPESAFE_API_KEY` first.
- Optional variable via environment: `JEV_MODEL` (defaults to `jev-1.13.0`).

Artifacts contain `raw.jsonl` and `summary.json`.

## Next experiment expansion

The next commit should grow the corpus to at least a few hundred held-out decisions and add:

- a vendored, hash-pinned copy of the Laya Feishu diagnostic;
- a synthetic workflow suite modeled after routing, moderation, prioritization, extraction/verification and tool/action selection;
- deterministic option permutations;
- ECE/Brier implementation with fixed bins;
- three warm repetitions per case;
- batch-size sweep and memory measurement;
- report generation that compares both artifacts only when they used the exact same dataset hash.
