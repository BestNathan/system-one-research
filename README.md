# System One Research

Reproducible experiments for understanding how **System 1 decision models** behave inside an Agent Runtime.

The current study compares:

- **Laya** — local typed-decision model
- **TypeSafe Jev** — hosted System 1 decision service

The goal is not only to ask which model is more accurate. The more important question is:

> How should a runtime expose state and available actions so that a fast System 1 model can make reliable decisions?

## Current findings

The experiments so far point to a strong architectural result:

> **System 1 quality depends materially on the size and semantic density of the action space. A large global action space should be progressively reduced to small local action sets instead of being exposed to the decision model all at once.**

Laya is especially sensitive to this effect. Jev is substantially more robust to large candidate sets and option ordering.

## Experiment 1 — 77-way BANKING77 baseline

The first benchmark uses 154 BANKING77 examples plus 4 synthetic workflow-routing cases. Both backends receive the same canonical state/questions representation.

| Metric | Laya | TypeSafe Jev |
|---|---:|---:|
| Cases completed | 158/158 | 158/158 |
| Overall accuracy | 42.4% | 76.6% |
| BANKING77 77-way accuracy | 42.2% | 76.0% |
| Workflow-routing accuracy | 50.0% | 100.0% |
| Mean latency | 1675 ms | 157 ms |
| p50 latency | 1675 ms | 153 ms |
| p95 latency | 1842 ms | 193 ms |

Latency is **not a hardware-equivalent comparison**: Laya runs locally on the GitHub-hosted CPU runner while Jev is a hosted API.

The 77-way result showed that Laya's prediction distribution collapses toward a smaller subset of intents and that high confidence does not necessarily imply correctness in a large action space.

## Experiment 2 — controlled small action spaces

To separate semantic difficulty from action-space cardinality, the second experiment keeps the **same input examples** while changing only the available candidate set.

Design:

- BANKING77 test split
- 20 deliberately confusable intents
- four domains: card payment, cash withdrawal, transfer, and top up
- 5 examples per intent = **100 base examples**
- each example is evaluated with **3 / 5 / 10 / 20** available actions
- the same 5-way choice is also tested under original, reversed, and rotated option order

### Accuracy vs action-space size

| Actions | Laya | TypeSafe Jev | Laya mean latency | Jev mean latency |
|---:|---:|---:|---:|---:|
| 3 | **88%** | **98%** | 442 ms | 155 ms |
| 5 | 78% | 95% | 546 ms | 161 ms |
| 10 | 73% | 94% | 801 ms | 156 ms |
| 20 | **66%** | **92%** | 991 ms | 159 ms |
| 77* | 42.2% | 76.0% | — | — |

*77-way is the earlier 154-example baseline and is not the exact same 100-example subset.*

For the controlled 3→20 sweep:

- Laya loses **22 percentage points**
- Jev loses **6 percentage points**

### Paired cardinality effect

Because the same 100 examples appear in both the 3-way and 20-way conditions, we can inspect individual transitions.

#### Laya

| Transition | Cases |
|---|---:|
| correct → correct | 61 |
| **correct → wrong** | **27** |
| wrong → correct | 5 |
| wrong → wrong | 7 |

The discordant pair count is **27 vs 5**. A two-sided exact paired test gives approximately p = 1.13e-4.

This is evidence that increasing the action set itself materially hurts Laya on otherwise identical semantic inputs.

#### TypeSafe Jev

| Transition | Cases |
|---|---:|
| correct → correct | 92 |
| correct → wrong | 6 |
| wrong → correct | 0 |
| wrong → wrong | 2 |

Jev is also affected by cardinality, but much less strongly.

## Confidence behavior

Laya shows an important failure mode: **confidence increases while accuracy decreases** as more actions are exposed.

| Actions | Laya accuracy | Laya mean confidence | Laya accuracy when confidence ≥ .90 |
|---:|---:|---:|---:|
| 3 | 88% | 73.6% | 100.0% |
| 5 | 78% | 72.0% | 100.0% |
| 10 | 73% | 83.1% | 91.8% |
| 20 | 66% | 88.5% | 78.8% |

At 20-way, 14 of Laya's 34 incorrect decisions still have confidence ≥ 0.90.

Therefore raw confidence should not be used as a universal autonomous execution gate without conditioning on the action-space regime and calibration profile.

Jev remains much better calibrated in the tested range: its accuracy among decisions with confidence ≥ .90 stays around 97–99%.

## Option-order sensitivity

The exact same 5 semantic options were presented in original, reversed, and rotated order.

| Model | Original | Reversed | Rotated | Predictions changed across orderings |
|---|---:|---:|---:|---:|
| Laya | 80% | 83% | 81% | **15%** |
| TypeSafe Jev | 95% | 96% | 96% | **1%** |

This matters for runtime design: action ordering should normally be an implementation detail, not part of the semantic state.

## Experiment 3 — Jev vs DashScope on compact BANKING77

Alibaba Cloud Model Studio exposes `decision-model-preview` through a TypeSafe System One-compatible endpoint, so Jev and DashScope can be tested with the exact same `state + questions` payload.

To keep paid evaluation small, this experiment uses **77 BANKING77 cases: exactly one example per intent**, while still exposing the complete 77-way action space on every request.

| Metric | TypeSafe Jev | DashScope decision-model-preview |
|---|---:|---:|
| Cases | 77/77 | 77/77 |
| Accuracy | 76.6% | **84.4%** |
| Brier ↓ | 0.328 | **0.200** |
| ECE ↓ | 0.106 | **0.055** |
| Wrong-decision mean confidence | 66.2% | **56.5%** |
| Mean latency | **170 ms** | 376 ms |
| p50 latency | **156 ms** | 363 ms |
| p95 latency | **250 ms** | 419 ms |

Paired correctness:

- both correct: 56
- Jev correct / DashScope wrong: 3
- DashScope correct / Jev wrong: 9
- both wrong: 9
- exact two-sided discordant-pair test: `p ≈ 0.146`

DashScope is +7.8 percentage points on this slice and also shows materially better Brier/ECE calibration, but the 77-case sample is intentionally small: a paired bootstrap for the accuracy difference spans roughly **-1.3 pp to +16.9 pp**. The result is therefore a positive signal, not a statistically decisive model ranking.

Jev is about **2.2× faster** on the measured hosted path. This includes network and regional routing, so it should not be interpreted as pure model-compute latency.

The architectural result is also important: **Jev and DashScope can sit behind the same System One backend abstraction without changing the decision state/question schema.** Backend selection can therefore be a runtime policy over quality, calibration, latency, cost, locality, language, and domain.

- [Detailed Jev vs DashScope research report](research/jev-vs-dashscope-banking77-2026-09-29.md)
- [GitHub Actions run #36565177329](https://github.com/BestNathan/system-one-research/actions/runs/36565177329)

## Experiment 4 — Jev vs DashScope on 100 typed-decision cases

To test whether the BANKING77 result generalizes, a second hosted-provider experiment samples **100 cases** from the official `LocalLLaMA/typed-decisions` test split:

- 25 agent-trace observability cases
- 25 customer-service cases
- 25 invoice-processing cases
- 25 security-incident cases
- 5 typed questions per case
- **500 paired decisions**
- 100 API calls per provider

The result reverses the BANKING77 ordering:

| Metric | TypeSafe Jev | DashScope decision-model-preview |
|---|---:|---:|
| Accuracy | **72.0%** | 60.4% |
| Soft accuracy | **0.536** | 0.483 |
| TV ↓ | **0.253** | 0.295 |
| Hard-label Brier ↓ | **0.387** | 0.562 |
| Soft-distribution Brier ↓ | **0.152** | 0.227 |
| ECE ↓ | **0.077** | 0.137 |
| Score MAE ↓ | **0.394** | 0.463 |
| Within one score level | **93.5%** | 86.5% |
| Case p50 latency | **181 ms** | 290 ms |

Paired decisions:

- both correct: 241
- Jev correct / DashScope wrong: **119**
- DashScope correct / Jev wrong: **61**
- both wrong: 79
- exact two-sided discordant-pair test: **p ≈ 1.84e-5**
- paired bootstrap for the Jev-minus-DashScope accuracy gap: roughly **+6.6 pp to +16.8 pp**

The largest gap is on `score` questions: **71% vs 51%**. The strongest domain effect is invoice processing: **80.8% vs 52.8%**. DashScope is not uniformly weaker, however; for example, on agent-trace `needs_review` it reaches **88% vs Jev 48%**.

Together with Experiment 3, this is evidence against treating provider quality as a single global scalar:

- **BANKING77 77-way choice:** DashScope 84.4%, Jev 76.6%
- **typed-decisions multi-head:** Jev 72.0%, DashScope 60.4%

Provider selection may therefore need to depend on **domain, question type, action-space structure, calibration, latency, and cost**, not just one aggregate benchmark.

- [Detailed typed100 research report](research/jev-vs-dashscope-typed100-2026-09-29.md)
- [GitHub Actions run #36568503884](https://github.com/BestNathan/system-one-research/actions/runs/36568503884)

## Runtime implication

The results suggest that the runtime should avoid exposing a huge global action set directly to System 1.

~~~text
global state
  -> choice(1000 actions)
  -> System 1
~~~

Instead, expose a progressively disclosed local action space:

~~~text
state
  -> small available action set
  -> System 1 decision
  -> state transition
  -> disclose next local action set
  -> System 1 decision
~~~

This makes **action-space construction part of the Agent Runtime architecture**, rather than treating it as a passive list passed into the model.

A useful way to think about decision difficulty is:

~~~text
decision difficulty
  = action-space cardinality
  + semantic density between candidate actions
~~~

Laya currently looks best suited to a runtime that can guarantee small, semantically local action sets. Jev behaves more like a general zero-shot decision service that tolerates larger candidate spaces.

## Reproduce

### BANKING77 / action-space research

`.github/workflows/system-one-benchmark.yml` runs the 77-way baseline and controlled 3/5/10/20-way experiments, then uploads the raw decisions and generated reports.

Latest successful small-action run:

- [GitHub Actions run #36399437676](https://github.com/BestNathan/system-one-research/actions/runs/36399437676)
- [Detailed small action-space research report](research/small-action-space-2026-09-28.md)

### Official typed-decisions replication

Typed-decisions uses two CI tiers.

**Push smoke:** `.github/workflows/typed-decisions-smoke.yml`

- triggered by changes to the typed-decisions experiment code
- samples one case from each of the four workflow families
- 4 cases / 20 typed decisions per backend
- runs base Laya, `laya-typed-decisions`, and TypeSafe Jev
- validates zero errors/dropped questions plus accuracy/Brier/ECE output
- uses CPU-only PyTorch and a shared Hugging Face checkpoint cache
- uses concurrency cancellation so only the newest smoke run survives

Latest validated smoke run:

- [GitHub Actions run #36418084288](https://github.com/BestNathan/system-one-research/actions/runs/36418084288)

**Manual full benchmark:** `.github/workflows/typed-decisions-benchmark.yml`

- only `workflow_dispatch`; normal pushes never launch the 400-case benchmark
- evaluates the pinned official 400-case / 2,000-decision test split
- runs base Laya, typed Laya, and Jev as three parallel jobs
- merges their artifacts in a final report job
- shares the checkpoint cache populated by smoke runs
- full runs use `cancel-in-progress` so a replacement run cannot stack on top of an older one

The Jev jobs are associated with the GitHub Environment **typesafe** and read **secrets.TYPESAFE_API_KEY** from that environment.

The current GitHub CPU runner is suitable for smoke validation but is not a latency-comparable inference environment: in the successful 4-case smoke, each Laya checkpoint required about 33.7 seconds of batched scoring, while Jev is a hosted API. Full Laya jobs are therefore parallelized and latency results must be interpreted separately from model quality.

## Next experiment

The next experiment should reuse the same 100 BANKING77 examples and compare:

1. **flat 20-way choice**
2. **two-level hierarchy:** 4 domain actions → 5 local intent actions
3. **progressive disclosure:** each state transition reveals the next local action set
4. **oracle hierarchy:** provide the correct coarse domain and measure only local decision quality

The oracle condition is important because it separates coarse routing error from local System 1 decision error.

If hierarchical/progressive execution recovers the small-space accuracy while preserving global coverage, it would provide direct evidence for a state-machine-style System 1 Runtime.

## Research principles

- Keep model inputs and typed schemas identical when comparing backends.
- Prefer paired experiments where the same semantic input is evaluated under controlled changes.
- Report quality, calibration, and systems characteristics separately.
- Do not collapse local-model latency and hosted-service latency into a single score.
- Preserve raw model responses for post-hoc analysis.
- Treat model confidence as an empirical quantity that must be calibrated, not as an intrinsic guarantee.
