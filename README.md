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

The GitHub Actions workflow is:

    .github/workflows/system-one-benchmark.yml

It runs:

1. the 77-way BANKING77 baseline
2. the controlled small action-space dataset generation
3. Laya baseline and small-action benchmarks
4. TypeSafe Jev baseline and small-action benchmarks
5. report generation
6. artifact upload

The benchmark job is associated with the GitHub Environment **typesafe** and reads **secrets.TYPESAFE_API_KEY** from that environment.

Artifacts include the frozen/generated cases, raw.jsonl, summary.json, REPORT.md, and SMALL_ACTION_REPORT.md.

Latest successful small-action run:

- [GitHub Actions run #36399437676](https://github.com/BestNathan/system-one-research/actions/runs/36399437676)
- [Detailed small action-space research report](research/small-action-space-2026-09-28.md)

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
