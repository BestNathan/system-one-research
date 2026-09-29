# Jev vs DashScope on a 100-case typed-decisions sample

Date: 2026-09-29

## Goal

Follow up the compact BANKING77 comparison with a structurally different System One benchmark while keeping paid API usage small.

The earlier BANKING77 slice used one 77-way choice per request. This experiment instead tests shared-state, multi-question typed decisions.

## Dataset and sampling

Source: `LocalLLaMA/typed-decisions`, pinned test revision:

`f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`

The full test split has 400 cases. This experiment deterministically selects the first 25 cases from each of the four workflow families:

- agent_trace_observability: 25
- customer_service: 25
- invoice_processing: 25
- security_incidents: 25

Total:

- 100 cases
- 5 typed questions per case
- 500 paired decisions
- 100 Jev API calls
- 100 DashScope API calls

Both providers receive identical `state` and `questions` objects.

## Models

- TypeSafe: `jev-1.13.0`
- DashScope: `decision-model-preview`

Both completed all 100 cases without errors.

## Main result

| Metric | TypeSafe Jev | DashScope decision-model-preview |
|---|---:|---:|
| Cases OK | 100/100 | 100/100 |
| Decisions | 500 | 500 |
| Accuracy | **72.0%** | 60.4% |
| Soft accuracy | **0.536** | 0.483 |
| KL from teacher ↓ | 1.551 | **0.952** |
| TV ↓ | **0.253** | 0.295 |
| Hard-label Brier ↓ | **0.387** | 0.562 |
| Soft-distribution Brier ↓ | **0.152** | 0.227 |
| ECE ↓ | **0.077** | 0.137 |
| Score MAE ↓ | **0.394** | 0.463 |
| Within one score level | **93.5%** | 86.5% |
| Case p50 latency | **181 ms** | 290 ms |
| Case p95 latency | **299 ms** | 315 ms |

Jev leads hard accuracy by **11.6 percentage points**.

A paired bootstrap over the 500 decisions gives an approximate 95% interval for the Jev-minus-DashScope accuracy gap of:

`+6.6 pp to +16.8 pp`

## Paired hard decisions

| Outcome | Decisions |
|---|---:|
| both correct | 241 |
| Jev correct / DashScope wrong | **119** |
| DashScope correct / Jev wrong | **61** |
| both wrong | 79 |

The exact two-sided discordant-pair test gives:

`p ≈ 1.84e-5`

The two providers emitted different hard labels on 213/500 decisions.

Unlike the 77-case BANKING77 result, the advantage here is statistically strong for this sampled benchmark.

## Whole-case exact match

A case is exact only when all five typed questions are correct.

| Outcome | Result |
|---|---:|
| Jev exact-case accuracy | **23%** |
| DashScope exact-case accuracy | 7% |
| both exact | 1 |
| Jev-only exact | 22 |
| DashScope-only exact | 6 |
| neither exact | 71 |

Paired exact-case test:

`p ≈ 0.00372`

Multi-decision compounding therefore widens the practical difference: a moderate per-decision accuracy gap becomes a much larger gap in the probability that an entire five-decision state transition is correct.

## By question type

| Type | Jev accuracy | DashScope accuracy | Jev Brier ↓ | DashScope Brier ↓ |
|---|---:|---:|---:|---:|
| choice | **70.0%** | 60.0% | **0.374** | 0.583 |
| noul | **75.3%** | 73.3% | **0.288** | 0.414 |
| score | **71.0%** | 51.0% | **0.470** | 0.659 |

The largest structural gap is `score`: **20 percentage points**.

This matters for an Agent Runtime because scores are often used for:

- severity
- urgency
- risk
- escalation thresholds
- prioritization

A backend can therefore be competitive on binary decisions but still behave very differently when the runtime asks for an ordinal control signal.

## By workflow

| Workflow | Jev accuracy | DashScope accuracy |
|---|---:|---:|
| agent_trace_observability | 60.0% | 60.0% |
| customer_service | **78.4%** | 68.0% |
| invoice_processing | **80.8%** | 52.8% |
| security_incidents | **68.8%** | 60.8% |

The overall difference is dominated by **invoice_processing**, where Jev leads by 28 percentage points.

### Invoice-processing failure pattern

The most divergent fields are:

| Question | Type | Jev | DashScope |
|---|---|---:|---:|
| matches_order | noul | **100%** | 56% |
| discrepancy_severity | score | **76%** | 40% |
| urgency | score | **64%** | 28% |
| disposition | choice | **72%** | 52% |
| duplicate | noul | **92%** | 88% |

DashScope shows a strong output-collapse pattern on the two invoice score heads:

- invoice `urgency`: it predicts level 1 for 25/25 sampled cases; gold is mostly levels 1–3
- invoice `discrepancy_severity`: it predicts level 0 for 24/25 sampled cases

For `matches_order`, DashScope also has a strong positive bias:

- all 12 gold-true cases are classified true
- 11/13 gold-false cases are also classified true

This is not a generic protocol parsing issue; the response contains valid probabilities and score legends. It is a domain/head-specific decision behavior.

## Where DashScope is stronger

DashScope is not uniformly weaker.

Two especially strong reversals are in agent-trace observability:

| Question | Type | Jev | DashScope |
|---|---|---:|---:|
| needs_review | noul | 48% | **88%** |
| action | choice | 52% | **68%** |

Customer-service `needs_human` also favors DashScope:

- Jev: 56%
- DashScope: 72%

This reinforces that provider quality is conditional on the local decision head and domain.

## The KL anomaly

DashScope has **lower KL from the soft teacher distribution** despite being worse on accuracy, TV, both Brier metrics, ECE, and score MAE:

- Jev KL: 1.551
- DashScope KL: 0.952

This is not contradictory.

KL is asymmetric and heavily penalizes assigning near-zero probability to outcomes that retain non-zero teacher probability. Jev is the sharper distribution:

- Jev mean top-choice probability: about 75.5%
- DashScope mean top-choice probability: about 73.1%
- Jev mean predictive entropy: about 0.550 nats
- DashScope mean predictive entropy: about 0.635 nats

DashScope's smoother probability mass can therefore reduce KL while still placing too much mass on the wrong region of the distribution.

For System One evaluation, **KL should not be interpreted alone**. TV, Brier, ECE, hard accuracy, and task-specific score error provide the more complete picture.

## Comparison with the BANKING77 result

The two experiments point in opposite directions.

### Compact BANKING77 — 77 cases, 77-way choice

- Jev: 76.6%
- DashScope: **84.4%**
- DashScope had better Brier/ECE
- Jev had lower latency

### typed-decisions — 100 cases, 500 decisions

- Jev: **72.0%**
- DashScope: 60.4%
- Jev has better Brier/ECE/TV/score MAE
- Jev also has lower latency

This is an important result:

> There is no evidence here for a single globally superior System One provider. Relative quality changes substantially with domain, question type, and decision-head semantics.

BANKING77 tests one large categorical action space.

typed-decisions tests:

`shared state -> several heterogeneous decision heads`

including binary and ordinal decisions.

Those are materially different System One workloads.

## Runtime implication

The backend abstraction should therefore not collapse provider selection to one global benchmark score.

A stronger design is:

```text
state
  -> decision schema / domain / local action space
  -> provider policy
       -> Jev
       -> DashScope
       -> local Laya specialist
  -> calibrated decision
  -> optional escalation to System 2
```

Provider policy can eventually condition on:

- question type
- action cardinality
- semantic density
- domain/workflow
- language
- latency budget
- confidence calibration
- cost
- locality/privacy

The current experiments suggest that **routing the System One backend itself may be part of the runtime scheduler**.

## Caveats

- This is a 100-case subset, not the full 400-case test split.
- Sampling is deterministic and stratified, but uses the first 25 cases per workflow rather than seeded random sampling.
- The dataset is synthetic and teacher-labelled.
- The test split is held out from the training split, but uses the same four workflow families.
- Provider endpoints and model previews can change.
- Hosted latency includes network and regional effects.

## Reproduction

Workflow:

`.github/workflows/jev-vs-dashscope-typed100.yml`

The workflow is manual-only after the initial experiment to prevent accidental paid reruns.

Run:

`36568503884`

Artifact:

`jev-vs-dashscope-typed100`
