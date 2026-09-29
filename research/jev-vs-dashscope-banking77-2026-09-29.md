# Jev vs DashScope decision-model-preview on BANKING77

Date: 2026-09-29

## Question

How does Alibaba Cloud Model Studio `decision-model-preview` compare with TypeSafe Jev when both receive the exact same System One request shape?

This is intentionally a **small, low-cost paired experiment**. It is not intended to establish a universal ranking.

## Protocol compatibility

DashScope exposes a TypeSafe System One-compatible endpoint:

- `POST /compatible-mode/v1/systemone`
- `state`
- typed `questions`
- `choice / noul / score`
- probability distributions and confidence

The experiment therefore uses the same canonical `state + questions` payload for both hosted providers.

## Dataset

BANKING77 test split.

To minimize paid calls while still covering the entire intent taxonomy:

- 77 intents
- exactly 1 test example per intent
- 77 cases total
- every case is a full **77-way choice**
- Jev calls: 77
- DashScope calls: 77
- total provider calls: 154

The BANKING77 option labels are kept exactly as defined in the official dataset metadata, including historical names such as `Refund_not_showing_up` and `reverted_card_payment?`.

## Models

- TypeSafe: `jev-1.13.0`
- DashScope: `decision-model-preview`

DashScope endpoint used:

`https://llm-cnwlvwjfb3z4aa29.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone`

Both completed all 77 cases without errors.

## Main result

| Metric | TypeSafe Jev | DashScope decision-model-preview |
|---|---:|---:|
| Cases OK | 77/77 | 77/77 |
| Accuracy | 76.6% | **84.4%** |
| Brier ↓ | 0.328 | **0.200** |
| ECE ↓ | 0.106 | **0.055** |
| Mean confidence | 87.0% | 85.9% |
| Confidence when correct | 93.4% | 91.3% |
| Confidence when wrong | 66.2% | **56.5%** |
| Mean latency | **170 ms** | 376 ms |
| p50 latency | **156 ms** | 363 ms |
| p95 latency | **250 ms** | 419 ms |

DashScope is +7.8 percentage points on accuracy in this slice.

Its mean Brier score is about 39% lower than Jev's, and its ECE is roughly half. The confidence gap between correct and incorrect decisions is also larger:

- Jev: about 27.1 pp
- DashScope: about 34.8 pp

This suggests better confidence separation on this specific benchmark.

## Paired correctness

The important comparison is paired because both systems see the same cases.

| Outcome | Cases |
|---|---:|
| both correct | 56 |
| Jev correct / DashScope wrong | 3 |
| DashScope correct / Jev wrong | 9 |
| both wrong | 9 |

So among cases where only one provider is correct:

`DashScope : Jev = 9 : 3`

The two-sided exact discordant-pair test gives approximately:

`p = 0.146`

A paired bootstrap for the accuracy difference gives a rough 95% interval of approximately:

`-1.3 pp to +16.9 pp`

Therefore the experiment shows a **positive signal for DashScope**, but 77 cases are not enough to claim a statistically robust accuracy advantage.

## Where DashScope recovered Jev errors

DashScope was correct while Jev was wrong on 9 cases, including:

- `balance_not_updated_after_bank_transfer`
- `cancel_transfer`
- `country_support`
- `disposable_card_limits`
- `order_physical_card`
- `pending_top_up`
- `reverted_card_payment?`
- `top_up_reverted`
- `virtual_card_not_working`

Several of these are fine-grained near-neighbor distinctions, which is notable because BANKING77 is intentionally difficult due to semantic overlap between intent labels.

## Where Jev recovered DashScope errors

Jev was correct while DashScope was wrong on 3 cases:

- `compromised_card`
- `contactless_not_working`
- `top_up_failed`

## Shared failures

Both systems failed the same 9 gold intents:

- `beneficiary_not_allowed`
- `card_about_to_expire`
- `card_arrival`
- `card_delivery_estimate`
- `fiat_currency_support`
- `get_physical_card`
- `top_up_by_bank_transfer_charge`
- `transfer_not_received_by_recipient`
- `wrong_exchange_rate_for_cash_withdrawal`

These are useful candidates for future semantic-neighborhood experiments because both models appear to struggle with nearby alternatives.

## Latency

Jev is substantially faster on this execution path:

- Jev mean: ~170 ms
- DashScope mean: ~376 ms
- DashScope / Jev mean-latency ratio: ~2.21x

This is **end-to-end service latency from GitHub-hosted runners**, not pure model compute.

The providers are reached through different infrastructure and regions, so latency should be interpreted as deployment-path performance rather than an intrinsic model-speed comparison.

## Interpretation

This small experiment supports four observations.

### 1. DashScope is genuinely System One protocol-compatible

No adapter-specific semantic transformation was required. The same state/question representation used for Jev worked directly against `decision-model-preview`.

That means the Agent Runtime can treat both as interchangeable remote System One backends behind the same interface.

### 2. DashScope shows a strong quality signal on large action spaces

BANKING77 is a difficult 77-way classification problem.

On the 77-case balanced slice:

- Jev: 76.6%
- DashScope: 84.4%

This is particularly interesting given the earlier Laya experiments, where large action-space cardinality strongly degraded the local base model.

### 3. DashScope's calibration signal is stronger in this slice

DashScope has:

- lower Brier
- lower ECE
- lower confidence on wrong decisions
- larger correct-vs-wrong confidence separation

For a System 1 runtime, this may matter as much as raw accuracy because confidence is often used for escalation, retry, or System 2 routing.

### 4. Jev currently has the latency advantage

For workloads where a decision must be made synchronously and very quickly, Jev's hosted path was around 2.2x faster in this run.

## Caveats

This experiment deliberately trades statistical power for low evaluation cost.

- Only one example per BANKING77 intent is used.
- BANKING77 is English-only.
- Every case exposes 77 actions; this does not characterize small local action spaces.
- Accuracy difference is not statistically decisive at this sample size.
- Latency includes network and region effects.
- Provider token/usage accounting is not directly comparable: the services report different usage fields/tokenization semantics.
- Results apply to the tested model versions and endpoints at the time of the run.

## Current conclusion

The strongest defensible conclusion is:

> On this compact, balanced 77-case BANKING77 test, DashScope `decision-model-preview` produced higher hard accuracy and better calibration metrics than Jev 1.13.0, while Jev had substantially lower end-to-end latency. The paired accuracy difference is directionally meaningful but not statistically decisive at this sample size.

For runtime architecture, the more important result is that both providers can sit behind the same System One abstraction, which makes backend selection a policy decision based on quality, calibration, latency, cost, locality, and language/domain requirements.

## Reproduction

Workflow:

`.github/workflows/jev-vs-dashscope-banking77.yml`

The workflow is manual-only to avoid accidental paid reruns.

Run:

`36565177329`

Artifact:

`jev-vs-dashscope-banking77`
