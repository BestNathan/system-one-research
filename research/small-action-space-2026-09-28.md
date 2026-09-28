# Small Action-Space Study — Laya vs TypeSafe Jev

Date: 2026-09-28  
Workflow run: https://github.com/BestNathan/system-one-research/actions/runs/36399437676  
Commit: `384b926f6ed0ea41b85b53a06e1b5acd3dc1194a`

## Question

How much of the observed Laya degradation in the 77-way BANKING77 benchmark is caused by action-space cardinality rather than by the underlying semantic difficulty?

## Experimental design

The experiment keeps the input examples fixed and changes only the available action set.

- Dataset: BANKING77 test split.
- 20 deliberately confusable intents drawn from four domains:
  - card payment
  - cash withdrawal
  - transfer
  - top up
- 5 examples per intent = 100 base examples.
- Each base example is evaluated with nested 3-way, 5-way, 10-way, and 20-way choice sets.
- The 5-way condition is also repeated with original, reversed, and rotated option ordering.
- Both Laya and Jev receive the same canonical `state/questions` representation.
- The existing 77-way 154-example benchmark is retained as an external reference; it is not the exact same 100-example subset.

## Accuracy vs action-space size

| Actions | Laya | Jev | Laya mean latency | Jev mean latency |
|---:|---:|---:|---:|---:|
| 3 | 88% | 98% | 442 ms | 155 ms |
| 5 | 78% | 95% | 546 ms | 161 ms |
| 10 | 73% | 94% | 801 ms | 156 ms |
| 20 | 66% | 92% | 991 ms | 159 ms |
| 77* | 42.2% | 76.0% | — | — |

`* 77-way uses the existing 154-case BANKING77 baseline.`

Within the controlled 3→20 sweep, Laya loses 22 percentage points while Jev loses 6 points.

## Paired cardinality effect

Because the exact same 100 examples appear in both the 3-way and 20-way conditions, the strongest comparison is paired:

### Laya

- correct at both 3-way and 20-way: 61
- correct at 3-way → wrong at 20-way: 27
- wrong at 3-way → correct at 20-way: 5
- wrong at both: 7

The discordant pair count is 27 vs 5. A two-sided exact binomial/McNemar-style test gives approximately `p = 1.13e-4`.

This is strong evidence that increasing the action set itself materially hurts Laya on the same semantic inputs.

### Jev

- correct at both: 92
- correct at 3-way → wrong at 20-way: 6
- wrong at 3-way → correct at 20-way: 0
- wrong at both: 2

Jev is also affected by cardinality, but much less strongly.

## Per-domain accuracy

### Laya

| Domain | 3-way | 5-way | 10-way | 20-way |
|---|---:|---:|---:|---:|
| card payment | 88% | 84% | 76% | 80% |
| cash withdrawal | 80% | 64% | 52% | 56% |
| transfer | 96% | 84% | 84% | 72% |
| top up | 88% | 80% | 80% | 56% |

The largest degradation occurs in cash-withdrawal and top-up intents. This suggests cardinality is not the only factor: semantic neighborhood density also matters.

### Jev

| Domain | 3-way | 5-way | 10-way | 20-way |
|---|---:|---:|---:|---:|
| card payment | 96% | 96% | 96% | 96% |
| cash withdrawal | 100% | 96% | 92% | 96% |
| transfer | 96% | 96% | 96% | 88% |
| top up | 100% | 92% | 92% | 88% |

Jev remains comparatively stable across domains and action counts.

## Confidence behavior

| Actions | Laya mean confidence | Laya acc @ conf>=.90 | Jev mean confidence | Jev acc @ conf>=.90 |
|---:|---:|---:|---:|---:|
| 3 | 73.6% | 100% | 97.0% | 97.8% |
| 5 | 72.0% | 100% | 95.9% | 97.7% |
| 10 | 83.1% | 91.8% | 94.5% | 98.8% |
| 20 | 88.5% | 78.8% | 94.9% | 97.5% |

The important Laya failure mode is that confidence increases while accuracy decreases.

For incorrect Laya predictions, mean confidence rises from about 32% at 3-way to about 79% at 20-way. At 20-way, 14 of the 34 wrong predictions have confidence >= 0.90.

This means raw Laya confidence is useful in very small spaces but becomes increasingly unsafe as a generic autonomous execution gate when the action space expands.

## Option-order sensitivity

The same 5-way semantic choice was evaluated under original, reversed, and rotated option orders.

| Model | Original | Reversed | Rotated | Base examples whose prediction changed |
|---|---:|---:|---:|---:|
| Laya | 80% | 83% | 81% | 15% |
| Jev | 95% | 96% | 96% | 1% |

Aggregate Laya accuracy moves only a few points, but 15% of individual decisions change when only option order changes. The cash-withdrawal domain is most sensitive at 24%.

This matters for an Agent Runtime because action ordering should normally be an implementation detail rather than part of the semantic state.

## Model-to-model paired comparison

On the same 100 examples:

| Actions | both correct | Jev correct / Laya wrong | Laya correct / Jev wrong | both wrong |
|---:|---:|---:|---:|---:|
| 3 | 86 | 12 | 2 | 0 |
| 5 | 77 | 18 | 1 | 4 |
| 10 | 72 | 22 | 1 | 5 |
| 20 | 64 | 28 | 2 | 6 |

The gap widens as the action set grows.

## Interpretation

The experiment supports three conclusions.

1. **Laya is viable in genuinely small action spaces.**  
   Its 3-way accuracy rises to 88%, far above the 42.2% observed in the 77-way benchmark.

2. **Action-space cardinality is a real causal pressure.**  
   On identical examples, Laya degrades from 88% at 3-way to 66% at 20-way, with 27 examples changing from correct to incorrect and only 5 moving the opposite direction.

3. **Runtime action-space design should be treated as part of the model architecture.**  
   For Laya especially, exposing all actions at once is materially worse than constraining the decision to a small local set. This supports hierarchical or progressively disclosed action spaces:
   
   ```text
   state
     -> small available action set
     -> System 1 decision
     -> new state
     -> disclose next local action set
     -> System 1 decision
   ```

Jev is much less sensitive to action count and ordering, so it is a stronger zero-shot general decision service in the tested range. Laya becomes more competitive when the runtime can guarantee a small, semantically local action set.

## Next experiment

The next useful test is not another unrelated dataset. It should reuse these same 100 examples and compare:

1. flat 20-way choice,
2. two-level hierarchy (4 domain actions -> 5 local intent actions),
3. progressive disclosure where the second-stage actions are generated from the first-stage state,
4. an oracle hierarchy where the correct coarse domain is supplied, to separate routing error from local classification error.

This would directly test whether a state-machine runtime can recover Laya's small-space performance while still covering a large global action space.
