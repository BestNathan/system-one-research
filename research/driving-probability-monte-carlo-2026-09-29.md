# Monte Carlo System One driving worldlines

Date: 2026-09-29

Workflow run: https://github.com/BestNathan/system-one-research/actions/runs/36597655294

## Question

The earlier single-trajectory experiment showed that Jev and DashScope both move their full action probability distribution when physical state, driver profile, or recent history changes.

This experiment asks the next question:

> If each model is allowed to generate and sample its own probability field over many closed-loop trajectories, what distribution of worlds emerges?

The goal is not to rank the models as driving controllers. Neither model is trained or validated as a production autonomous-driving policy. The goal is to study how a System One probability field compounds through repeated state transitions.

## Design

Each provider runs its own closed-loop simulator:

~~~text
state_t
  -> provider-specific System One
  -> P(action_t | state_t)
  -> sample action_t
  -> toy physics
  -> state_t+1
  -> repeat
~~~

Configuration:

- 1,000 worldlines per provider
- 16 steps per worldline
- 16,000 System One calls per provider
- 32,000 total calls
- 8 concurrent trajectories per provider
- same initial state
- same action space
- same exogenous schedule
- slower vehicle cuts in at step 4
- lane clears at step 8

The action space remains:

- hard_brake
- brake
- keep_speed
- accelerate
- hard_accelerate

### Separate worlds, matched random numbers

Jev and DashScope do **not** share world state.

A Jev trajectory advances only from Jev probabilities and Jev-sampled actions. A DashScope trajectory advances only from DashScope probabilities and DashScope-sampled actions.

For comparison, worldline ID N uses the same PRNG seed for both providers. This is a matched-random-number design. The random draw sequence is paired, but the probability distributions and resulting states are provider-specific.

Using matched random numbers does not change either provider's marginal trajectory distribution; it reduces sampling noise in paired comparisons.

## Completion and usage

Both providers completed all 1,000 worldlines with zero failed trajectories.

| Metric | Jev | DashScope |
|---|---:|---:|
| Worldlines | 1,000 | 1,000 |
| Steps/worldline | 16 | 16 |
| Calls | 16,000 | 16,000 |
| Input tokens | 11,223,737 | 5,300,089 |
| Output tokens | 1,008,245 | 0 |
| Mean latency | 89.7 ms | 483.1 ms |
| p50 latency | 83.8 ms | 319.5 ms |
| p95 latency | 135.2 ms | 1,039.8 ms |

## Main worldline result

The two probability fields generate very different trajectory distributions.

| Outcome | Jev | DashScope |
|---|---:|---:|
| Near miss (<7m) | 70.8% | 88.6% |
| Critical gap (<3m) | 50.1% | 84.0% |
| Hit 1m simulator floor | 38.6% | 81.5% |
| Mean minimum gap | 4.814m | 2.910m |
| Median minimum gap | 2.682m | 1.000m |
| Mean final speed | 46.845 mph | 20.576 mph |
| Mean maximum speed | 55.022 mph | 54.186 mph |
| Hard brakes/worldline | 1.716 | 3.306 |
| Hard accelerations/worldline | 5.016 | 3.067 |
| Non-argmax samples/worldline | 6.074 | 7.710 |

These are outcomes in this toy environment only. They are not real-world safety measurements.

## Paired worldlines

Because worldline IDs use matched random draws, the paired outcome table isolates how strongly the two probability fields alter the resulting world.

### Near miss

| Outcome | Worldlines |
|---|---:|
| both | 676 |
| Jev only | 32 |
| DashScope only | 210 |
| neither | 82 |

The discordant split is 32 vs 210.

### Critical gap

| Outcome | Worldlines |
|---|---:|
| both | 479 |
| Jev only | 22 |
| DashScope only | 361 |
| neither | 138 |

### Hit 1m simulator floor

| Outcome | Worldlines |
|---|---:|
| both | 364 |
| Jev only | 22 |
| DashScope only | 451 |
| neither | 163 |

The paired differences are extremely large relative to Monte Carlo noise. This does **not** establish that one provider is a generally safer model; it establishes that their zero-shot conditional probability fields produce materially different world-state distributions under the same toy dynamics.

## What happens around the cut-in

The most informative part is the event window.

| Step | Event | Jev brake | DashScope brake | Jev accelerate | DashScope accelerate | Jev mean gap | DashScope mean gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | pre-event | 0.0% | 2.8% | 100.0% | 84.2% | - | - |
| 4 | cut-in | 65.7% | 9.4% | 33.1% | 74.8% | 10.849m | 11.673m |
| 5 | +1s | 82.2% | 11.3% | 16.6% | 73.7% | 6.385m | 5.757m |
| 6 | +2s | 88.5% | 34.7% | 11.1% | 53.3% | 5.386m | 4.079m |
| 7 | +3s | 89.0% | 53.8% | 9.9% | 37.6% | 6.403m | 3.832m |
| 8 | lane clear | 15.9% | 46.2% | 83.1% | 48.1% | - | - |
| 9 | +1s clear | 16.2% | 56.1% | 82.7% | 38.4% | - | - |

This exposes two distinct response shapes.

### Jev: fast hazard response, fast recovery

Before the cut-in, Jev is almost entirely in the acceleration modes.

At the cut-in:

- braking jumps to 65.7%
- one second later it is 82.2%
- by step 7 it is 89.0%

When the lane clears:

- acceleration immediately returns to 83.1%
- it remains about 82% afterward

The probability field behaves like a rapid state-dependent regime switch.

### DashScope: delayed hazard response, persistent post-event braking

DashScope reacts much more slowly to the cut-in:

- step 4 braking: 9.4%
- step 5 braking: 11.3%
- step 6 braking: 34.7%
- step 7 braking: 53.8%

By then, many trajectories have already reached a small gap.

After the lane clears, the response does not immediately reset:

- step 8 braking: 46.2%
- step 9: 56.1%
- step 10: 62.9%
- step 11: 68.3%
- step 12: 71.6%
- step 15: 77.1%

This produces a striking closed-loop effect: DashScope has roughly the same mean maximum speed as Jev, but its mean final speed falls to 20.6 mph while Jev recovers to 46.8 mph.

## History creates hysteresis

The simulator carries recent-history fields such as:

- near_miss_last_10s
- hard_brake_last_10s
- last_action

Once a near miss occurs, that information remains in the supplied state for the rest of this short rollout.

The earlier counterfactual experiment already showed DashScope is highly sensitive to near-miss history. In the Monte Carlo experiment, that sensitivity becomes a feedback loop:

~~~text
late reaction
  -> small gap / near miss
  -> near-miss history enters state
  -> braking probability rises
  -> slower vehicle state
  -> history remains salient
  -> braking remains high even after lane clears
~~~

This is an important System One runtime result.

A seemingly small state-design choice can create **behavioral hysteresis** over many transitions. The runtime is therefore not merely a transport layer for model state. State semantics and state decay are part of the policy.

A better next version should timestamp history events and expire or decay them explicitly rather than using a latched boolean.

## Worldline divergence

The matched random-number experiment also shows how quickly two probability fields produce different worlds.

The fraction of paired worldlines sampling the exact same action at each step is:

- step 0: 33.2%
- step 3: 59.1%
- step 4 cut-in: 25.3%
- step 5: 12.0%
- step 7: 32.0%
- step 9: 34.8%
- step 12: 20.9%
- step 15: 19.0%

Mean absolute speed difference between paired worldlines grows from about 2.3 mph at step 0 to about 29.8 mph at step 15.

So repeated probability sampling creates a genuine trajectory-distribution effect:

> Small local probability differences do not remain local. They are fed back through the environment, alter future state, alter the next probability distribution, and compound into macroscopically different worlds.

## Sampling self-consistency

The sampler itself behaves correctly.

| Action | Jev predicted | Jev sampled | DashScope predicted | DashScope sampled |
|---|---:|---:|---:|---:|
| hard_brake | 10.7% | 10.7% | 20.6% | 20.7% |
| brake | 17.5% | 18.0% | 19.9% | 20.2% |
| keep_speed | 0.9% | 0.8% | 10.9% | 10.8% |
| accelerate | 39.0% | 39.2% | 29.1% | 29.2% |
| hard_accelerate | 31.9% | 31.4% | 19.4% | 19.2% |

Maximum absolute empirical-minus-predicted gap:

- Jev: 0.57 percentage points
- DashScope: 0.28 percentage points

This is **not** external calibration evidence. Since the simulator samples directly from the reported model distribution, convergence is expected. It verifies that the stochastic simulation layer faithfully realizes the model's probabilities.

## What this experiment establishes

The result strengthens the original probability-simulator idea.

System One can be treated as a learned conditional transition-policy component:

~~~text
P(action_t | state_t, actor_profile, history_t)
~~~

The runtime then samples an action and lets the environment generate the next state.

This is qualitatively different from writing fixed random rules such as:

~~~text
if aggressive_driver:
    10% hard_accelerate
    20% accelerate
~~~

The probabilities themselves are recomputed after every state transition.

The Monte Carlo result shows that this matters: provider-specific probability fields generate provider-specific distributions over entire future histories.

## What it does not establish

The experiment does not tell us whether either learned distribution matches real human drivers.

That requires an external empirical distribution:

~~~text
real trajectories
  -> estimate P_real(action | state, profile, history)

model
  -> P_model(action | state, profile, history)

compare P_model vs P_real
~~~

Only then can Brier, log score, calibration curves, conditional coverage, and rare-event frequency be interpreted as behavioral calibration rather than sampler self-consistency.

Likewise, the toy 1D dynamics have no steering, lane geometry, perception uncertainty, reaction delay, collision dynamics, controller saturation, or continuous trajectory policy.

## Next experiment

The next high-value step is to separate three sources of stochasticity:

1. **actor policy randomness**
   - System One estimates the driver's action distribution

2. **environment/event randomness**
   - another System One head estimates events such as cut-in, pedestrian entry, sudden braking, obstruction, or signal change

3. **physics/sensor uncertainty**
   - continuous noise or a dedicated world model

That would give a richer probabilistic simulator:

~~~text
state_t
  -> actor System One -> P(actor action)
  -> event System One -> P(exogenous event)
  -> sample both
  -> physics/world transition
  -> state_t+1
~~~

At that point, multiple actors can each carry their own persistent profile and history, and the simulator can study emergent tail events rather than only one ego driver's response to a fixed cut-in.
