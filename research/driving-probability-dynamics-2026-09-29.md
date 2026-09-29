# Dynamic System One probability distributions in a toy driving loop

Date: 2026-09-29

Workflow run: https://github.com/BestNathan/system-one-research/actions/runs/36594467165

## Question

When the state changes, does a System One decision model change the full conditional
action distribution in a structured way, or does it behave more like a static classifier
whose only meaningful output is the argmax label?

The experiment also asks two follow-up questions:

1. Can a stable driver-style variable change the distribution even when the physical road
   state is held simple?
2. If actions are sampled from the returned distribution, do non-argmax actions actually
   appear in a reproducible closed loop?

## Setup

The environment is intentionally small: a one-dimensional longitudinal driving world.

The action space is:

- hard_brake
- brake
- keep_speed
- accelerate
- hard_accelerate

Both providers receive the exact same state/question structure:

- TypeSafe Jev 1.13.0
- DashScope decision-model-preview

The model is asked to predict what this particular driver is most likely to do during the
next second, not what an idealized safest driver should do.

This is a probability-dynamics experiment. It is not a driving-safety benchmark and neither
model should be interpreted as a production vehicle controller.

## Probes

### 1. Identical-state repeatability

The exact same state is submitted five times.

This estimates the model/provider output-noise floor before interpreting state-conditioned
distribution movement.

### 2. Front-gap intervention

Hold driver profile and ego speed fixed while changing only the distance to a slower lead
vehicle:

8 m -> 15 m -> 30 m -> 60 m -> 120 m -> clear road.

### 3. Aggressiveness intervention

Hold the road clear and sweep the supplied driver aggressiveness from 0.00 to 1.00.

### 4. History counterfactual

Hold the physical state fixed and change only recent history:

- neutral
- a near miss three seconds ago
- just hard-accelerated into an opening

### 5. Closed loop

Run a 16-step seeded loop.

The driver begins on a clear road. At step 4 a slower vehicle cuts in. At step 8 the lane
clears again. Each model returns a probability distribution, an action is sampled from that
distribution with a fixed PRNG seed, physics is advanced, and the new state is sent back to
the model.

The fixed seed is 20260929.

## Main metrics

| Metric | Jev | DashScope |
|---|---:|---:|
| identical-state mean pairwise JS | 0.00197 | 0.00000 |
| front-gap adjacent mean JS | 0.02199 | 0.00940 |
| front-gap endpoint JS | 0.25519 | 0.04039 |
| gap vs expected acceleration Spearman | 1.000 | 0.829 |
| gap vs acceleration-mass Spearman | 1.000 | 0.086 |
| aggressiveness vs expected acceleration Spearman | 0.821 | 1.000 |
| aggressiveness vs acceleration-mass Spearman | 0.975 | 1.000 |
| near-miss-history JS vs neutral | 0.11641 | 0.50500 |
| cut-in JS | 0.40983 | 0.02662 |
| cut-in delta braking probability | +66.0 pp | +5.9 pp |
| lane-clear delta acceleration probability | +79.0 pp | +32.5 pp |
| closed-loop non-argmax samples | 3 / 16 | 5 / 16 |
| minimum sampled-action probability | 22.0% | 26.0% |
| mean hosted latency | 110 ms | 526 ms |

DashScope returned exactly the same distribution on all five identical-state repeats, so a
signal-to-repeat-noise ratio is undefined rather than meaningfully "infinite".

## Result 1: the distribution really does move with state

The strongest controlled result is the front-gap sweep.

### Jev

At an 8 m gap:

- braking mass: 49%
- acceleration mass: 44%
- expected acceleration: -0.460 m/s2

With a clear road:

- braking mass: 0%
- acceleration mass: 100%
- expected acceleration: +1.980 m/s2

The expected-acceleration relationship with front gap is perfectly monotonic in this sweep
(Spearman = 1.0), and the endpoint JS divergence is 0.255.

This is far larger than Jev's identical-state noise floor of about 0.002 JS.

### DashScope

At an 8 m gap:

- braking mass: 9.9%
- acceleration mass: 28.7%
- expected acceleration: +0.297 m/s2

With a clear road:

- braking mass: 3.0%
- acceleration mass: 54.5%
- expected acceleration: +0.955 m/s2

DashScope therefore also changes the distribution with the physical state, but its gap
response is materially weaker in this zero-shot setup. Endpoint JS is 0.040.

The important point is not that one model is "the better driver". These models were not
trained or validated as driving controllers. The important result is that both expose a
state-conditioned distribution rather than a fixed action prior.

## Result 2: latent-style state can reshape the distribution

DashScope reacts especially strongly to the supplied aggressiveness variable.

### DashScope

Aggressiveness 0.00:

- acceleration mass: 1.0%
- expected acceleration: -0.025 m/s2
- argmax: keep_speed

Aggressiveness 1.00:

- acceleration mass: 74.3%
- expected acceleration: +1.446 m/s2
- argmax: accelerate

Both rank correlations are 1.0.

### Jev

Jev also moves in the expected direction, but it begins with a very strong acceleration
prior even at aggressiveness 0.00:

- acceleration mass at 0.00: 92%
- acceleration mass at 1.00: 100%

Expected acceleration still rises from 1.500 to 1.935 m/s2, but the probability mass has a
large ceiling effect.

This is a useful reminder that "the distribution changes with state" and "the model has the
right behavioral prior" are separate questions.

## Result 3: history changes the distribution even when physics is identical

For the same physical state, changing only recent history produced a large shift.

### Jev

Neutral history:

- braking mass: 0%
- acceleration mass: 99%

Near miss three seconds ago:

- braking mass: 28%
- acceleration mass: 68%

JS divergence: 0.116.

### DashScope

Neutral history:

- braking mass: 5.1%
- acceleration mass: 31.3%

Near miss three seconds ago:

- braking mass: 94%
- acceleration mass: 2%

JS divergence: 0.505.

So the effective decision function in this setup is better described as

P(action | physical_state, driver_profile, recent_history)

rather than only

P(action | physical_state).

The model itself is still stateless across requests. "Memory" exists only because recent
history is explicitly represented in the supplied state.

## Result 4: sampling produces non-argmax actions

The closed loop does not always execute the highest-probability action.

With the same fixed PRNG seed:

- Jev sampled a non-argmax action 3 times in 16 steps.
- DashScope sampled a non-argmax action 5 times in 16 steps.
- the lowest selected probabilities were 22% and 26%, respectively.

For example, Jev step 3 had hard_accelerate as the argmax but sampled accelerate at only
22% probability.

This demonstrates the architectural separation that motivated the experiment:

state
-> System One estimates P(action | state)
-> sampler draws one action
-> environment transitions
-> new state
-> System One estimates a new distribution

System One is therefore not a source of "true randomness". It is the conditional
distribution estimator. Randomness comes from the sampler. A hardware or operating-system
entropy source could replace the seeded PRNG without changing the model architecture.

## Result 5: the closed loop also exposes a negative safety result

Jev reacts strongly when the slower vehicle cuts in:

- cut-in JS: 0.410
- braking probability increases by 66 percentage points

When the lane clears again, acceleration probability increases by 79 percentage points.

DashScope reacts much less strongly to the same cut-in:

- cut-in JS: 0.0266
- braking probability increases by only 5.9 percentage points

In the sampled DashScope trajectory, the gap eventually reaches the simulator's 1 m floor
while the model still assigns most probability mass to acceleration.

That is not evidence that DashScope is intrinsically unsafe. It is evidence that a generic
zero-shot decision model plus natural-language driving state is not a valid safety
controller.

A real autonomous-driving stack would need learned driving-specific representations,
continuous trajectory generation, explicit physical constraints, and a hard safety/control
layer beneath any semantic System One policy.

## Conclusion

The core hypothesis is supported by this experiment:

> A System One decision model can be treated as a dynamic conditional probability model.
> When state, latent-style variables, or recent history change, the returned action
> distribution can move materially even when the action space itself is unchanged.

The experiment also makes the boundary clear:

> The decision model estimates the distribution; sampling creates the stochastic event.

That distinction is useful for probabilistic simulation. Instead of encoding fixed event
rates such as "5% chance of aggressive acceleration", the simulator can repeatedly infer

P(event | current_state, actor_profile, history)

and then sample from the newly inferred distribution after every state transition.

The next useful experiment is Monte Carlo rather than a single trace: run many trajectories
from the same initial state, measure the empirical event frequencies against the model's
reported probabilities, and test whether state-conditioned rare events are calibrated over
time.
