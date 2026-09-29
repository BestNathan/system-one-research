# Driving probability dynamics

This experiment probes whether a System One decision model changes its full action
probability distribution as state changes.

It uses a deliberately small one-dimensional longitudinal driving world so the
interventions are easy to reason about:

- identical-state repeats establish a model/provider noise floor;
- front-gap sweeps change one physical variable at a time;
- aggressiveness sweeps change one driver-profile variable at a time;
- history counterfactuals hold physical state fixed and alter recent memory;
- a seeded closed loop inserts a sudden cut-in and later clears the lane.

The action set is longitudinal and discrete: hard brake, brake, keep speed,
accelerate, hard accelerate. This is a probability-dynamics experiment, not a
claim that Jev or DashScope should directly control a real vehicle.

The workflow records complete distributions and reports Jensen-Shannon
divergence, total variation, expected acceleration, monotonic rank correlations,
and the sampled closed-loop trace.

The random sampler uses a fixed PRNG seed (20260929). System One provides the
conditional action distribution; the PRNG only samples from it.
