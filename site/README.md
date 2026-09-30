# GitHub Pages experiment routing

The Pages site is a project-level research catalog rather than a single-experiment page.

## Routes

- `/` — project landing page
- `/experiments/` — experiment catalog
- `/experiments/<slug>/` — experiment-owned visualization

Current experiment:

- `/experiments/probability-world/`

## Adding an experiment

1. Create `site/experiments/<slug>/index.html` plus any experiment-owned assets.
2. Add one entry to `site/experiments/manifest.json`.
3. Keep generated/raw experiment data under the same route, usually
   `site/experiments/<slug>/data/`.
4. Experiment workflows should produce reproducible artifacts, not own the project root.
5. The global Pages workflow assembles experiment artifacts into their routes and deploys
   the complete `site/` tree.

This keeps visualizations isolated from one another and lets multiple experiments coexist
on the same GitHub Pages site.


## Persistent probability-world records

The probability-world experiment is intentionally iterative rather than a large batch.

Each manual workflow run generates exactly one paired Jev/DashScope world, writes a compact replay to:

`site/experiments/probability-world/runs/<workflow-run-id>/record.json`

and prepends metadata to:

`site/experiments/probability-world/runs/manifest.json`

The experiment page loads that manifest and lets the viewer switch between saved iterations with `?run=<workflow-run-id>`.

Keep the random seed fixed when comparing runtime/model changes so the sampled uniform streams stay matched across iterations. Change the seed only when intentionally exploring a different stochastic realization.
