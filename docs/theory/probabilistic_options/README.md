# Probabilistic option transitions: evidence and reproduction

See [the theory and interpretation](../minimal_probabilistic_options.md).
MNIST is paused. These are natural-width, full-batch shallow-ReLU experiments.

## Chronology and files

- Root `models.json` and fitting-source snapshots retain the first exploratory
  development round. It compared the original nested feature sets.
- `locked/models.json` is the **final pre-validation selection**, after adding
  compact ablations using only the same historical data. It records all CV
  scores, coefficients and hashes of historical trajectories. `locked/` retains
  exact producer/core snapshots and `source_manifest.json`.
- `fresh/protocol.json` fixes nine cases, widths, seed bank 297101, 64 seeds,
  step .025 and final optimizer time 2000. `fresh/models.json` is an exact copy
  of the locked model file. Four implementation sources are snapshotted there.
- `fresh/forecasts.jsonl` records every prediction with only initial or time-20
  features. `forecasts_locked.json` hashes it after all time-20 landmarks and
  before any continuation. `trajectory.jsonl` contains measurements throughout.
  Landmark NPZ files store initial/time-20 parameters; final NPZ files retain
  endpoints. `complete.json` marks successful production.
- `validation.json` and `fresh_forecasts.png` are derived by the saved analyzer.
  It verifies hashes, replays every probability, remeasures 5,184 initial,
  landmark and final parameter states, reconstructs event labels from raw
  trajectories and preserves all model comparisons and endpoint summaries.

There are 27 cohorts of 64 runs, not 1,728 independent random seeds. Initial
banks are paired across widths and compatible task dimensions. Confidence
intervals bootstrap the 64 seed indices jointly. Scores weight each component
by 1/N; four opening deadlines are dependent observations of one episode.

## Reproduction

From the repository root:

```sh
MPLCONFIGDIR=/tmp/neural-competition-mpl .venv/bin/python scripts/studies/analyze_probability_validation.py
.venv/bin/pytest -q tests/test_probabilistic_options.py tests/test_implementation_moments.py tests/test_toy_resources.py
```

The analyzer recomputes derived files and checks the archived evidence. To run
new training against the unchanged locked models, choose a **new** output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/studies/run_probability_validation.py --output /tmp/neural-probability-replay
```

This repeats the archived seed bank unless `--seed` is changed; it is a numerical
reproduction, not a fresh statistical replication. The producer refuses existing
protocol output. Defaults execute all 1,728 runs. Inspect `--help` for smaller
checks. Current analysis targets the archived `fresh/` directory.

The development producer `scripts/studies/fit_probabilistic_options.py` refuses
to overwrite `locked/models.json`. To reproduce selection, copy the archived
`locked/` Python sources to a temporary folder and change only the ROOT/OUT
paths in the fitting driver to the repository and a new directory. Do not edit
archived coefficients to improve validation scores. The selected closure model
has two slopes; the opening model has seven, with simpler saved ablations.

## Interpretation boundaries

Coverage .95 and relative gap .1 define sampled structural events, not exact
continuous event times. The same grid is used for historical and fresh early
windows. Strength at horizon 1000 is a current local response, not knowledge of
future training. Opening means first observed return of support, not sustained
availability or solved loss. Deadlines beyond 2000 are unvalidated. The smaller
fresh step is deliberate; mixed-resolution development and discrete gate effects
preclude a claim about a universal continuous-time hazard.

All model and feature choices were frozen before fresh training. Additional
bootstrap contrasts and probability replay checks were added during outcome
analysis; they do not alter forecasts. The task/width and resource comparators
were also frozen. Keep their successes alongside failures: reopening timing is
underestimated, and the extra resource proxy has mixed score/horizon effects.
