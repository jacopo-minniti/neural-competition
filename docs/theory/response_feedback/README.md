# Local response model: evidence and reproduction

Read the [theory and conclusions](../local_response_model.md) and
[maintained research context](../../research_context.md). This continuation tests
one mechanism: coupled coefficient–feature learning changes correction responses.
The physical ReLU readout is evaluated at forecast parameters; its gates need not
be used to evolve those parameters. No outcome-fitted coefficients are used.

## Records and chronology

| Directory | Experiment | Independent initialization seeds |
|---|---|---|
| `pilot` | Scalar feedback and single-mode relaxation; negative result beyond very short horizons; synchronized-copy control | 283001–283004 |
| `bilinear_pilot` | Fixed-mask bilinear forecast; full-gradient branches | Same four development seeds |
| `sgd_pilot` | Independent forecast batches; four streams, batch 128 | Same four |
| `peer_pilot` | Twenty steps on nine classes, then 80 full-task recovery; fixed-demand option value | Same four |
| `peer_decoded_pilot` | Add physical ReLU evaluation at predicted parameters | Same four |
| `decoded_sgd_pilot` | Physical readout for both bilinear and linear evolution | Same four |
| `bilinear_validation` | Locked full-gradient intervention forecast | 283101–283108 |
| `sgd_validation` | Locked intervention forecast; eight independent forecast/outcome stream pairs | Same eight confirmation seeds |
| `peer_validation` | Locked physical-response forecast under peer pressure | Same eight confirmation seeds |
| `decoded_validation` | Locked comparison of physical readout with/without bilinear feedback; four stream pairs | 281801–281808 |

All checkpoint trajectories pre-existed. The last eight are the earlier ordinary
MNIST cohort, **unused for developing these counterfactual intervention forecasts**,
not newly drawn training runs. Forecasts use current checkpoint parameters, labels,
and the declared pressure schedule. Intervention forecasts are written before the
corresponding actual branch is executed. The three locked designs are
[initial validation](validation_plan.json), [peer validation](peer_validation_plan.json),
and [decoded validation](decoded_validation_plan.json). Later hypotheses were
chosen using pilot results; their chronology is recorded explicitly.

Each producer directory contains `protocol.json`, executed script snapshots,
`forecasts.jsonl`, `outcomes.jsonl`, and `complete.json`. Initial matching is checked
on full data and the full-kernel algebra is independently tested with autograd.
The scalar pilot checks exact synchronized-copy trajectory equivalence.
Forecast protocols hash input checkpoints; peer protocols identify the input
collection but did not hash checkpoints at launch. The analyzer verifies source
snapshots and all available checkpoint hashes, unique row counts and completion
markers. Shared pilot outcomes are identical when forecast readouts are added;
these repeated branches are not counted as new initialization replicates.
Original input sources and data provenance remain in the
[observed-learning evidence](../observed_learning/README.md).

[analysis.json](analysis.json) is derived, not a new experiment.
[summary.png](summary.png) shows intervention forecasts, width-dependent effects,
and the distinction between focal loss harm and focal response weakening.
[model_comparison.png](model_comparison.png) compares the unified physical-readout
model with its two ablations. Error ratios are RMSE divided by a zero-change
forecast's RMSE. Streams are averaged before scoring seed/width/checkpoint/horizon
states. Uncertainty resamples initialization seeds, preserving dependence within
seeds; the many rows are not independent training replicates. Eight-seed bootstrap
intervals are descriptive and do not certify a universal accuracy threshold.

## Reproduce

From the repository root, use a **new** output path; producers refuse to replace
an existing protocol. These commands require the local prepared MNIST data and
saved checkpoints. Full-gradient and peer runs are longer than SGD forecasts.

```bash
# Scalar pilot, including both unsuccessful scalar closures and copy control.
.venv/bin/python scripts/studies/probe_response_feedback.py --output tmp/feedback-scalar

# Initial held-out intervention forecasts: full gradient and independent-stream SGD.
.venv/bin/python scripts/studies/forecast_response_feedback.py \
  --input docs/theory/observed_learning/mnist_confirmation \
  --output tmp/feedback-full
.venv/bin/python scripts/studies/forecast_response_feedback.py \
  --input docs/theory/observed_learning/mnist_confirmation \
  --output tmp/feedback-sgd --steps 20 100 800 --batch 128 --streams 8 \
  --independent-forecast

# Held-out capability forecast under other-class pressure.
.venv/bin/python scripts/studies/probe_peer_response.py \
  --input docs/theory/observed_learning/mnist_confirmation \
  --output tmp/feedback-peer --decode

# Unified decoded model versus linear dynamics with the same physical readout.
.venv/bin/python scripts/studies/forecast_response_feedback.py \
  --input docs/theory/observed_learning/mnist_original_replay \
  --output tmp/feedback-decoded --steps 20 100 --horizons 20 80 \
  --batch 128 --streams 4 --independent-forecast --decode

# Audit retained evidence and regenerate all reported scores and figures.
MPLCONFIGDIR=/tmp/neural-mpl .venv/bin/python scripts/studies/analyze_response_feedback.py
.venv/bin/pytest -q tests/test_response_feedback.py
```

The current scripts contain later optional readout ablations. Exact historical
versions are retained per directory; the options and seeds in each protocol
specify the executed comparison. The tests check the entire matched kernel,
one-step identity across activation crossings, the smooth transport derivative,
independent autograd gradients for both approximations, and the matrix-free
full-data resolvent against a dense calculation. Existing raw outcomes are not
silently replaced when code evolves.
