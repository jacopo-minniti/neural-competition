# Allocation and residual-work evidence

See the [report](../../reports/allocation.md) for conclusions, mathematical
definitions, limitations and source citations. This directory contains every
setting and seed, including negative transfer and reversed width comparisons.

## Files and units

| File | Content |
| --- | --- |
| `teacher_episodes.csv.gz` | 6,125 sampled states × two step fractions; one row per diagnostic episode after aligning intervention arms |
| `teacher_fullcal_episodes.csv.gz` | Four core conditions repeated using all training examples and identical test examples |
| `teacher_matched_episodes.csv.gz` | Four fresh candidate minibatches at each of 486 teacher target states, two fractions |
| `teacher_matched_coverage.csv` | All 490 intended target states, including four missing targets |
| `mnist_episodes.csv.gz` | 119 matched checkpoints × four fresh candidate minibatches × two fractions |
| `per_seed.csv.gz` | Within-seed pooled metrics by condition, width, scope and fraction |
| `summary.csv.gz` | Equal seed means and pointwise bootstrap intervals |
| `width_contrasts.csv.gz` | Largest minus smallest width, computed within paired seed before bootstrap |
| `readout_per_seed.csv`, `readout_summary.csv` | All 245 late-state readout refits, training/test losses and matrix rank |
| `*_provenance.json` | Producer/source/input hashes, coverage and numerical details |
| `manifest.json`, `audit.json` | Analysis input hashes, cohort counts and numerical/source audit |

Raw arm-level results, proposed-state tensors and logs remain under ignored
`runs/allocation/` and `runs/allocation_mnist/`. Their compact derived outcomes
are versioned here. These are new outputs; the original retained archive was
read-only. Small pilots and the first full-calibration pass using all test data
remain under separate ignored directories and are **not** silently pooled into
the reported cohorts.

`positive_work` and `negative_work` are P and N for **the ordinary proposal** on
the stated test examples. `curvature` is its exact loss-space remainder.
`progress=P-N-curvature` equals ordinary before-minus-after loss. MSE means
half squared error; MNIST MSE sums over classes and averages examples. CE is
mean cross-entropy. `corrective_cosine` is squared alignment of the ordinary
prediction/logit movement with the negative prediction/logit-loss gradient.

`neuron_benefit` is uniform-grid loss minus neuron-selected loss on held-out
examples; positive is favorable to neuron allocation. `sign_benefit` and
`group4_benefit` use the corresponding alternative. `beyond_sign` is sign-filter
loss minus joint-allocation loss. `beyond_shuffled` is shuffled-coefficient loss
minus selected-coefficient loss. `oracle_allocation_gain` is the **calibration
quadratic** ray-versus-box difference; it is not a held-out oracle.

Suffix conventions:

- `_raw`: mean objective-unit effect over sampled episodes within seed.
- `_work_pct`: 100 × sum of effect / sum of ordinary P within seed.
- `_loss_pct`: 100 × mean of effect / current held-out loss within seed.
- `anti_work_pct`: 100 × sum N / sum P within seed.
- `corrective_cosine_pct`: 100 × mean squared corrective cosine.

These quantities are not fractions of eventual error explained or measured
training speedups. The sampled log-time sums are not every-update cumulative
training accounts. Different step fractions have different work denominators.

## Selection, coverage and controls

The original teacher pass uses the 25 prespecified times from the temporal
study, the next ordinary minibatch proposal, and frozen optimizer state.
`all_sampled`, `after_100`, and `late` respectively include all times, times
at least 100, and times at least 1,000. `shared_loss_band` restricts the original
512-example calibration MSE to .0766–.2. It is a broad phase filter, not a
matched-target analysis. The full-calibration sensitivity keeps the exact same
state membership using this original loss, even though its selection objective
uses all 8,192 examples. The same 1,024 test examples are used in both passes.

`teacher_matched_targets` uses .2/.1 MSE targets with 5% tolerance, checked after
each update through update 1,000 and every ten thereafter. Matching uses the
original 512 training examples; coefficient fitting uses all training examples.
It uses the first qualifying state, not all subsequent revisits. Four candidate
batches are drawn from new deterministic streams with optimizer moments frozen.
These streams are paired across widths. Calibration includes the proposal's
training examples; only the test evaluation is strictly separate.

The four missing teacher targets are SGD width 16, seed 1, target .2; and
teacher-16/2037 Adam .001 width 16, seeds 1/2/4, target .1. Missing sampled
coverage does not certify that a trajectory never crossed the target.
For a paired seed/target, every intended width must be observed. This removes
the corresponding observed partners as well. Of 3,888 teacher matched episode
rows, 3,856 enter paired summaries; each includes one of the two fractions.
Separate `teacher_target_0.2` and `teacher_target_0.1` scopes are retained.

MNIST preserves the existing checkpoints and matching tolerances. Main
`matched_targets` summaries require both widths for each seed/target; the one
historically missing SGD target removes its observed partner, leaving 944/952
episode rows. `all_available_targets` and target-specific scopes show the
unrestricted coverage and the separate stages.

Every intervention uses coefficients between zero and one. The uniform control
tries fractions 0, 1/32, …, 1 plus the analytic quadratic optimum, selects on
calibration, and evaluates its chosen value on test. It is not test-selected
and not a guaranteed continuous nonlinear line optimum. The joint solver uses
cyclic exact coordinate minimization with a convex duality-gap bound, at most
1,000 sweeps. The bound is normalized by sum|b| + sum|M|. Maximum reported gaps
are retained, including the few that do not reach the requested 1e−10 tolerance.

Four-group assignment is balanced after a random permutation within each width.
The shuffled control permutes the joint coefficients within the same model.
The sign filter keeps b_r<0 and optimizes one common scalar on this direction.
It is a direction-quality control, not a proof that any remaining difference
is exclusively an off-diagonal pair effect.

Diagnostic interventions operate on float64 copies of the saved float32 state
and actual rounded proposal. This avoids quantizing the 1% control. They are
mathematical parameter interventions, not claims of bitwise float32 optimizer
execution after coefficient scaling. The ordinary float64 endpoint corresponds
to the actual rounded float32 proposal.

## Reproduction

From the repository root, with its existing environment and retained data:

```sh
.venv/bin/python scripts/studies/run_allocation.py
.venv/bin/python scripts/studies/run_allocation_mnist.py
.venv/bin/python scripts/studies/check_allocation_calibration.py \
  --condition trajectory/sgd \
  --condition trajectory/momentum \
  --condition robustness/k8_teacher2027/adam_001 \
  --condition robustness/k8_teacher2027/adam_01
.venv/bin/python scripts/studies/run_allocation_matched.py
.venv/bin/python scripts/studies/check_allocation_features.py
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_allocation.py
.venv/bin/python scripts/studies/audit_allocation.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

The producers skip completed caches. **Before reusing them after source or
protocol changes, run the audit**; it verifies recorded producer and numerical
module hashes. To perform a changed experiment, preserve/rename the existing
output directory before running; do not treat a skipped old cache as a fresh
run. MNIST's producer also caches per checkpoint. `--limit` puts pilot results
in separate directories. Readout refits and analysis are deterministic
postprocessing and regenerate their files.

No extra packages were installed. The seven added numerical tests cover
duplication invariance, convex optimum/error certificates, scalar and multiclass
Jacobian products, CE/MSE quadratic forms, and exact residual-work accounting.
Bootstrap draws use seed 741 and resample training seeds 4,000 times.

## Inference limits

All probes/targets/sources within a training seed are dependent. Teacher
replications share some initialization streams, and the two original momentum
cohorts overlap; do not pool them as independent studies. Confidence intervals
are conditional on the tested data/teachers and exploratory, without multiplicity
adjustment. Calibration size, loss-stage selection, scalar step-size bounds,
parameterization and grouping affect the question being asked. No sustained
allocation trajectory or mediation fraction is estimated here.
