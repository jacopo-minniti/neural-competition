# Objective-scale magnitude and predictive audit

New analysis of existing tables, 18 October 2026. No optimization, replay,
intervention, dataset modification or archive mutation is performed here.
The [report](../../../reports/effect_magnitude.md) interprets the results.

## Producer and inputs

From the repository root:

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_effect_magnitude.py
```

[analyze_effect_magnitude.py](../../../../scripts/studies/analyze_effect_magnitude.py)
uses the retained robustness/trajectory aggregates, versioned cohort tables,
the prior verified strict-replay caches, and the existing momentum/Adam probe
tables. `manifest.json` records each input SHA-256 and the script hash. Only
this output directory and `docs/figures/effect_magnitude/` are written.

Fixed-reference budgets cover 205 robustness and 40 original SGD/momentum
runs. Reference sets differ (256 versus 512 examples), so the studies are
kept separate. Strict-harm scale uses the 90 original teacher-8 runs recovered
by the [previous replay](../strict_matched_loss/README.md). No purposively
retained checkpoint subset is substituted for a full cohort.

## Outputs

- `cumulative_per_seed.csv`: G/H/C_plus/C_minus, loss decrease, total linear
  residual work W, squared-movement cost Q, and anti-work bounds for every
  fixed-reference run.
- `budget_contrasts_per_seed.csv` / `budget_contrasts.csv`: exact bridge
  from narrow to wide final reference loss, including initialization,
  standalone and net pair terms; paired seed intervals.
- `matched_efficiency_per_seed.csv` / `matched_efficiency_summary.csv`:
  objective progress and costs per standalone helpful unit at matched loss.
- `matched_contrasts_per_seed.csv` / `matched_contrasts.csv`: paired width
  differences and their exact retention decomposition.
- `strict_cumulative_scale.csv` / `strict_cumulative_summary.csv`: strict
  ordered-pair harm, counts and scale relative to objective loss decrease.
  Strict harm overlaps neurons; this is not a causal/additive fraction.
- `direct_objective_progress.csv`: actual finite-step loss improvement on
  fresh target batches for SGD/momentum, early and later phases.
- `adam_first_order_scale.csv`: early net first-order gain and global
  harmful-tail magnitude; these are not finite-step loss changes.
- `prediction_pairs.csv`: early width-contrast features and later objective
  targets. Each prefix has 140 contrasts from the 205 original runs.
- `held_out_predictions.csv`: every actual target, held-out prediction and
  squared error, including the baseline model.
- `prediction_scores.csv`: prediction RMSE changes, held-out-fold counts and
  descriptive within/between comparison correlations.
- `manifest.json`: input hashes, scope, aggregation and prediction protocol.

Floating-point values are stored as fractions unless an outcome is explicitly
in loss units; e.g. .1012 in a retention contrast means 10.12 percentage points.
Displayed reports average per-seed ratios, not ratios of the displayed means.

## Algebra and verification

The producer checks both identities for all fixed-reference runs:

\[
D=L_0-L_T=G-H-(K_+-K_-)=W-Q,
\]

\[
I=\sum_{t,r}\|d_{r,t}\|^2/2,\quad W=G-H+I,\quad Q=I+K_+-K_-.
\]

I is recovered from the original ratio `penalty_per_energy=K_plus/I`.
Every used cumulative record has positive K_plus and a positive ratio; the
producer asserts this instead of treating 0/0 as zero. It reconstructs
neither individual signed linear-work parts nor perpendicular work from an
aggregate ratio.

For s_r=−<e,d_r>, A_r=−s_r+I_r implies
max(H−I,0)≤sum[-s_r]_+≤H. Bounds pool time and neurons and are conservative.
N/P bounds use P=W+N and W>0, applying the ratio within each seed before
averaging. Bounds are not a new measurement of the exact hidden numerator.

Matched retention uses ratios of sums within seed/bin, then equal bin/seed
weights. The 200k study uses all 12 original .02–.2 bins and ten seeds; the
five teacher-8 robustness conditions use the same five .0766–.2 bins and
five seeds. Bin coverage, finite ratios and both additive decompositions
are asserted. Intervals use 4,000 paired-seed resamples (seed 741), are
exploratory and pointwise, and do not resample dependent pairs as independent.

Tests check invariance of the total-step budget under identical neuron-copy
subdivision, the anti-residual-work bound on an explicit opposing correction,
and that changing held-out outcomes cannot affect fitted predictions.

## Forecast protocol and interpretation

For each prefix 1k/5k/10k, early cumulative metrics are paired between the
minimal teacher-sized student and each larger width. A reduction of cost
is positive; increased favorable credit is also positive. Future outcomes
measure narrow-minus-wide loss, so positive means a width benefit.

Future average loss integrates the saved full-training half-MSE curve from
the prefix through 100k with the trapezoid rule (1,000-update grid); the
other target averages the 91k–100k evaluations. Both use the teacher's
zero-predictor training half-MSE as their scale, rather than a log-error
ratio that could magnify tiny differences near zero. Initial and current
training losses are measured on the full training set; predictor metrics
come from the original fixed reference. No later-time metric is included
in the prefix features.

An ordinary least-squares baseline has a separate intercept for each of the
28 teacher/optimizer/width comparisons, two current losses, and the initial
loss gap. Each added model receives one extra early metric. One entire
initialization seed is held out across all conditions/widths simultaneously.
Standardization and coefficients use only the other four seeds. No tuning,
choice of cutoff based on its score, or nonlinear feature search is performed.
Cutoffs/features are exploratory, not preregistered.

Comparison intercepts remove the mean benefit expected for that width pair.
The check therefore asks about predicting individual deviations from it,
not whether all mean width effects can be explained by an interaction.
Between-comparison correlations provide a separate descriptive check of
average effects; overlapping contrasts and common teachers are not treated
as independent experiments. Five held-out seed folds do not establish
generalization to a new teacher, dataset, regime or optimizer.

Prediction RMSE improvement is not a percentage of width benefit explained.
The baseline already contains information about the early fitting trajectory.
An unsuccessful incremental linear forecast neither validates a different
metric nor rules out all causal, nonlinear or phase-specific mechanisms.
