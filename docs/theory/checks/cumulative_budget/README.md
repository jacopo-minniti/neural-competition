# Running objective-budget plots

This is a read-only analysis of original cumulative tables, not a new training
experiment. See the [report](../../../reports/cumulative_budget.md).

Reproduce from the repository root:

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/plot_cumulative_budget.py
```

The [producer](../../../../scripts/studies/plot_cumulative_budget.py) reads the
retained cumulative tables, full-cohort endpoint tables, retention manifest
and available seed-0 raw update arrays. It writes only this directory and
`docs/figures/cumulative_budget/`.

- `budget_curves.csv.gz`: exact prefix sums for every retained plotting time,
  with condition/width/seed, G/H/net pair cost, positive cost, favorable credit,
  reconstructed actual progress and after-update reference loss.
- `endpoints_per_seed.csv`: each original run's endpoint.
- `endpoints_mean.csv`: seed-mean endpoints.
- `manifest.json`: input and script hashes, scope and numerical checks.

The data cover 90 original teacher-8 robustness runs (five seeds; momentum
.1, Adam .001/.01, and Adam .001 output-std .1/1) plus 40 original 200k
SGD/momentum runs (ten seeds). Graphs display widths 8/16/32 where available;
the robustness data additionally retain width 10. References are fixed
training subsets, 256 examples for robustness and 512 for the 200k study.
Their raw loss levels are kept separate.

Every actual training update contributes to each prefix sum. Only plotting
times are subsampled in the original archive. G/H monotonicity and net cost
= positive cost minus favorable credit are checked. G−H−net matches the
original final reference-loss decrease for all 130 runs. On retained seed-0
traces it is also checked at every saved plotting time against independently
evaluated raw after-step losses. The original plotting table's pre-update
`reference_mse` value is not mistaken for after-step loss.

Curves average seeds at each training step. Bands use 1,500 seed-bootstrap
resamples (seed 741), with the same draws across time/components. They are
pointwise, exploratory intervals, not independent-update uncertainty or
paired-difference tests. No division by G or by initial loss is performed.
