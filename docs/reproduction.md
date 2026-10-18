# Inspecting results and reproducing experiments

Run commands from the repository root. Reports are self-contained under `docs/`;
the representative raw archive is ignored under `runs/retained/`. The editable
and executed notebooks have been removed as requested.

## Supported checks on the cleaned repository

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/summarize_results.py
uv run python scripts/audit_archive.py
uv run pytest
uv run ruff check src scripts tests
```

The summary command reads the versioned tables, so it works without raw data.
The archive audit requires the local ignored archive and verifies its checksums,
representative per-update accounting and retained gradient/optimizer evidence.
It does not reconstruct a full-cohort confidence interval from the selected
raw cases. Reports link directly to versioned figures; no notebook is needed.

The full original computed tables, including per-seed results and dense curves,
are compressed under `runs/retained/aggregates/`. Read them with
`pandas.read_csv(path)`; gzip is detected from the extension. Original historical
paths inside tables are provenance fields and do not imply those source trees
still exist. See [the data guide](data.md) for checkpoint and raw-array loading.

## A small new training run

The [residual-growth report](reports/residual_growth.md#evidence-and-reproduction)
provides the ordinary-training and polarity-control commands, including source
version checks and reconstruction of final losses from saved parameters.

The [specialization study](reports/specialization.md#evidence-and-reproduction)
provides commands for the new correlated-feature training sweep, perturbation,
bias, rate and longer-training controls, and its source/duplicate audit.

```sh
uv run python scripts/run_experiments.py --smoke --device cpu --output runs/smoke
```

For the new residual-work and cross-fitted allocation diagnostics, use the
[allocation reproduction guide](tables/allocation/README.md#reproduction).
It includes original-trajectory replay, MNIST probes, all-training calibration,
tighter matched targets, readout refits, analysis and provenance audits.

For new experiments, use a new output directory and an explicit configuration.
The generic runner supports SGD, momentum, Adam and the hybrid Muon setup.
`--resume` skips completed/diverged sequential cases and restarts an incomplete
case from initialization; it is not optimizer-state continuation. The batched
trajectory runner has separate real checkpoint-resume behavior.

## Repeating the original studies

Original producers and numerical audits are retained in `scripts/studies/`.
They expect **full original-layout cohorts**, not `runs/retained/`. Running an
analysis immediately after cleanup cannot recreate deleted inputs from the
representative subset. Regenerate the required cohorts first. This is deliberate:
silently calculating forty-seed estimates from two selected seeds would be wrong.

Original focused SGD study:

```sh
uv run python scripts/run_experiments.py --config configs/teacher8.json --device cpu --output runs/teacher8
uv run python scripts/run_experiments.py --config configs/teacher8_counterexample.json --device cpu --output runs/teacher8_counterexample
uv run python scripts/studies/audit_results.py
```

The focused configurations reproduce the original curated subset. The broader
20-seed reliability counts are the historical cohort in
`docs/tables/core/teacher8_history.csv`, not estimates from those configurations.

Distribution and control cohorts (these are substantial runs):

```sh
for name in pier_distributions pier_early pier_early_extra pier_optimizer_pilot pier_initialization_control pier_stability_pilot; do
  uv run python scripts/run_experiments.py --config configs/studies/$name.json --output runs/$name --device cpu --resume
done
uv run python scripts/studies/continue_stability.py
uv run python scripts/studies/check_tails.py
uv run python scripts/studies/analyze_distributions.py
uv run python scripts/studies/plot_agreement.py
uv run python scripts/studies/analyze_neurons.py
uv run python scripts/studies/audit_neurons.py
uv run python scripts/studies/compare_momentum.py
uv run python scripts/studies/audit_momentum.py
uv run python scripts/studies/analyze_pairs.py
uv run python scripts/studies/audit_pairs.py
```

The continuation config is `configs/studies/pier_stability_continuation.json`.
The historical broad pair-budget derivation is preserved in the archive's
`provenance/summarize_pair_budget.py`, together with its exact source arrays for
representative cases and the full-cohort computed tables. Its formulas and
published numbers are in [the pair report](reports/pair_interactions.md).

Every-update SGD/momentum study, using the retained original dataset:

```sh
uv run python scripts/studies/run_trajectory.py --data runs/retained/datasets/k8_teacher2027.pt
uv run python scripts/studies/analyze_trajectory.py
uv run python scripts/studies/audit_trajectory.py
```

Robustness study (the partition commands may run in separate terminals):

```sh
uv run python scripts/studies/run_robustness.py --prepare-only
uv run python scripts/studies/run_robustness.py --partition 0
uv run python scripts/studies/run_robustness.py --partition 1
uv run python scripts/studies/run_initialization_controls.py
uv run python scripts/studies/analyze_robustness.py
uv run python scripts/studies/audit_robustness.py
uv run python scripts/studies/analyze_adam_cross_batch.py
```

The robustness base-data default is the retained teacher-8 dataset. Both
trajectory families create new full-layout results under `runs/pier_study/`.
The latest Adam script needs the freshly generated complete teacher-8 Adam
groups, including their original model lists and moments; it cannot use the
repacked archive as a drop-in resume folder.

The current engine and original study configurations remain versioned. Exact
execution sources from before cleanup, plus each batched group's frozen sources,
are retained in archive provenance. Code paths changed during organization;
original execution hashes refer to those immutable copies, not the relocated
live files. Float32 arithmetic and operation order can change individual basins
even when the experimental protocol is the same.

## Audit objective-scale magnitude and early predictive value

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_effect_magnitude.py
```

This reads existing cohort/retained aggregate tables and the previously
generated strict-replay caches. It performs no optimization or replay.
The [report](reports/effect_magnitude.md) and [computation directory](theory/checks/effect_magnitude/README.md)
document objective budgets, matched efficiency, direct probes and held-out
seed prediction. Outputs include input hashes and every held-out prediction.

## Plot cumulative G, H and net pair cost against training steps

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/plot_cumulative_budget.py
```

This reads exact original every-update prefix sums from the retained aggregate
tables, without training or replay. The [report](reports/cumulative_budget.md)
and [data directory](theory/checks/cumulative_budget/README.md) document the
raw-unit plots, 130-run endpoint checks and independent retained-update loss
checks. Only plotting times are subsampled; every update contributes to each sum.

## Recover strict harm at matched loss from the retained archive

The [strict-interference follow-up](reports/strict_matched_loss.md) has a
separate producer that reconstructs the original teacher-8 cohorts using
their configurations, model batch layouts and minibatch generators:

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_strict_matched_loss.py
```

It checks retained parameters/moments and RNG states bitwise, every retained
seed-0 update field exactly, and archived five-seed bin frequencies/visits.
Original archives are read-only. Outputs, definitions, hashes and cache
semantics are documented in the [diagnostic directory](theory/checks/strict_matched_loss/README.md).
Cached summaries and figures can be regenerated with `--summarize-only`
without replaying optimization. Use `--refresh` to force a replay.

## Run sampled temporal peer-update interventions

```sh
.venv/bin/python scripts/studies/run_temporal_branches.py --match-width-dose
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_temporal_branches.py
```

This replays the original 245 trajectories to recover sampled states, then runs
short paired counterfactual branches with identical future minibatches. It
changes one sender update, observes all other neurons and calibrates uniform
controls. Prediction-space pulse size is matched across widths within each
seed/checkpoint. Original archives are read-only. Each completed condition is
cached under `runs/temporal_branches/`; `--refresh` regenerates it.

For the fixed-20% sensitivity comparison, omit `--match-width-dose` in the
producer and pass `--design fixed_fraction` to the analyzer. The
[report](reports/temporal_branches.md) and
[computation directory](theory/checks/temporal_branches/README.md) document
scope, exact checks, sampling, outcomes, uncertainty and raw-data locations.

## MNIST width study

The [MNIST report](reports/mnist.md) and [complete tables](tables/mnist/README.md)
describe the new real-data cohorts. These commands download the four public
MNIST archives and run a fresh full study. They train 200 thirty-epoch runs plus
96 five-epoch tuning starts; diagnostic replays add computation without adding
independent starts. Use a new directory when changing the source or protocol.

```sh
.venv/bin/python scripts/studies/run_mnist.py --phase tune --download --output runs/mnist_new
.venv/bin/python scripts/studies/run_mnist.py --phase main --output runs/mnist_new
.venv/bin/python scripts/studies/check_mnist.py --phase rates --input runs/mnist_new
.venv/bin/python scripts/studies/check_mnist.py --phase matched --input runs/mnist_new
.venv/bin/python scripts/studies/check_mnist.py --phase matched --collection lower_lr --input runs/mnist_new
.venv/bin/python scripts/studies/check_mnist.py --phase matched --collection width_tuned --input runs/mnist_new
.venv/bin/python scripts/studies/audit_mnist.py --all --input runs/mnist_new
.venv/bin/python scripts/studies/probe_mnist_objective.py --input runs/mnist_new
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_mnist.py --input runs/mnist_new
```

The original completed study is in `runs/mnist`. Its original producers skip
completed cases only when configurations match, and check their frozen source
copies. Data preparation verifies archive MD5 hashes and preserves split and
reference indices. No torchvision dependency is needed: the loader reads the
official IDX format directly. Use the repository environment described above.
The study uses four CPU workers with one thread each; no GPU is required.

The every-update audit checks all original main endpoints and optimizer moments
bitwise before replacing sampled primary G/H/C_all totals. Individual pair costs
still sample training time, and CE also samples pairs. Analysis saves complete
per-seed tables and all figure panels; it does not substitute a retained example
subset for the full cohort.
