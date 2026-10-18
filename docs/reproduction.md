# Reproduction

Run from the repository root. CPU execution is sufficient. Full studies are
substantially more expensive than the checks below.

## Install and validate a fresh clone

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/summarize_results.py
uv run python scripts/check_repository.py
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/run_experiments.py --smoke --device cpu --output runs/smoke
```

These commands do not require the ignored historical data. The repository check
verifies relative documentation links and executed-source snapshots. On the
original workspace, add `--local-data` to verify the retained raw-file inventory.
That option is expected to fail in a fresh clone without those local files.

## Recompute current published results from retained local data

```sh
MPLCONFIGDIR=/tmp/neural-competition-mpl uv run python scripts/studies/analyze_residual_growth.py
MPLCONFIGDIR=/tmp/neural-competition-mpl uv run python scripts/studies/analyze_mnist.py
```

The first reconstructs final held-out losses from all 72 non-pilot residual-growth
and polarity parameter files and verifies the original source hashes. The second
uses retained MNIST scalar outputs and dense replay summaries. Neither reruns
training. Both write versioned tables and figures; inspect the resulting diff.
The refactor preserves numerical bodies and retains exact older source snapshots.

## Residual-growth training

The [report](reports/residual_growth.md#evidence-and-reproduction) lists the exact
main sweep and every follow-up command. It includes all negative sign/bias
controls, 20-start replication, changed data/rates/readout scale, hidden biases,
and the 100k-step extension. On a fresh clone those commands recreate the
original collection layout for analysis.

For a small new exploratory collection:

```sh
uv run python scripts/studies/run_residual_growth.py --degrees 30 --ratios -3 --seeds 2 --steps 100 --collection exploratory
uv run python scripts/studies/run_polarity.py --degrees 30 --seeds 2 --steps 100 --collection exploratory
```

Do not run the publication analyzer on a mixture of incomplete exploratory and
complete original collections: its figures assume the full reported study.
Use collection `pilot` for temporary smoke runs; the analyzer excludes it.
Producers skip completed tags. A skipped run is not a new replication or a
numerical validation. Use a new collection for changed settings. The 20-start
cohort includes the first five starts, and longer runs extend them.

## MNIST width study

A fresh complete study trains 200 thirty-epoch runs plus 96 five-epoch tuning
starts. Diagnostic replays add compute, not independent starts. These commands
download the public archives if needed and use a fresh output directory:

```sh
uv run python scripts/studies/run_mnist.py --phase tune --download --output runs/mnist_new
uv run python scripts/studies/run_mnist.py --phase main --output runs/mnist_new
uv run python scripts/studies/check_mnist.py --phase rates --input runs/mnist_new
uv run python scripts/studies/check_mnist.py --phase matched --input runs/mnist_new
uv run python scripts/studies/check_mnist.py --phase matched --collection lower_lr --input runs/mnist_new
uv run python scripts/studies/check_mnist.py --phase matched --collection width_tuned --input runs/mnist_new
uv run python scripts/studies/audit_mnist.py --all --input runs/mnist_new
uv run python scripts/studies/probe_mnist_objective.py --input runs/mnist_new
MPLCONFIGDIR=/tmp/neural-competition-mpl uv run python scripts/studies/analyze_mnist.py --input runs/mnist_new
```

The original completed study remains under `runs/mnist`. Production scripts
check frozen execution sources; use fresh directories after source changes.
All main checkpoints remain available, but intermediate control states and
control matched checkpoints were pruned. Replaying those diagnostics requires
recomputing the missing states; completion markers alone do not restore them.
The prepared dataset cache regenerates from retained official archives with the
same split/normalization; its serialized-file checksum is historical provenance,
not a claim that serialization is byte-identical across PyTorch versions.

## Historical studies

Read [research history](reports/research_history.md) and its
[table guide](tables/history/README.md). The complete old tracked tree is at
`b03c53f`; inspect files with `git show b03c53f:PATH` or use a separate checkout.
The old full archive audit no longer applies after pruning. Historical raw
arrays removed from ignored `runs/` must be regenerated with their original
producers. Do not use the current test count or retained example states to
claim that every historical experiment has been rerun.
