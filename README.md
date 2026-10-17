# Neural competition

Small PyTorch experiments on whether extra neurons improve optimization and how
minibatch updates compete through shared neuron parameters. The current experiment
uses one fixed, noiseless eight-neuron ReLU teacher, with both student layers trained.

## Distribution and optimizer follow-up

See [Pier's experiment results](docs/pier_experiments.md) and
[the distribution notebook](analyses/03_pier_distributions.ipynb) for the newer
SGD/momentum/Adam/Muon comparisons, 40-seed early distributions, fresh 512-batch
tail checks, and saved-state learning-rate interventions.

```sh
uv run python scripts/analyze_distributions.py
uv run python scripts/plot_agreement.py
uv run python scripts/audit_distributions.py
```

All generated data are under ignored `runs/`; the executed notebook and figures
are in `runs/pier_study/analysis/`.
Reproduction commands and the separate training budgets are in the report.

## Neuron-level follow-up

See [the neuron-interference findings](docs/neuron_interference.md) and
[the neuron notebook](analyses/04_neuron_interference.ipynb) for positive/negative
neuron gradient masses, exact prediction changes, partial cancellation and
step-size sensitivity at widths 8 and 16. These analyses reuse saved states.

```sh
uv run python scripts/analyze_neurons.py
uv run python scripts/audit_neurons.py
```

## Momentum comparison

See [the momentum replication](docs/momentum_comparison.md) and
[the comparison notebook](analyses/05_momentum_comparison.ipynb) for raw versus
momentum-directed neuron conflict, exact prediction changes, step-length controls,
and fresh-batch validation. The saved momentum buffers are retained in candidate steps.

```sh
uv run python scripts/compare_momentum.py
uv run python scripts/audit_momentum.py
```

## Original curated study

- [Protocol and reproduction](docs/experiment.md)
- [What we found](docs/findings.md)
- [Interference definitions and limitations](docs/measurements.md)
- [Saved data, verification and cleanup](docs/data.md)
- [Training outcomes](analyses/01_training.ipynb)
- [Gradient and neuron interference](analyses/02_interference.ipynb)

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/audit_results.py
```

The original retained runs contain gradients, actual updates, losses
and curvature probes. Cleanup reused and verified them; no training rerun was needed.

Open the notebooks with this repository's `.venv` Python kernel.

## Focused experiment

Inputs are 20-dimensional standard Gaussian; 8,192 train and 4,096 test examples.
Students have widths 8, 9, 10, 16 and 128 at seeds 0, 1 and 3. Plain SGD uses learning
rate 1, batch size 64 and 100,000 updates. A separate width-8/10 pair at seed 5 is
retained as a counterexample. Training optimizes **half-MSE**; displayed findings
and training plots use ordinary MSE. See the protocol for initialization and scaling.

```sh
uv run python scripts/run_experiments.py --config configs/teacher8.json --device cpu --output runs/reproduction
uv run python scripts/run_experiments.py --config configs/teacher8_counterexample.json --device cpu --output runs/reproduction_counterexample
```

Use new output directories for reproduction, or `--resume` to skip completed runs.
The generic runner also supports momentum/Adam and CPU/MPS/CUDA. No features are frozen.

All raw data and generated figures are under ignored `runs/`. The small versioned
`docs/teacher8_history.csv` preserves the broader sweep before raw pruning. The
17 retained cases were selected for contrasting outcomes; use the broader history,
not this selected subset, to describe success rates. Across the completed broader
100k sweep, width 8 achieved test MSE below 1e-6 in 6/20 starts, versus 18/20 at width 10.

```sh
uv run pytest
uv run ruff check src scripts tests analyses
```
