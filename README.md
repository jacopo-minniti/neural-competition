# Neural competition

Small PyTorch teacher–student experiments testing whether extra neurons improve
optimization through less interference or more useful reinforcement.

Width improves fitting in the tested shallow ReLU problems. Some early momentum
measurements support reduced conflict, but the mechanism depends on the optimizer
and training phase. Wider Adam students can fit better while accumulating more
pairwise interaction cost. See the [research summary](docs/README.md) for the
evidence, counterexamples and remaining causal questions.

- [Protocol and study ledger](docs/protocol.md)
- [Metrics, signs and interpretation](docs/metrics.md)
- [Detailed reports](docs/reports/)
- [Literature-guided explanations and predictions](docs/theory/README.md)
- [Temporal causal peer-update experiment](docs/reports/temporal_branches.md)
- [MNIST width experiments](docs/reports/mnist.md)
- [Result tables](docs/tables/README.md) and [figures](docs/figures/README.md)
- [Data retention](docs/data.md) and [reproduction](docs/reproduction.md)

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/summarize_results.py
uv run python scripts/audit_archive.py
uv run pytest
uv run ruff check src scripts tests
```

Reports, compact tables and figures are versioned. Representative raw data and
all original computed tables live in ignored `runs/retained/`; the archive audit
requires that local data. All notebooks have been removed. Selected raw cases
illustrate outcomes; use the full-cohort tables for rates and uncertainty.

The small package is in `src/neural_competition/`. `scripts/run_experiments.py`
is the generic training entry point; `scripts/studies/` preserves the original
study producers and audits. Historical study configurations are grouped under
`configs/studies/`. Full reruns use fresh original-layout outputs, separately
from the compact archive; see the reproduction guide before running them.

For a small new run:

```sh
uv run python scripts/run_experiments.py --smoke --device cpu --output runs/smoke
```
