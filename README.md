# Neural competition

Small PyTorch experiments investigating when competing feature demands obstruct
learning and whether extra width helps. We seek a conditional mechanism with a
meaningful loss effect, not a universal explanation of overparameterization.

Start with [OBJECTIVE.md](OBJECTIVE.md) and the [evidence overview](docs/README.md).

| Where | Contents |
| --- | --- |
| [Current research](docs/reports/residual_growth.md) | Ordinary training, signed feature roles, strong width gaps, and failed sign predictions |
| [MNIST](docs/reports/mnist.md) | Real-data width benefits and limits of interference diagnostics |
| [Earlier findings](docs/reports/research_history.md) | Consolidated positive, negative, and superseded experiments |
| [Theory](docs/theory/README.md) | Residual potential, assumptions, and unchanged prospective protocols |
| [Code](src/neural_competition/README.md) / [scripts](scripts/README.md) | Shared numerical modules and the two maintained study pipelines |
| [Tables](docs/tables/README.md) / [figures](docs/figures/README.md) | Auditable evidence and a small set of retained plots |
| [Reproduction](docs/reproduction.md) / [data retention](docs/data.md) | Commands, local-data requirements, and recovery of historical work |

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/summarize_results.py
uv run python scripts/check_repository.py
uv run pytest -q
uv run ruff check src scripts tests
```

A small end-to-end check, without downloading data:

```sh
uv run python scripts/run_experiments.py --smoke --device cpu --output runs/smoke
```

Reports and full-cohort result tables are versioned. Raw runs and datasets are
local and ignored by Git. `scripts/check_repository.py --local-data` also verifies
the retained local files. The [retention guide](docs/data.md) distinguishes
recomputing summaries, inspecting saved states, and rerunning training.

The complete pre-curation research tree is recoverable at Git commit `b03c53f`.
It preserves retired reports, plots, source code, and versioned evidence; deleted
untracked raw arrays are not recoverable from Git.
