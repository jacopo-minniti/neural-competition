# Entry points

Run commands from the repository root with the environment installed as described
in [reproduction](../docs/reproduction.md).

| Entry point | Purpose |
| --- | --- |
| `summarize_results.py` | Print headline current results from versioned tables; no raw data needed |
| `check_repository.py` | Check internal documentation links and retained source hashes; `--local-data` also checks the local inventory |
| `run_experiments.py` | Generic configurable teacher/student training; `--smoke` is a tiny integration check |
| [studies/](studies/README.md) | Maintained feature-profile, residual-growth/polarity and MNIST pipelines |

Study scripts configure experiments and publish evidence. Numerical kernels and
task construction belong in [the package](../src/neural_competition/README.md).
Retired one-off producers and audits are in Git commit `b03c53f`, along with their
original configuration files. They are not required by the maintained pipelines.
