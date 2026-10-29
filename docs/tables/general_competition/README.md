# General competition exploratory evidence

`endpoints.csv` retains all 270 model endpoints; `summary.csv` gives means and
paired 5000-resample bootstrap intervals in target-variance MSE percentage points.
`audit.csv` reconstructs test MSE from all nine local checkpoint bundles.
`provenance.json` and `sources/` retain exact source hashes and snapshots.
`score_check.json` is a separate exploratory Monte Carlo sanity check, not a
finite-sample certified population bound.

Run from the repository root:

```sh
.venv/bin/python scripts/studies/probe_general_competition.py
.venv/bin/python scripts/studies/analyze_general_competition.py
```

The producer is an exploratory fixed-design script and overwrites its own output
under ignored `runs/general_competition/`. It does not modify earlier studies.
See the [report](../../reports/general_competition.md) and
[theory](../../theory/general_competition.md) for assumptions and limits.
