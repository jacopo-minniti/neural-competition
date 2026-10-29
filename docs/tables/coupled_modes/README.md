# Coupled-mode forecast evidence

- `forecasts.csv`: all 120 horizon rows across four exploratory stages; no
  unstable cases removed. Each row contains both branch losses and their contrast.
- `prediction_metrics.csv`: RMSE, zero-prediction baseline, signs and mean effects.
- `audit.csv`: endpoint reconstruction for 40 saved branch-pair bundles.
- `provenance.json`, `analysis_provenance.json`, `sources/`: executed source snapshots.

Five paired initialization seeds at two widths recur across stages and horizons.
Do not treat these rows or stages as independent training replicates.
Raw parameters, predictions saved before continuations and input subset indices
are retained under ignored `runs/coupled_modes/`.

```sh
.venv/bin/python scripts/studies/run_coupled_modes.py --collection pilot256
.venv/bin/python scripts/studies/run_coupled_modes.py --collection fit4096 --train-size 4096
.venv/bin/python scripts/studies/run_coupled_modes_ordinary.py --collection ordinary4096
.venv/bin/python scripts/studies/run_coupled_modes_ordinary.py --checkpoint 2145 --collection tangent_epoch5 --tangent
MPLCONFIGDIR=/private/tmp/neural-competition-mpl .venv/bin/python scripts/studies/analyze_coupled_modes.py
.venv/bin/pytest -q
```

These exploratory producers overwrite their named collection. The current pilot
producer additionally records validation forecasts and metadata absent in the
initial executed version; its exact older source is retained by hash. Training
data come from the existing checksummed MNIST archives via `prepare_data`.

Read the [report](../../reports/coupled_modes.md),
[definition](../../theory/competition_definition.md), and
[chronology](../../theory/coupled_modes_protocol.md) before interpreting metrics.
