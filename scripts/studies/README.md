# Protection-model studies

Run scripts from the repository root; producers that save protocols require a
new output directory. Use `--help` for the exact options.

- `competition_reserve.py`: protected response, dominant-flow transport,
  completion curves, and gradient-direction diagnostics.
- `run_competition_reserve.py`, `analyze_competition_reserve.py`, and
  `plot_competition_reserve.py`: reserve confirmation and summary.
- `probe_reserve_learning_curves.py`: the selected three-pattern example in
  the report. Its measured loss curves are not full loss predictions.
- `reserve_bridge.py`, `run_reserve_bridge.py`, and `analyze_reserve_bridge.py`:
  the co-occurring-factor study and analysis.
- `run_natural_parity.py` and `analyze_natural_parity.py`: the parity study.

See [reproduction](../../docs/reproduction.md) for the validation commands and
[evidence](../../docs/README.md) for retained protocols, forecasts and results.
The [archive](../../archive/README.md) contains earlier alternative pipelines.
