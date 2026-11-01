# Reproduction

Run from the repository root:

```sh
.venv/bin/python scripts/check_repository.py
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

The active producers and analyzers are listed in
[study scripts](../scripts/studies/README.md). The retained
[reserve](evidence/competition_reserve/README.md),
[bridge](evidence/reserve_bridge/README.md), and
[parity](evidence/natural_parity/README.md) evidence describe their protocols
and commands. Use fresh output directories: several producers deliberately
refuse to overwrite an existing protocol.

The main studies retain their compact analyses and, for reserve and the bridge,
their recorded forecasts, outcomes and historical source snapshots. The
parity analysis and local raw outcomes are retained; the producer can regenerate
its results. The selected learning-curve record retains the initial state,
forecast, and measured outcomes. Historical sources document what was executed;
the active source is maintained separately.

Large raw simulation arrays and ignored run directories from secondary studies
were removed during cleanup. Their summaries, protocols, and earlier code are
in [the archive](../archive/README.md). Tracked historical arrays remain in Git
history; ignored local runs can be regenerated but were not Git-tracked.
