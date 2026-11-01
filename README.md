# Neural competition: protected reserve

This project studies how learning one part of a task changes the corrections a
network can make later, and whether greater width preserves those corrections.
The current model separates residual demand from response capability, identifies
response protected from a specified dominant pressure, and predicts response
after a declared amount of dominant learning.

[The report](report.tex) is the main entry point. It develops the theory and
presents the selected learning-curve example, the co-occurrence study and the
parity test. The main evidence is conditional: the selected loss example is not
a prevalence estimate; the completion forecast is approximate under full
training; and parity exposes a prediction failure. A causal account of a
width-dependent loss advantage remains an open goal.

## Explore the project

- [Research objectives and evidence standards](OBJECTIVES.md)
- [Protected-reserve model and its assumptions](docs/model/competition_reserve.md)
- [Co-occurrence, parity and the next tests](docs/model/reserve_relaxation.md)
- [Evidence and reproduction](docs/README.md)
- [Active study scripts](scripts/studies/README.md)
- [Earlier models and supporting evidence](archive/README.md)

## Validate

```sh
.venv/bin/python scripts/check_repository.py
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```
