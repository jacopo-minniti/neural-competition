# Sign-survival evidence

See the [report](../../reports/sign_survival.md),
[prospective design and follow-up chronology](../../theory/sign_survival_protocol.md),
and [spectral derivation](../../theory/signed_capacity.md).

- `trajectories.csv.gz`: every recorded state-level metric, all cohorts/arms/seeds.
- `endpoints.csv`: all 1845 endpoint rows, including repeated consistency arms.
- `summary.csv`: cohort means, medians, active/zero signs and intervention budgets.
- `contrasts.csv`: paired improvements, 5000-draw seed-bootstrap intervals and wins.
  Improvement is baseline minus intervention in target-variance MSE percentage
  points; raw loss columns elsewhere contain half-MSE.
- `population_certificates.csv`: exact Gaussian population MSE, second-Hermite
  loss, sign-count lower bound and remaining orthogonal components at endpoints.
- `crossings.csv`: post-result source-minibatch attribution during the first
  1000 ordinary balanced updates, per seed. Events are not independent replicates.
- `audit.csv`: endpoint reconstruction and divergence counts for all 21 bundles.
- `provenance.json`, `analysis_provenance.json`, `crossings_provenance.json`,
  `sources/`: executed arguments and content-addressed source snapshots.

Local raw bundles and saved states live under ignored `runs/sign_survival/`.
The versioned tables and snapshots are sufficient to inspect reported values;
recomputing the audits requires those bundles or rerunning the producer.

## Reproduce

The main source snapshot predates the movement-control option. Completed matching
collections are skipped; changed arguments require a different collection name.

```sh
.venv/bin/python scripts/studies/run_sign_survival.py
.venv/bin/python scripts/studies/run_sign_survival.py --collection movement_control --movement-control
.venv/bin/python scripts/studies/run_sign_survival.py --collection angle60 --degrees 60 --seeds 10
.venv/bin/python scripts/studies/run_sign_survival.py --collection balanced_target --ratio -1 --seeds 10
.venv/bin/python scripts/studies/run_sign_survival.py --collection data2 --data-seed 9037 --seeds 10
.venv/bin/python scripts/studies/run_sign_survival.py --collection low_rate --rate-factor .3 --seeds 10
.venv/bin/python scripts/studies/run_sign_survival.py --collection longer --steps 100000 --seeds 5
MPLCONFIGDIR=/private/tmp/neural-competition-mpl .venv/bin/python scripts/studies/analyze_sign_survival.py
.venv/bin/python scripts/studies/trace_sign_crossings.py
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

Main seeds 200–219; controls share 200–209 and longer reruns share 200–204.
Each optimizer has 20 main initialization replicates, not hundreds of independent
observations. The movement-control rerun repeats original arms exactly; only
its damping arm is a new intervention. The protected set's initial sign counts
are deliberately compatible with the known teacher.
