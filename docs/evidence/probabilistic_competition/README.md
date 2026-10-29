# Analytic probability-law audit

This is a consistency check for the
[probabilistic competition derivation](../../model/probabilistic_competition.md),
not a fitted model or a validation of natural-width failure probabilities.

The [raw record](audit.json) retains all six probability checks and both step
sizes of the deterministic trap checks. Reproduce with:

```sh
.venv/bin/python scripts/studies/audit_probabilistic_option_transport.py
MPLCONFIGDIR=/private/tmp/neural-competition-mpl .venv/bin/python scripts/studies/plot_probabilistic_option_transport.py
.venv/bin/pytest -q tests/test_probabilistic_option_transport.py
```

The tests independently compare the Gaussian orthant formula to a polar integral,
differentiate the exact flow to recover the gradient field, check absorption at
the common-gate boundary and composition of the flow, and compare the claimed
null event to the full weighted prediction Jacobian.

The audit draws 400,000 independent Gaussian blocks, then transports them by the
closed-form solution. It compares single-time and joint blindness to the derived
probabilities at two geometries and three pressure exposures. The same draws are
reused across conditions, so errors across rows are correlated. The largest
single-time discrepancy is approximately 3.72 binomial standard errors. The
separate deterministic integral checks agree to the tests' 2e-13 tolerance.
The network probabilities in the record are analytic predictions from iid
blocks, not observed frequencies of trained-network failures.

The [figure](pressure_laws.png) and [vector export](pressure_laws.svg) plot these
analytic laws. Conditional reopening compares width-specific initially closed
subsets, not a matched-checkpoint causal recovery experiment.

The explicit four-unit construction starts with full Jacobian row rank on all
three prototypes. Common exposure 1.5 closes both rare gates in every unit, with
an inactive margin approximately .31469 and positive readouts. The common branch
reaches this exposure at physical time approximately .66989 for q=.01.

Two full-task runs **from the same initial state, with the rare targets present
throughout**, at q=.01 and steps approximately .002/.001 enter the strict
invariant-trap inequalities by that time. Their rare inactive margins are
.31453/.31473. This is one deliberately constructed example, not a random seed
cohort or an estimated q threshold. The analytic robustness argument establishes
an open set of nearby initial states and sufficiently small q, without evaluating
its Gaussian mass or its admissible q range.

As a separate continuation check, release the exact common-branch checkpoint to
the full task with q=.1. Both steps .01/.005 keep all rare gates inactive through
time 50. Losses are .05002095/.05002099, above the proved .05 floor. The theorem
concerns gradient flow; these Euler checks are numerical consistency evidence.

No coefficients, exposure, geometry, or width exponent were fitted to endpoint
losses. No MNIST runs or existing empirical evidence were changed.
