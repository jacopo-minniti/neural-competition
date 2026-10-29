# Delayed-demand options: evidence and reproduction

Main interpretation: [delayed demand and option recovery](../delayed_demand_option_recovery.md).
This continues the GD-derived response definition. Earlier notes and results
remain unchanged. All losses in the new toy experiments are half mean squared
errors, with the explicitly stated data weights.

## Records

| Folder or file | Contents |
|---|---|
| `history_audit.json` | Selected recomputations from prior raw MNIST and polarization records |
| `main/` | 12 fresh local-calibration seed/task pairs; response-matched reserve, ordinary, rate, and exposure branches |
| `affine/` | Six no-refinement controls sharing seed IDs with the first six main tasks |
| `analysis.json`, `summary.png` | Paired analyses of those experiments |
| `delayed/` | 64 fresh nested starts per task setting, first all-gates recovery-clock experiment |
| `delayed_confirmation/` | Independent 64-start bank, including the blind-set clock before continuation |
| `delayed_half_step/` | Same confirmation starts with half the numerical step; not an independent cohort |
| `delayed_analysis.json`, `delayed_summary.png` | Timing, losses, forecast coverage and false permanence comparisons |
| `delayed_intervention/` | Third independent 64-start bank, early/late reserve, exact clones and rate controls |
| `intervention_analysis.json` | Absolute and paired original-loss intervention outcomes |
| `retrospective_correction_trace.json` | Final model-discovered residual traced backward; explicitly uses future information |
| `verification.json` | Provenance, completeness and numerical consistency checks |

Protocols precede their associated outcomes. Each producer saves `source.py`
and its hash. Source snapshots identify executed versions; the current producer
files were later formatted, and an ambiguous variable was renamed. The initial
delayed snapshot predates the blind-set refinement, so it should be used to
reproduce that exact original version. Baseline construction functions are
imported from the existing study scripts; the delayed support is the same
contextual task documented in the prior theory.

JSONL files keep per-seed forecasts and outcomes, not just aggregate tables.
NPZs retain data/checkpoints/endpoints for sensor experiments, trajectories and
time-20/time-100/final states for delayed experiments, and intervention states.
No claim is made that every intermediate parameter state was saved. Multiple
widths, task settings, horizons, and branches reuse the same random draws;
analyses treat starts or seed/task pairs as the independent units.

## Recompute analyses and audits

Run from the repository root:

```sh
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/neural-competition-mpl
.venv/bin/python scripts/studies/audit_option_history.py
.venv/bin/python scripts/studies/analyze_option_transport_resilience.py
.venv/bin/python scripts/studies/analyze_delayed_option_recovery.py
.venv/bin/python scripts/studies/analyze_delayed_intervention.py
.venv/bin/python scripts/studies/audit_delayed_correction_trace.py
.venv/bin/python scripts/studies/verify_option_transport_resilience.py
.venv/bin/pytest -q tests/test_option_recovery.py
```

The sensor producers accept a new output directory:

```sh
.venv/bin/python scripts/studies/probe_option_transport_resilience.py --output /private/tmp/options-sensor-repeat --seeds 282901 282902 282903 282904 282905 282906 282907 282908 282909 282910 282911 282912
.venv/bin/python scripts/studies/probe_option_transport_resilience.py --output /private/tmp/options-affine-repeat --seeds 282901 282902 282903 282904 282905 282906 --affine
.venv/bin/python scripts/studies/probe_delayed_option_recovery.py --output /private/tmp/options-delayed-repeat --seed 282941 --count 64
.venv/bin/python scripts/studies/probe_delayed_option_recovery.py --output /private/tmp/options-delayed-half-repeat --seed 282941 --count 64 --dt .025
```

The intervention producer intentionally refuses to overwrite its fixed result
directory. To rerun it elsewhere, copy its saved source and change only its
`ROOT` output assignment while preserving imports and the recorded protocol.
For the original pre-refinement delayed experiment, likewise use its saved
source, keeping the study imports on `PYTHONPATH`.

## Scope

The response-matching identity and smooth transport factor are independently
checked against autograd. Recovery-clock mathematics applies to gradient flow
with a fixed metric; finite-step experiments and conservative floating-point
allowances are numerical audits, not rigorous interval certificates. Continuous
training uses quadrature and dense-quadrature evaluation, not an exact population
proof. The retrospective closing-step pulses are not prospective predictors or
long-run causal interventions.
