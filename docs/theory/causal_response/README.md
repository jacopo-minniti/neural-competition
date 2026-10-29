# Finite-budget competition: evidence and reproduction

Read the [theory and conclusions](../finite_budget_competition.md) and
[current framework](../../research_context.md). This continues the existing
spectral response model with budget-matched correction tests and one midpoint
activation-mask refresh. Negative causal results are central to the conclusions.

## Chronology and cohorts

| Record | Purpose | Status |
|---|---|---|
| `pilot` | Common-output correction and ordinary recovery; old concentrated copies | Exploratory cases chosen from previous outcomes; ineffective preservation retained |
| `signed_pilot` | General matched-copy construction with coefficients (2,-1); first midpoint refresh | Same three selected cases; immediate rescue but later reversals |
| `quadratic_pilot` | Add the deterministic-budget spectral filter | Same actual branches, verified identical to signed pilot |
| `closure_audit` | Frozen quadratic, tangent CE, and bilinear CE recovery from actual post-pressure states | Retrospective explanation; not prospective prediction |
| `checkpoints` | Ordinary MNIST SGD, seeds 284101–284110, widths 32/128, checkpoints 20/100 | Fresh training starts; no manipulated labels |
| `validation_a`, `validation_b` | Two halves of one locked ten-seed test at checkpoint 100, covering all ten focal digits | Original versus refreshed model; early signed copies; all outcomes retained |
| `timing_audit` | Add copies only after pressure, for the three seed/class pairs containing forecast-selected harm | Follow-up selected after broad outcomes; both widths retained |
| `class8_checkpoints` | Ordinary SGD, seeds 284201–284208 | Eight additional fresh starts |
| `class8_replication` | Focused digit-8 test selected after the only broad-cohort 20-step deficit | No persistent deficits replicate; model predicts their absence |
| `class8_timing` | Late-only timing controls for every focused-replication state | Designed before inspecting focused outcomes |

Selection and locked predictions are recorded in [pilot_design.json](pilot_design.json),
[validation_plan.json](validation_plan.json), [class8_plan.json](class8_plan.json),
and [timing_plan.json](timing_plan.json). Case files give all seed/class/checkpoint
choices. The first pilot cases are old seeds 283104/20/class4,
283106/100/class6, and 283101/100/class1; they are not independent confirmations.

Producer directories contain protocol/source snapshots, raw forecasts and
outcomes (or retrospective results), and completion markers. Forecasts precede
the corresponding actual branches. Source and checkpoint hashes and the MNIST
data hash are checked where recorded. Training-source manifests are retained
alongside the actual checkpoint files. No prior result directory was overwritten.

[analysis.json](analysis.json) contains scores, seed-level uncertainty, individual
fresh-state contrasts, and timing controls. [summary.png](summary.png) is generated
from the broad locked test. [correction_windows.png](correction_windows.png)
shows forecast-selected risk windows and the failed focused rescue replication.
[environment.json](environment.json) records the post-run reproduction environment. Bootstrap samples keep all measurements from a seed
together. The focused cohort replicates a class across seeds; the broad cohort
has one assigned class per seed and cannot separate class from seed effects.

The common-output probe uses a fixed additive output function and soft background
targets. It deliberately holds initial predictions, error and curvature equal
across histories, preserving their different Jacobians. It is not ordinary MNIST
training or a deployable intervention. Original-label recovery outcomes are kept
separate. The copy construction likewise changes parameterization and optimizer
metric; it is a causal control, not ordinary widening.

## Reproduce

Use a fresh output directory for every run. Producers refuse to overwrite a
protocol. All commands run from the repository root and require prepared local
MNIST data. Exact executed versions are saved per producer; current scripts add
optional ablations while preserving the old records.

```bash
# Fresh ordinary checkpoints used by the broad test.
.venv/bin/python scripts/studies/train_response_checkpoints.py \
  --output tmp/causal-checkpoints --seed 284101 --count 10

# One broad cohort. The retained run was split into two computational halves.
.venv/bin/python scripts/studies/probe_causal_response.py \
  --input tmp/causal-checkpoints --output tmp/causal-validation \
  --cases docs/theory/causal_response/validation_cases.json \
  --signed --compact --quadratic --forecast-refresh 0 10

# Late-only control for the prospectively selected immediate-risk states.
.venv/bin/python scripts/studies/probe_late_preservation.py \
  --input tmp/causal-checkpoints --output tmp/causal-timing \
  --cases docs/theory/causal_response/timing_cases.json

# Focused replication: fresh seeds, same untuned predictor and intervention.
.venv/bin/python scripts/studies/train_response_checkpoints.py \
  --output tmp/class8-checkpoints --seed 284201 --count 8
.venv/bin/python scripts/studies/probe_causal_response.py \
  --input tmp/class8-checkpoints --output tmp/class8-replication \
  --cases docs/theory/causal_response/class8_cases.json \
  --signed --compact --forecast-refresh 10
.venv/bin/python scripts/studies/probe_late_preservation.py \
  --input tmp/class8-checkpoints --output tmp/class8-timing \
  --cases docs/theory/causal_response/class8_cases.json

# Selected-case explanatory audit, using the existing confirmation checkpoints.
.venv/bin/python scripts/studies/audit_correction_closure.py --output tmp/closure-audit

# Verify retained records and regenerate figures/scores.
MPLCONFIGDIR=/tmp/neural-mpl .venv/bin/python scripts/studies/analyze_causal_response.py
.venv/bin/pytest -q tests/test_causal_response.py
```

The new tests independently check signed-copy full-kernel matching and feature
velocities, fixed-offset gradients, infinitesimal correction gain, exact one-step
refresh behavior, and the matrix-free deterministic-budget gain against a dense
spectral calculation. A two-mode counterexample verifies that discounted-value
ordering cannot order all finite-budget gains.
