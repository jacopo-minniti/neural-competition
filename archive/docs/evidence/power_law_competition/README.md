# Power-law refinement experiments

**Archive note:** Large per-run `.npz` arrays were removed from the working tree during cleanup. Summary records, protocols, source snapshots and figures remain. Previously tracked arrays are available in Git history at their original paths.

These records support the [general conditional model](../../model/power_law_competition.md).
No survival rates, width exponents, or forecast coefficients were fitted to
training outcomes. All datasets are synthetic. The richer version adds associated
co-occurring factors, continuous attributes, context jitter, nuisance inputs, and
dense mixing; it is not a real-world dataset validation.

The [summary figure](summary.png) is also available as a [vector export](summary.svg).

## Cohorts and provenance

| Directory / summary | Purpose | Independent seed banks | States |
|---|---|---:|---:|
| `confirmation/`, [uncentered summary](uncentered_summary.json) | Initial uncentered input stress test; frozen/grouped-force failure retained | 6 | 216 |
| `centered_confirmation/`, [primary summary](summary.json) | Centered 24-factor tasks, three initialization families, five forecast models | 6 fresh | 216 |
| `transfer32/`, [transfer summary](transfer32_summary.json) | 32 factors; alpha 1.4; association .35; unchanged predictor | 6 fresh | 48 |
| `pressure/`, [interventions](pressure/results.json) | Common/refinement/full/no-pressure branches from matched checkpoints | 6 fresh | 48 |

The first directory's historical name `confirmation` does not make it the final
positive result. Its negative results motivated centering and restoring the
existing coupled local model. Primary alpha is 1.15 and association .65. Transfer
uses fan-in Gaussian initialization; the primary cohort additionally tests small
Gaussian and fan-in uniform initialization.

Protocols and source snapshots are stored inside each probability cohort.
The saved protocols' legacy `prediction` string names the frozen-force baseline;
array prefixes and source snapshots identify the additional local-model forecasts.
The current producer also writes an explicit model list for future runs.
Each `*_forecast.npz` was written before the corresponding `*_actual.npz`.
Forecast and actual continuations use independent minibatch streams, 64 each.
Each initial state and dataset is reproducible from its seed bank and sources.
Files retain per-factor response samples, not large model checkpoints or complete
parameter trajectories. The earliest one-bank exploratory pilot ran in temporary
storage and was not used as confirmation evidence or to fit model coefficients.

Within a seed, widths, initializations, factors, checkpoints, and fork outcomes
are dependent. The primary uncertainty unit is the seed/dataset bank. Six banks
give only coarse uncertainty estimates; thousands of factor/fork records do not
turn into thousands of independent training experiments.

## Main quantitative findings

The coupled model evolves both layers and residuals within frozen gate cells,
refreshes once at the midpoint, and decodes physical endpoint response. At
16 steps, its expected-response RMSE relative to persistence is:

| Distribution | 24 factors | 32-factor transfer |
|---|---:|---:|
| Prototype | .485 [.395,.553] | .244 [.177,.346] |
| Associated continuous | .205 [.124,.303] | .105 [.095,.111] |

Brackets are seed-bootstrap 95% intervals. Lower than one improves on predicting
no change. These are response forecasts, not forecasts of final population loss.
The same model beats persistence in all three initialization families separately.
One-step models represent the same exact conditional update law; differences in
their estimated scores arise from independent finite Monte Carlo samples.

The primary 16-step Brier skills for a 10% relative response decrease are .400
and .408, compared with a constant predictor using the evaluated group's own
prevalence. This is an optimistic constant baseline, not a separately deployable
prediction rule. Resolution-crossing probabilities show a remaining failure:
the prototype's below-.1 crossing rate is predicted as .492% versus 1.009%
observed. The associated result is 2.315% versus 2.307%.

[Width and reopening analysis](mechanism_summary.json) retains paired width
contrasts and their six-seed uncertainty. In the associated late checkpoints,
16-step 10% weakening falls from .506% at width 50 to .181% at width 100; the
prediction is .582% to .391%. The paired interval excludes zero there, but not
at checkpoint 200. Late prototype width ordering is slightly reversed and
uncertain. Final loss ordering is also not uniformly favorable to width.

Of initially available factor/fork records that fall below .1 at step four,
76.0%/55.8% reopen by step sixteen in prototype/associated tasks. This is a
conditional endpoint statistic with correlated records, not a continuous-time
first-passage estimate. It rejects treating closure as absorbing in these tasks.

The [pressure interventions](pressure/results.json) predict both signs: common
pressure weakens prototype capability slightly and creates associated-task
capability on average. Refinement-only pressure has the opposite mean signs.
All isolated pressure arms still improve mean later loss over no pressure, so
capability weakening is not sufficient evidence of harmful competition.
[Step sensitivity](pressure/step_sensitivity.json) preserves mean signs but finds
material event-level uncertainty for the small prototype/common effect.

## Reproduction

Use a new output directory when producing a cohort; the runner refuses to
overwrite an existing protocol. Original commands, aside from log redirection:

```sh
.venv/bin/python scripts/studies/run_power_law_competition.py --output <new-uncentered-dir> --uncentered --seeds 6 --seed-bank 84100 --samples 64 --checkpoints 0 200 2000
.venv/bin/python scripts/studies/run_power_law_competition.py --output <new-primary-dir> --seeds 6 --seed-bank 95100 --samples 64 --checkpoints 0 200 2000
.venv/bin/python scripts/studies/run_power_law_competition.py --output <new-transfer-dir> --seeds 6 --seed-bank 117100 --samples 64 --checkpoints 200 2000 --initializations fanin_gaussian --factors 32 --alpha 1.4 --association .35 --widths 66 132
.venv/bin/python scripts/studies/analyze_power_law_competition.py <cohort-dir> --output <summary-file>
.venv/bin/python scripts/studies/analyze_power_law_mechanism.py
.venv/bin/python scripts/studies/probe_power_law_pressure.py
.venv/bin/python scripts/studies/audit_power_law_pressure_steps.py
MPLCONFIGDIR=/private/tmp/neural-competition-mpl .venv/bin/python scripts/studies/plot_power_law_competition.py
.venv/bin/pytest -q tests/test_power_law_competition.py
```

The current runner adds the later linear/bilinear ablations to an uncentered
rerun. For exact reproduction of the original uncentered record, use that
cohort's source snapshot. Fixed-path pressure and analysis scripts reproduce
their existing deterministic records; write to a separate workspace to preserve
original timestamps when desired.

The independent tests verify capacity over bounded continuous and co-occurring
inputs, power-law marginals and association covariance, response susceptibility
against finite differences, exact aggregation of checkpoint sample gradients,
first-step local-model equivalence, and the shared-count binomial special case.
