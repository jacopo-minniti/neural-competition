# Toy resource competition: executed evidence

29 October 2026. See the [theory and interpretation](../heterogeneous_option_resources.md)
and [maintained context](../../research_context.md).

## Records and chronology

| Collection | What was executed |
|---|---|
| `pilot/` | 32 seeds, RNG 295001, six families, two widths, four pressure arms, time 500 |
| `events_pilot/` | Every gate-changing ordinary update through time 100; same-state common/refinement counterfactuals for structural coverage changes above .03 |
| `recycle_pilot/` | Four random direction replacements per eligible time-20 state; control shares exact readout redistribution; continuation to 500 |
| `confirmation/` | Fresh 64-seed bank 295101, same six families and pressure arms, time 2000 |
| `events_confirmation/` | The same event audit on the fresh ordinary paths |
| `recycle_confirmation/` | Same recycling construction on fresh time-20 states, continued to 2000 |
| `step_check/` | Same fresh banks for discrete, continuous and separate-attribute families; ordinary training at dt=.025 to 2000 |
| `events_step_check/` | Event audit at the smaller step; supports width contrast, not identical event attribution |
| `initial/` | 4,096 independent initialization banks, RNG 295501; six widths and four task families |
| `frozen_recycling/` | Post-outcome replay of recycling starts; exact frozen-flow and stable frozen-GD response at the remaining 1,980-time-unit budget |
| `analysis_sources/` | Executed analyzer and its local mathematical dependencies |

The initial pilot motivated the broader event and recycling audits. Thresholds
are descriptive choices: structural refinement coverage .95, substantial error
in training-measure loss above .05 times initial centered-refinement energy. Sensitivities .90/.99 and
.01/.10 respectively are in [summary.json](summary.json). These are not independent
prospective forecasts or a preregistered mediation analysis. Producers save source
snapshots and protocol hashes before training; the frozen-response audit is
explicitly retrospective. Runs within a task are nested by width; task families
reuse banks where input dimensions match. Do not pool them as independent cohorts.

`trajectory.jsonl` contains per-seed loss, pressure energy/power, actual residual
loading, response strengths, structural coverage, readout floor and a diversity
heuristic. Structural coverage uses normalized columns of the actual whitened
Jacobian, so tiny initialization is not mislabeled structural absence. Options
remain defined by the unnormalized, optimizer-scaled response. In continuous
tasks, exact-full-span coverage is too strict to identify substantial obstruction.

`events.jsonl` records every large support change, including repeated boundary
oscillations; the summary uses the **first loss event per seed**. The thousands
of event rows are not independent samples. Classification C means common-only
causes a drop >.03 and refinement-only does not; R reverses this; CR means both;
“joint” means only the full update crosses this drop threshold. These are
one-update counterfactuals, not whole-run removal effects.

`outcomes.jsonl` in recycling collections contains the start, time 100 and final
observations. Ineligible checkpoints are retained in `ineligible.jsonl`. The
replacement direction is random, not chosen by eventual improvement or target
alignment. The control and treatment share post-redistribution predictions and
kernel trace; the original model need not share that kernel. For continuous
inputs, preservation is verified on training and dense grids, not certified
between grid points. Bootstrap intervals average the four direction trials
within each seed before resampling seeds (4,000 resamples, RNG 295701).

The old blind-set bound is recomputed at time 100 using its general algebraic
quantities only. Its specialized old task contrasts are not used for new families.
It certifies no fresh ordinary discrete/continuous failures through time 2000;
this lack of coverage is retained. The bound is a gradient-flow theorem, not a
certified Euler-GD guarantee.

## Principal findings and retained negatives

- Initial coverage improves with width, but missing initialization support does
  not explain the fresh discrete width gap: all runs start covered, then 15/64
  narrow runs close and three recover; wide runs have no observed closure.
- Common pressure directly causes some first losses of support, especially in
  the separate-attribute family. Yet removing it throughout early learning often
  hurts: it also supplies useful feature creation.
- Same-width recycling reduces loss in eligible discrete-width-4 and
  separate-attribute-width-8 subsets while matching predictions and trace(K).
  Eligibility is restrictive: no continuous-width-4 and only one
  separate-attribute-width-4 state qualify. The latter cannot support inference
  about a population effect.
- The two-refinement pilot's absence of terminal obstruction does not replicate
  perfectly: two fresh narrow runs fail. Seven of eight initially covered narrow
  paths that close recover. Its recycling mean benefit has an interval spanning
  zero and includes one harmful trial.
- Centering away the shared target does not remove every width gap. Most first
  losses there are refinement-induced; they are a control against attributing
  every failure to common pressure.
- Static spectral breadth/strength and an inverse-participation implementation
  diversity measure do not consistently explain recycling effects. Coverage
  changes track the discrete gains, but recovery and fine continuous residuals
  prevent promoting coverage alone into a universal mediator.
- Even the exact **budget-matched frozen** response misses recruitment. On
  discrete width 4 it predicts mean loss reduction .001503 versus observed
  .012938; the seed-level correlation is −.019. On width 8 it predicts a .001137
  effect that disappears under actual learning. This motivates retaining the
  existing feature–coefficient dynamics, not changing the response definition.

The [summary](summary.json) includes all cohorts' protocols, confirmation statistics,
step checks, seed intervals, cutoff sensitivities and descriptive correlations.
The [frozen audit](frozen_recycling/audit.json) contains every replayed trial and
its exact frozen-flow/GD comparison. The two figures are
[natural transitions](natural_transitions.png) and [interventions](interventions.png).

## Reproduction

Run from the repository root using `.venv/bin/python`. Producers refuse to
overwrite a protocol. For a full rerun, use a new parent directory and substitute
it consistently below; the analyzer's `BASE` constant must then point to it.
Existing completion markers are evidence, not a fresh execution.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/neural-competition-mpl
.venv/bin/python scripts/studies/run_toy_resource_competition.py --output docs/theory/toy_resource_competition/pilot --count 32 --seed 295001 --end 500
.venv/bin/python scripts/studies/audit_toy_option_events.py --input docs/theory/toy_resource_competition/pilot --output docs/theory/toy_resource_competition/events_pilot
.venv/bin/python scripts/studies/recycle_toy_resources.py --input docs/theory/toy_resource_competition/pilot --output docs/theory/toy_resource_competition/recycle_pilot --end 500
.venv/bin/python scripts/studies/run_toy_resource_competition.py --output docs/theory/toy_resource_competition/confirmation --count 64 --seed 295101 --end 2000
.venv/bin/python scripts/studies/audit_toy_option_events.py --input docs/theory/toy_resource_competition/confirmation --output docs/theory/toy_resource_competition/events_confirmation
.venv/bin/python scripts/studies/recycle_toy_resources.py --input docs/theory/toy_resource_competition/confirmation --output docs/theory/toy_resource_competition/recycle_confirmation --end 2000
.venv/bin/python scripts/studies/run_toy_resource_competition.py --output docs/theory/toy_resource_competition/step_check --count 64 --seed 295101 --end 2000 --dt .025 --cases discrete continuous separate_attribute --arms ordinary
.venv/bin/python scripts/studies/audit_toy_option_events.py --input docs/theory/toy_resource_competition/step_check --output docs/theory/toy_resource_competition/events_step_check --cases discrete continuous separate_attribute
.venv/bin/python scripts/studies/probe_initial_option_coverage.py --output docs/theory/toy_resource_competition/initial --count 4096 --seed 295501
.venv/bin/python scripts/studies/audit_recycled_frozen_response.py
.venv/bin/python scripts/studies/analyze_toy_resource_competition.py
.venv/bin/pytest -q tests/test_toy_resources.py
```

The frozen-response auditor also has a `BASE` constant. The analyzer verifies
producer source snapshots, input-state hashes and completion markers, reconstructs
seed-level summaries, and regenerates figures. Tests independently verify the
weighted pressure projection, preservation of the older task definitions,
autodiff Jacobian and optimizer updates, smooth susceptibility/response transport,
recycling invariants, and the finite-response progress inequality.
