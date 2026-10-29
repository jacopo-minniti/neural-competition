# Functional implementation-state study: evidence and reproduction

29 October 2026. [Theory and conclusions](../functional_implementation_state.md);
[maintained research context](../../research_context.md).

## Collections

| Directory | Executed study |
|---|---|
| `pilot/` | 24 seeds, bank 296001, twelve N/frequency/attribute configurations, two sufficient widths, time 500 |
| `confirmation/` | Fresh 48 seeds, bank 296101, same twelve configurations, two widths, time 4000, dt=.05 |
| `wider/` | Same fresh nested banks, width 40 on N=4 ratios .25/.1 with separate attributes |
| `step_check/` | Same banks at dt=.025: N=1 separate attributes and N=4 ratios .25/.1 separate attributes |
| `finer_step/` | N=4 ratio .25, separate attributes, dt=.0125; apparent coarse-step obstruction disappears |
| `option_matched/` | 48 artificial matched pairs per task, bank 296201: identical function/full K, different signed implementation moments; short response probes and training to 2000 |
| `boundary_probe/` | Natural-state forecasts on first 12 seeds at times 2/20/100; moment-preserving mixing at time 20 on all eligible fresh states, continued to 2000 |
| `early_moment_mix/` | Same construction at time 2, added after the mostly ineffective time-20 intervention; retained improvements and harms |

Every producer saves a protocol and exact local source snapshots; upstream NPZ
hashes are included in the natural-state intervention protocols. Completed
collections have `complete.json`. Current source versions may differ from older
snapshots; historical experiments are not silently relabeled as new executions.
The analyzer verifies saved source and input hashes before making summaries.

The mathematical moment closure was derived before evaluating its natural-path
accuracy. The early mixing follow-up and finer-step check were motivated by prior
results. These are mechanism studies, not independently preregistered endpoint
forecasts. Matching, algebraic sufficiency, descriptive relationships, and
intervention effects have different evidential roles.

## Raw data and definitions

`trajectory.jsonl` stores one row per seed/time, with arrays indexed by refinement:
structural coverage, finite-response strengths at τ=20/1000, normalized conditional
contrast error, actual residual loading, and context losses. It also records
joint-subspace minimum strength and total loss. Checkpoint NPZs retain raw network
states for exact replay. All contexts are present at every training step.

The target is realizable at m=2+2N; larger widths are 2m and, in selected cases,
4m. The generator asserts an explicit representation witness. Total refinement
frequency is .2; q_j follows a geometric hierarchy. Signs of refinement targets
alternate. The two attribute-sharing settings do not assign neuronal roles and
are not assumed to prescribe actual implementation overlap. The analyzer computes
gate-resolved participation overlap separately; it remains descriptive.

The primary transition cutoff is structural coverage .95. New demanded closures
are counted only before first mastery, defined by relative conditional-gap error
≤.1. Recovery means subsequent structural coverage crosses back above the cutoff;
it does not imply correction is complete. Mastery and recovery are reported
separately. Counts depend on checkpoint spacing. Sensitivity cutoffs are retained
in [summary.json](summary.json), as are raw continuous means and seed-bootstrap
intervals. Components, widths, budgets and interventions share seeds and are not
independent samples. No across-component pooled significance is claimed.

`option_matched/outcomes.jsonl` includes full observed and moment-predicted
response matrices at short horizons. Equal initial predictions/full K are
asserted from saved initial states. Readout sign reversal changes signed hidden
contributions, even when they sum to zero. Long-run differences have both signs;
this is not a rescue policy.

`boundary_probe/forecasts.jsonl` compares fixed-gate joint moments, frozen K, and
an exact functional particle representation. The latter has no incoming weights:
it updates weighted preactivation functions using the known input Gram geometry.
It is an equivalence check, not independent predictive validation. At time 100,
7/72 frozen-K baselines have dt·lambda_max(K)>2; their Euler forecast can diverge.
The analyzer retains this limitation rather than using the resulting large
baseline errors to advertise a compact model.

`mixed_outcomes.jsonl` compares ordinary and same-gate orthogonal pair rotations.
The rotations preserve every joint moment, prediction, full K, parameter norm,
width and optimizer at the intervention. No future outcome guides selection.
Ineligible states are recorded. The control continuation is checked against the
saved original endpoint. The early follow-up additionally records the first gate
trajectory divergence between paired networks. Bootstrap intervals resample
original seeds (4,000 draws, RNG 296501). They describe eligible subsets only.

## Results that should not be lost

- Current options alone do not determine response evolution. Signed
  gate-conditioned implementation moments can distinguish matched-option states.
- Full moments still fail to determine boundary flux. An explicit two-point
  counterexample and natural-state moment-preserving interventions demonstrate
  this; an exact angular feature-shape measure retains the missing information.
- Fixed-gate moment forecasts are essentially exact without crossings but are
  not a generally adequate natural-path closure. Their storage is not necessarily
  smaller than the original model in these small networks.
- For N=4, ratio .1, separate attributes, demanded closures of the rarest
  refinement occur in 27/48, 9/48, and 0/48 runs at widths 10/20/40. All observed
  closures eventually recover structurally, yet only 30/48, 39/48, and 48/48
  master that refinement by time 4000.
- Shared task attributes produce more implementation participation overlap and
  often easier learning. Overlap alone is not a competition score.
- Moment-preserving mixing is usually ineffective late and has both signs early.
  Its role is identifying missing state information, not improving training.
- An apparent N=4 ratio-.25 persistent width gap vanishes under step refinement:
  narrow mean loss .001061 → .00002324 → 5.94e−8. Do not cite the coarse result as
  robust continuous-time obstruction.
- Earlier continuous recovery, failed copy/preservation interventions and weak
  recovery certificates remain valid historical evidence. MNIST was not resumed.

## Reproduction

From the repository root, use a fresh output parent to rerun producers. They
refuse to overwrite existing protocols. Update the analyzer's `BASE` constant for
a new parent. Existing completion markers are not evidence of a fresh rerun.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/neural-competition-mpl
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --output docs/theory/implementation_state/pilot --count 24 --seed 296001 --end 500
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --output docs/theory/implementation_state/confirmation --count 48 --seed 296101 --end 4000
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --output docs/theory/implementation_state/wider --count 48 --seed 296101 --end 4000 --factors 4 --only n4_r0.25_o0 n4_r0.1_o0
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --output docs/theory/implementation_state/step_check --count 48 --seed 296101 --end 4000 --dt .025 --only n1_r1_o0 n4_r0.25_o0 n4_r0.1_o0
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --output docs/theory/implementation_state/finer_step --count 48 --seed 296101 --end 4000 --dt .0125 --only n4_r0.25_o0
.venv/bin/python scripts/studies/probe_option_matched_implementations.py --output docs/theory/implementation_state/option_matched
.venv/bin/python scripts/studies/probe_moment_boundary_closure.py --input docs/theory/implementation_state/confirmation --output docs/theory/implementation_state/boundary_probe
.venv/bin/python scripts/studies/probe_moment_boundary_closure.py --input docs/theory/implementation_state/confirmation --output docs/theory/implementation_state/early_moment_mix --mix-time 2 --skip-forecasts
.venv/bin/python scripts/studies/analyze_implementation_state.py
.venv/bin/pytest -q tests/test_implementation_moments.py
```

The tests check the hierarchy witness and frequency law, moment reconstruction,
exact fixed-gate updates, input rotations and unit permutations, moment-matching
invariants, the boundary-flux counterexample, and inert identical duplication
under the declared mean-field metric. Figures are
[hierarchy budgets](hierarchy_budget.png) and
[resource interventions](resource_interventions.png).
