# Prediction targets and experiment agenda

The current work follows the [reserve-relaxation ladder](model/reserve_relaxation.md).
The [first bridge study](evidence/reserve_bridge/README.md) preserves the earlier
aggregate response-weakening prediction while introducing co-occurring factors.
The first primary target is the fraction and frequency-weighted mass of corrections
losing substantial response over natural learning phases, versus width. Duration,
recovery, and consequential residual error follow. The direct parity exploration
is retained as diagnostic evidence; accurate future-SGD simulation is outside
the current target. The targets below retain their original evidence standards.

Status: **research objectives, with the executed subset linked above**. The core theory and
reserve results are described in [report.tex](../report.tex) and the
[reserve chapter](model/competition_reserve.md); no unrun experiment below is a
result. The [dataset design](dataset_design.md) supplies controlled settings.

## Definitions for all targets

Fix a train/validation distribution and a separate reference audit, a checkpoint,
a response horizon or discount scale, a reference residual or fixed contrasts,
a follow-up budget, and thresholds before inspecting outcomes. The current
spectral definition of an option uses the nonzero residual projection onto an
eigenspace. A factor contrast is a useful *probe*, not necessarily one spectral
option. Eigenvectors may rotate, swap or be degenerate: track fixed probes,
subspaces or projectors, not a sorted eigenvector index. If exact modal option
counts are unstable, report subspace-level quantities instead.

Distinguish exact zero response with demand remaining (structural closure),
crossing below a positive practical threshold, and a relative response decline.
Record demand satisfaction separately. Define openings and reopenings analogously,
including initial availability, endpoint versus first-passage events, and time
resolution. A positive directional response does not establish that an entire
contrast is correctable. Report frequency-weighted and balanced results.

## 1. Option dynamics and attributes

- Conditional on a checkpoint and a stated future sampling or pressure law,
  forecast \(P(\text{closure in }[t,t+H])\), opening and reopening probabilities,
  and expected counts \(E[N]=\sum_r P(I_r=1)\). This sum requires no independence;
  probabilities of any closure, simultaneous closures and count variance need the
  joint law. Validate calibration, event rates, Brier/log scores where appropriate,
  ranking, and clustered uncertainty against prevalence, current response and
  persistence baselines. A completion curve is a deterministic scenario unless
  a future-pressure distribution is supplied and independently validated.
- Predict response-strength trajectories or distributions, time below threshold,
  and time to recovery for fixed probes. Compare estimated reserve floors and
  completion curves with actual full-training paths at both matched progress and
  matched steps. Allow nonmonotonic paths. Use confidence intervals and count
  small-sample tail events explicitly.
- Describe demand-weighted response spectrum, low-response residual mass,
  protected fraction, signed pressure effects and update/route correlations.
  These are explanatory intermediate measures, not by themselves endpoint claims.
- **Plot:** over steps and functional progress, show predicted and measured
  response, demand, chosen thresholds, closures and reopenings for predeclared
  probes, plus reliability plots for event forecasts.

## 2. Options and loss

- At each checkpoint compute discounted value \(V_\tau(e)=e^T C_\tau e/(2n)\)
  and a finite-horizon frozen quadratic prediction with its explicit horizon and
  learning rate. Plot predicted versus observed loss reduction and residual
  components on held-out data. An exponentially discounted value is not itself
  the loss reduction after exactly \(\tau\) steps. Include persistence, simple
  loss-trend and frozen-kernel baselines.
- Test whether deviations are explained by response transport and demand
  rotation; quantify gate changes, kernel drift, actual-vs-predicted residuals,
  and local forecast error as horizon increases. A local formula can be extended
  across several steps only where measured drift and/or a derived bound makes the
  accumulated error acceptable. Predeclare the useful horizon and tolerance.
- Predict conditional loss after a fixed remaining budget from initial state,
  demands and predicted option dynamics; separate a model's own moving residual
  from a fixed reference residual. Calibration must be assessed on new seed banks.
  Do not infer that closing a probe implies a loss floor without the necessary
  geometric and dynamical assumptions.
- Test causality with same-checkpoint branches that alter pressure or preserve a
  response, alongside controls for immediate prediction, residual and update
  magnitude. Then restore identical full training with matched data and budget.
  Assess both response and later loss, retain harmful and null effects. If an
  intervention also changes other capabilities, report that it is not a pure
  mediation test. Compare matched-demand states where possible; association alone
  cannot establish that a response caused a loss gap.

## 3. Width and loss

- Compare widths at fixed processed examples/steps and separately matched
  functional progress, using paired dataset/seed banks and independent
  confirmation. Report initial response coverage, response loss, reopening,
  demand-weighted value and held-out loss. Account for normalization,
  architecture, optimizer scaling, representational sufficiency and compute.
- For a fixed pressure and suitable independence assumptions, the earlier
  Gaussian special case has complete-loss probability \(b(s)^m\). Investigate
  whether correlated shared updates permit an effective dependence-aware bound
  or law; measure dependence and test alternatives. Do not fit an exponent and
  call it a derivation, or claim fixed-progress laws automatically hold at fixed
  training time. No general scaling-law objective is implied.
- Estimate a width-associated loss difference and ask how much is explained or
  causally affected by maintaining demanded response. Compare initial coverage,
  survival and recovery; width changes all three, plus gradients and trajectory.
  A natural width comparison and a response-preserving intervention answer
  different questions. Clearly report uncertainty and cases where wider loses.

## Flagship plot and validation protocol

**Desired plot:** x-axis actual training steps; y-axis held-out loss, with narrow
and wide observed curves and separately labeled prospective predicted curves or
intervals. Mark predicted and observed events, and add aligned panels showing
fixed-probe response and current/reference demand. Show component loss where
possible, so ongoing progress on frequent factors does not hide a stalled
correction. Where exact closure is absent, label a threshold crossing honestly.

A convincing study needs: (1) choose a predeclared population of width pairs,
not only an outcome-selected example; (2) save forecasts, event definitions and
horizon before outcomes, using no future trajectory to generate them; (3) show
aggregate results and uncertainty plus representative successes and failures;
(4) include intervention and baseline comparisons, including recovery; and
(5) test whether any predicted loss plateau follows from a proved conditional
bound or is merely an empirical forecast. A predeclared initial-state selection
may define a conditional example but does not imply an unconditional width rate.
If no relevant closure or loss plateau occurs, that is a valid outcome.

The existing [three-prototype example](evidence/competition_reserve/learning_curve_example_20261030/README.md)
shows a selected pair, a reserve-based closure forecast, and a loss floor from an
additional special-case invariant-trap theorem. Its full loss curves are observed,
not general reserve forecasts. It is an illustration and a design reference, not
proof of this broader target.
