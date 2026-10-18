# Earlier findings and unsuccessful approaches

This consolidates the older reports as of 18 October 2026. These studies informed
the [current objective](../../OBJECTIVE.md), but their many diagnostic plots and
individual narratives are no longer the main navigation. Full original reports
and producers remain in Git commit `b03c53f`; compact cohort evidence remains in
[historical tables](../tables/history/README.md). Deleted local raw arrays require
rerunning the historical producers to recover. Numerical results below describe
their original settings, not the newer residual-growth task.

## 1. Width helps; simple agreement does not explain it

Original shallow ReLU teacher-8 experiments used Gaussian inputs, trainable
hidden biases and both layers, usually zero outgoing initialization. At the
reported long budget, test MSE below 1e−6 occurred in 6/20 starts at width 8,
8/20 at width 9 and 18/20 at width 10. The larger teacher-8/16 studies also
found width benefits. Width changes the optimization problem even when the
narrow architecture can represent the teacher.

Distribution shape and aggregate gradient agreement were insufficient objects.
At about 20k SGD updates, raw agreement was .257 at width 8 and .133 at width 16,
although width 16 fit much better. Early skewness/tail changes varied by optimizer.
A saved-state rate reduction removed one high-rate SGD plateau. Initialization,
finite-step stability, and training phase must remain alternative explanations.

## 2. Batch transfer and same-step neuron effects are distinct

Early momentum, near training MSE .2, showed a reproducible local pattern over
40 paired seeds: harmful ordered batch directions fell from 7.46% to 1.79%
at width 8→16, helpful transfer increased, and harmful transfer decreased.
This did not persist uniformly across loss stages or optimizers.

On the same batch, strict events require two neurons to help in isolation but
one to become harmful after the other's fixed update. Early momentum's event
frequency fell from 1.2448% to .2532% per eligible ordered pair. Total events
per batch were almost unchanged (.481→.474). More pairs at greater width make
frequency and aggregate burden different questions.

Broader positive interaction cost includes ordinary diminishing returns, not
only harmful reversals. More partial cancellation of neuron movements did not
imply net cancellation or coordinated protection. These are useful definitions
and controls, not a universal account of competition.

## 3. Cumulative accounting is exact but not causal attribution

For squared loss, isolated helpful gain G, isolated harm H, and net pair cost
K obey L_initial−L_final = G−H−K on a fixed reference set. All terms can become
large through repeated movement and undoing of movement.

In the 200k momentum study, raw positive pair cost fell from .49534 to .05514
with width 8→16. At equal accumulated helpful activity the reduction was about
12%, rather than 89%; much of the fixed-budget difference accompanied earlier
convergence. Conversely, Adam .001 accumulated about 68% more positive pair
cost for teacher 8 and 29% more for teacher 16 while fitting much better.
These counterexamples rule out a universal explanation through reduced cost.

The objective-scale audit found tiny cumulative strict harm for low-rate Adam.
Early cumulative measurements added little stable forecasting information once
early loss and the width/optimizer setting were included. A cumulative cost
saving is not a removable fraction of final loss, and matching loss does not
match residual structure or optimizer history. Matched-loss strict-event rates
sometimes favor width even when whole-trajectory rates reverse; both are
conditional descriptions, not competing causal proofs.

## 4. Temporal interventions: influence exists, but is conditional

Short paired continuations changed an earlier sender-neuron update and measured
later recipient harm across all 245 original trajectories. Prediction-matched
pulses, uniform controls, held-out examples, and multiple lags were included.

In teacher-8 momentum at rate .1, the ordinary earlier update often *prevented*
later harm: a narrow-network benefit of 8.78 [4.43,14.44] units per 100 units of
initial pulse impact in the 200k cohort. The wider effect was −.94 [−1.54,−.32].
This is temporal cooperation strongest in the narrow model, not a universal
harmful-peer mechanism relieved by width. Signs changed with teacher, rate,
phase and horizon. Low-rate Adam's corresponding effects were near zero or
very small, despite substantial fitting differences.

Pulse-normalized percentages exaggerate practical size if confused with total
loss: one narrow momentum comparison was only .00170 harm units per 100 units
of ordinary helpful work over the continuation window. Actual loss effects and
changes in standalone harm also differed. These interventions establish local
temporal influence, not mediation of the final width benefit.

## 5. Allocation and frozen-feature probes: useful diagnostics, weak width rule

Cross-fitted reweighting of existing neuron updates improved held-out one-step
loss beyond the best tested uniform shrink. On MNIST, extra benefit per 100
units of ordinary helpful work was 5.28→8.34 for squared-loss SGD, 21.44→16.32
for momentum, 26.91→20.70 for Adam, and 11.28→2.35 for cross-entropy momentum
(width 16→128). The width ordering is inconsistent; several contrasts are
uncertain. Larger means more recoverable benefit under this intervention,
not a larger final-error fraction explained.

For example, narrow squared-loss Adam's 26.91 helpful-work units corresponded
to only .28% of current held-out loss per tested step. Sign filtering and tiny
steps showed that allocation regret can arise from individually harmful
updates, not necessarily harmful interaction between individually useful ones.
Readout refits indicated a substantial feature limitation in narrow Adam
failures, but an oracle refit does not identify how that limitation developed.

The next direction therefore needs a derived object and prospective conditional
prediction, rather than selecting another diagnostic with a convenient trend.

## 6. Explicit specialization: valid geometry, limited training explanation

A correlated positive-output ReLU teacher pair has regimes where moving its
best merged feature raises loss but separating two copies lowers it. The local
ordinary-motion and splitting curvatures have opposite signs for pair
half-angles between 0 and about 63.76 degrees under the stated construction.

Teacher-informed crowded versus balanced initial allocations confirmed a
specialization delay. However, the most striking relative effects concerned
small unresolved detail (about .3% target-variance MSE in the highlighted
narrow setting), balanced minimal-width starts could outperform expansion,
and the tested five narrow starts caught up at 100k updates. Explicit copies,
cluster allocation, and a perturbation scheme favoring separation were material
assumptions. This was evidence for a constructed delay, not persistent trapping
or a general mechanism in ordinary random initialization.

The current [residual-growth experiments](residual_growth.md) remove those
initialization requirements and find stronger, persistent gaps with opposing
outputs. They retain a serious limitation: two natural scalar sign predictions
fail, and beneficial shrinkage can inflate an apparent suppression count.

## What carries forward

Keep objective units and baselines, paired seeds, rate/initialization controls,
negative regimes, and a distinction between local identities and trajectory
causation. Preserve the real counterexamples: width can help without lowering
an interaction score, and a feature can shrink because learning is working.
The current [residual potential](../theory/residual_growth.md) is a candidate
mechanistic object, not a retrospective resolution of all these findings.
