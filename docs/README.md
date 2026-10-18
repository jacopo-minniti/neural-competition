# Research summary: width, competition and update interference

The experiments support an optimization benefit from additional neurons in this
teacher–student problem. They support **some local interference mechanisms**,
especially early in momentum training, but do **not** establish that wider
networks generally succeed because they have less interference. Adam provides
important counterexamples: wider students can fit much better while accumulating
more of the measured pairwise interaction cost.

This is the starting point for the repository's reports. All numerical claims
refer to completed experiments, rather than proposed experiments. The detailed
reports preserve the optimizer, loss stage, seed counts, normalization and
uncertainty needed to interpret each comparison.

The new [allocation and residual-work study](reports/allocation.md) tests
duplicate-invariant alternatives to pair cost on all original teacher cohorts
and the MNIST matched states. Reweighting neuron updates improves held-out
one-step loss in several settings, but the gain does not consistently decrease
with width. Sign-filter and tiny-step controls separate individually harmful
directions from finite-step relative allocation; full-cohort readout refits
also locate a substantial feature limitation in narrow Adam failures.

The later [specialization study](reports/specialization.md) starts from a
[geometric mechanism](theory/specialization_geometry.md): a shared feature can
oppose ordinary movement while favoring separation into two features. New
both-layer training experiments support a conditional specialization delay,
with balanced minimal-width networks and longer-training catch-up limiting the
claim. This is distinct from establishing harmful neuron-to-neuron updates.

The newest [residual-growth study](reports/residual_growth.md) removes the
splitting intervention entirely. Ordinary random training on overlapping
features with opposing outputs produces larger width gaps, persistent to 100k
updates in the extended starts, and loss of required positive-output roles in
narrow models. Its [residual-potential derivation](theory/residual_growth.md)
gives a local signed growth law, but two simple sign-based predictors fail.
The report distinguishes competition between feature demands from a proved
causal mechanism of harmful peer updates.

The [MNIST experiments](reports/mnist.md) extend the main measurements to a
one-hidden-layer image classifier under squared loss and cross-entropy. Across
120 main runs and 80 rate controls, width materially improves fitting, but
batch conflict, strict pair frequency and accumulated C_plus give different
answers. Primary G/H/total-interaction budgets measure every actual update.

[Literature-guided explanations and predictions](theory/README.md) connects
these specific findings to selected landscape and dynamics papers. It separates
paper results, deductions, small checks of retained states and proposed tests,
including alternatives to interpreting C_plus as universal competition.

The [matched-loss strict-interference follow-up](reports/strict_matched_loss.md)
revisits Adam's whole-training strict-event reversal. In the common loss range,
the wider models have lower per-pair frequency; the report checks harm magnitude
and total event counts as well, using verified replays of the original runs.

The [objective-scale magnitude audit](reports/effect_magnitude.md) asks the
harder question of whether these differences are large and predictive. It
finds substantial local SGD/momentum effects, tiny cumulative strict harm
for low-rate Adam, and little stable extra forecasting value in the early
cumulative scores after accounting for early loss and the width setting.

The [running objective-budget plots](reports/cumulative_budget.md) show G,
H and net pair cost cumulatively against actual training updates, alongside
the loss decrease they account for exactly. These use raw half-MSE units,
with no loss-bin averaging or division by G.

The [temporal peer-update experiment](reports/temporal_branches.md) now directly
changes one earlier neuron update and measures later harm at other neurons.
It uses short paired continuations, prediction-matched pulse controls and all
245 original trajectories, rather than inferring causality from lag correlations.

## The hypothesis and what would count as support

The working hypothesis is that a minimally sized network asks a small collection
of neurons to satisfy incompatible demands. Extra neurons could let useful
features specialize, distribute corrections, or compensate for disturbances.
This could produce less harmful transfer between batches, less antagonism
between neuron updates on one batch, more useful reinforcement, and eventually
more reliable fitting. The architecture can represent the teacher at the
minimal width, so an improvement above that width is an optimization benefit,
not an increase in the teacher function's representational necessity.

We now distinguish four questions:

1. **Across data:** does the update proposed by batch B help or harm batch C?
   The network-level optimizer-direction score answers this directly to first
   order. Decomposing that score into neuron blocks attributes the conflicting
   data demands to neurons; it does not yet measure a direct neuron–neuron effect.
2. **Between neurons on one batch:** do their fixed updates reduce one another's
   benefits, or make an otherwise helpful update harmful? Exact isolated and
   combined neuron prediction changes answer this locally for squared loss.
3. **Throughout training:** how much unfavorable pair interaction is actually
   experienced, relative to standalone gains, favorable credits and final fit?
   Every-update accounting answers this descriptively, including when the
   wider network stops moving earlier.
4. **Across time:** does an earlier neuron update change later standalone harm
   at other neurons? Short paired interventions now test this causal influence;
   they do not identify how much final width benefit is mediated by it.

There is no single privileged definition that settles all three questions.
[Metrics and signs](metrics.md) explains their relationship and limitations.

## What the evidence says

| Question | Main observation | Assessment for the hypothesis |
| --- | --- | --- |
| Does extra width help fitting? | Original teacher-8 SGD study: test MSE < 1e-6 in 6/20 starts at width 8, 8/20 at 9, and 18/20 at 10. Larger expansions also help in the later teacher-8/16 sweep. | Supported for this task and tested budgets; not a guarantee for each initialization or a universal width ranking. |
| Does width change gradient-dot tails? | Early excess kurtosis falls at width 8→16 for SGD (1.62→1.41) and momentum (1.57→1.28); Adam has no clear corresponding change. | Some geometry changes; skewness/kurtosis are not themselves interference metrics. |
| Does global raw agreement explain fitting? | At about 20k SGD updates, A is .257 at width 8 versus .133 at 16, although width 16 fits much better. | A is not a universal predictor or sufficient explanation. |
| Do raw neuron conflicts shrink? | Early SGD positive mass increases, while negative mass has no clear reduction; later negative mass can increase with width. | More reinforcement is plausible in some phases; universal reduction of local negative demands is unsupported. |
| Does momentum's actual direction transfer better? | Near MSE .2: P_step .2761→.3570, N_step .02764→.01651, and harmful ordered batch directions 7.46%→1.79%, with 40 paired seeds. | Clear early descriptive support. The same pattern is not established at every later loss. |
| Does Adam show the same pattern? | Five-seed early matched-loss check: harmful direction frequency decreases, but is already below 1%; P_step/N_step have no clear paired improvement. Later width-32 patterns are stronger but at very different fitting stages. | Weaker, phase-dependent support; not a replication of the full early momentum result. |
| Do neuron movements cancel protectively? | Partial cancellation increases with width, but net energy ratio R remains above one, including after projection away from the residual. | More partial offsetting, not dominant net cancellation or proof of learned coordination. |
| Are strict same-batch interference events reduced? | Early momentum frequency per eligible ordered pair 1.2448%→.2532%; total events per batch .481→.474. Aggregate marginal harm is much smaller, but already small. | A lower per-pair probability and smaller diagnostic harm; not fewer total events or milder events conditional on occurring. |
| Does the broader interaction cost shrink? | Early momentum S_plus 14.38%→8.88%; early SGD 10.83%→9.49%. Later momentum at MSE .01 gives 4.22%→4.45%. | Broader local evidence early; again depends on phase. Includes ordinary diminishing returns. |
| Is the accumulated penalty smaller? | In the 200k momentum study, raw reference cost .49534→.05514; at equal standalone activity it is about 12% smaller rather than 89% smaller. SGD also favors width, with a smaller equal-activity effect. | Supported for these SGD/momentum trajectories. Faster convergence accounts for much of the fixed-budget difference and is itself relevant. |
| Does that explanation survive Adam and other teachers? | Momentum's pattern repeats across teacher sizes, a lower rate and a second teacher. Adam .001 incurs about 68% more pair penalty at teacher 8, and 29% more at teacher 16, while fitting far better. | Refutes a universal explanation through reduced C_plus. It does not refute every possible definition of competition. |
| Are there alternative explanations? | Narrow zero-output Adam models acquire only nonpositive outgoing coefficients. Random outgoing initialization greatly improves them. A saved-state learning-rate intervention removes one high-rate SGD plateau. | Initialization/feature coverage and finite-step dynamics materially matter. |

## Main settings and where to read them

The original main mechanism experiments use a one-hidden-layer ReLU network, Gaussian
inputs in 20 dimensions, noiseless teacher targets, 8,192 training and 4,096 test
examples, batch size 64, and output scaling 1/sqrt(width). Both student layers
and hidden biases train. The default outgoing coefficients start at zero.
Training minimizes half-MSE; fitting results generally report ordinary MSE.
These conventions are detailed in [the protocol](protocol.md).

| Study | Scope | Detailed report |
| --- | --- | --- |
| Longer training and mild expansion | Teacher 8; K, K+1, K+2, 2K, 8K, 16K; rate controls and up to 20 seeds at small widths | [Training and width](reports/training_width.md) |
| Distribution study | 150 main runs; 40-seed early comparison; 512 fresh probe batches; initialization/Muon/rate pilots and saved-state rate continuation | [Batch distributions](reports/batch_distributions.md) |
| Neuron attribution and prediction changes | Raw P/N, accurate-example harm, partial cancellation, R and 1% step controls | [Neuron transfer](reports/neuron_transfer.md) |
| Momentum replication | Raw versus actual optimizer direction, positive/negative masses, collateral harm and later-stage controls | [Momentum](reports/momentum.md) |
| Same-batch pairs | Strict events plus broad J costs, frequency/severity, fresh sources, tolerance and rounding controls | [Pair interactions](reports/pair_interactions.md) |
| Accumulated cost | 40 SGD/momentum runs; 200k updates; every-update source/reference accounting, equal activity and matched-loss controls | [Cumulative penalties](reports/cumulative_penalties.md) |
| Robustness | 205 runs at 100k; teachers 8/16; widths K/K+2/2K/4K; two Adam rates; teacher/rate/initialization controls | [Robustness](reports/robustness.md) |
| Strict interference at matched loss | Verified replays of 90 original teacher-8 runs; five seeds; both Adam rates, output initialization controls and momentum | [Strict frequency and harm](reports/strict_matched_loss.md) |
| Magnitude and objective connection | Existing 245 cumulative trajectories, matched-loss budgets, direct objective probes and early-feature forecasts across 205 robustness runs | [Effect magnitude](reports/effect_magnitude.md) |
| Running objective budget | Exact every-update G/H/net pair sums for 130 original SGD, momentum and Adam trajectories, including initialization controls | [Cumulative G, H and net pair cost](reports/cumulative_budget.md) |
| Temporal causal influence | Sampled sender-update interventions in all 245 original trajectories; matched prediction pulses, uniform controls, held-out outcomes and lags through 128 updates | [Temporal peer effects](reports/temporal_branches.md) |
| Quick Adam direction check | 180 saved/replayed states; students 8/16/32; five seeds; 128 fresh batches per state | [Adam cross-batch transfer](reports/adam_cross_batch.md) |

Historical teacher-4/64/512 scans were already discarded in an earlier curation;
their precise cohort results cannot be reconstructed from this repository.
They motivate the longer-budget work but are not used as quantitative evidence
in this synthesis. Splitting was only a mathematical motivation for those
original experiments; the later [specialization study](reports/specialization.md)
now performs controlled perturbed expansions in a new correlated-feature task.

## What the different percentages mean

The denominator matters as much as the numerator. A harmful-batch percentage
counts batch pairs whose first-order score is below zero. A strict pair frequency
counts cases in which two neurons help alone but the second becomes harmful
after the first, conditional on both helping alone. Neither includes severity.

S_plus instead divides positive pair-interaction cost by summed standalone
helpful gains. Thus 14.4% means **14.4 half-MSE units of positive interaction cost
per 100 units of standalone helpful gains**, accumulated over the diagnostic
batches within each seed. It is not 14.4% of current loss, of final error, or of
neuron pairs. Favorable interaction credits and standalone harmful updates also
enter the exact loss budget.

More generally, "positive interaction" has two meanings that must not be mixed.
A positive gradient-transfer dot is helpful to first order. A positive J adds
loss relative to the sum of isolated neuron effects: it is an unfavorable cost
in that accounting. It can still arise when both neurons help overall, simply
because correcting the same error has diminishing returns.

## Which accumulated comparison should be primary?

For the question **"how much penalty did the actual trajectory incur over a
fixed training budget?"**, use raw accumulated C_plus, alongside net interaction
C_plus−C_minus, standalone harm/gain and fitting curves. Earlier convergence
and smaller subsequent motion are part of that observed width benefit.

Equal accumulated standalone gain and matched-loss summaries are complementary
controls. They ask whether the interaction cost remains smaller when conditioning
on activity or phase. They are not universally fairer replacements for the raw
total. Equal standalone gain is not equal net learning progress; matching loss
does not match the residual pattern, representation or optimizer history. Some
narrow Adam models never reach the full matched-loss interval, so missing
comparisons are not zeros and cannot be silently restricted to successes.

The large reduction in the 200k momentum raw total and the much smaller
equal-activity reduction are both real descriptions. Conversely, Adam .01 can
have less total cost but more cost at equal activity. Reporting both prevents a
conclusion from depending on a single conditioning choice.

## What remains unproven

The experiments do not show that accumulated interaction cost is a removable
fraction of final loss. Changing an update would change subsequent parameters,
residuals, features and optimizer moments. Exact local accounting is not a
causal decomposition of the eventual fitting advantage.

Several explanations remain compatible with the observations: useful feature
coverage, allocation across neurons, initialization-dependent sign restrictions,
step-size/curvature effects, optimizer history, and changes in how updates are
distributed across units. The shared scalar learning rate and 1/sqrt(width)
scaling do not fix hidden-feature learning strength across widths. The metrics
also depend on parameterization and the chosen neuron blocks.

The [temporal peer-update experiment](reports/temporal_branches.md) now intervenes
on one proposed neuron update while retaining its current contribution, then
measures other neurons over short continuations. This identifies that pulse's
causal influence, not the share of final width benefit caused by competition.
Earlier saved-state interventions varied learning rate. A sustained allocation
intervention with useful-correction controls and an optimized uniform-step
baseline remains a stronger test of the full proposed mechanism.

All uncertainty is conditional on the tested teacher functions and datasets.
Bootstrap intervals resample paired training seeds, not correlated batch pairs
or neuron pairs. Multiple exploratory comparisons, approximate loss matching,
rare events, and near-zero-gradient ratios limit strong mechanistic claims.

## Repository and evidence after cleanup

Reports, tables and figures are versioned under `docs/`. Notebooks have been
removed. [Data retention](data.md) documents the representative raw archive,
the full-cohort result tables, source hashes and what was deleted.
[Reproduction](reproduction.md) distinguishes inspecting retained evidence from
repeating the original full cohorts. Success rates and uncertainty must continue
to use the full-cohort tables, never the purposively retained raw subset.
