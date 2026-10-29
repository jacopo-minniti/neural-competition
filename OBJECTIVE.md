# Research objective

Find out whether **competition between neurons or feature demands can materially
impede learning, and whether extra width alleviates that obstruction**, under
explicit, interpretable conditions. This is a search for one qualified mechanism
of overparameterization, not a theory explaining every benefit of width.

A narrow model's ability to represent the target does not imply that ordinary
training can find that representation. We want to understand whether useful
features fail to form, survive, or specialize because the same trainable units
must respond to competing demands. Suppression can also be appropriate learning;
identifying harmful competition requires more than observing contraction,
negative gradient dots, overlapping updates, or a width advantage.

## What we are looking for

- A mathematical object derived from the learning dynamics or loss geometry,
  with assumptions under which it predicts when competition should be strong
  or weak. Prefer an interpretable condition over a post hoc score selected
  because its width ordering looks right.
- A substantial effect in a controlled setting, reported in objective units as
  well as relative percentages, that survives reasonable changes of seeds,
  training duration, optimizer, rates, and initialization.
- A mechanism that appears during ordinary training without requiring assigned
  neuron groups, copied neurons, or teacher-informed starting directions.
  Such constructions can illuminate geometry, but are insufficient evidence
  for the phenomenon we ultimately want to explain.
- Transfer across several relevant settings, including eventual experiments
  without known teacher features. Universality is neither expected nor required.
  Explicit assumptions such as target geometry, feature overlap, sign structure,
  initialization scale, or data composition are welcome when they explain the
  presence and absence of the effect.

Start with experiments and short derivations that inform one another. Do not
spend the project formalizing an artificial special case disconnected from
ordinary training, or proliferating diagnostics without a prospective prediction.
A negative result, a small effect, or a failed sign condition is useful evidence
and must remain visible in the research summary.

## Evidence needed before making a causal claim

1. Establish a meaningful width benefit and its limits. In teacher problems,
   verify that the narrow network can represent the target. In real data,
   acknowledge that increased capacity can also explain improvements.
2. Derive a conditional mechanism and test both its predicted positive regime
   and controls where it should weaken. Separate exploratory follow-ups from
   predictions recorded before the experiment.
3. Show that the proposed obstruction occurs on ordinary trajectories and
   precedes the fitting deficit. Distinguish a changing representation from
   a frozen-feature or fixed-update calculation.
4. Use a targeted intervention, with appropriate rate/movement controls, to
   test whether relieving the obstruction improves learning. Local accounting
   identities and correlations do not establish mediation of the width benefit.

Report absolute loss, relevant baselines, paired seed uncertainty, failed starts,
and overlapping cohorts. A large reduction of a tiny residual is not a large
fraction of the original task explained. Do not pool neurons, batches, or
extensions of the same starts as independent training replicates.

## Current position — 21 October 2026

The [sign-survival follow-up](docs/reports/sign_survival.md) narrows the intervention:
project only readout updates that would cross their initial sign, starting from
randomly assigned four-positive/four-negative width-8 models. Early protection
rescues all 20 fresh paired starts per optimizer and closes 95–97% of the main
mean narrow-wide gap. Matched coefficient-displacement damping is much weaker.
A [Gaussian Hermite-moment bound](docs/theory/signed_capacity.md) now quantifies
approximation error from missing signed roles, certifying 34–38% of ordinary
balanced narrow population MSE in the main cohort.

The limits are substantive: active roles can temporarily vanish even under
protection, fixed protection duration transfers poorly to low-rate Adam, and
balanced-target controls are mixed. At all 253 early positive-to-negative
crossings inspected, the source-batch signal is already adverse without peers.
Thus this supports consequential sign conversion and later recruitment of
complementary roles, not direct harmful peer-induced gradient reversals or unique
mediation of width. Predicting release and recovery remains the next objective.

## Previous position — 18 October 2026

The strongest current setting uses overlapping Gaussian ReLU teacher features
with opposing output signs. Ordinary width-8 networks lose required signed
feature roles and leave roughly 2–5% of target variance unexplained; width 32
fits much better across SGD, momentum and Adam. Most narrow failures persist
in the five starts extended to 100k updates.

The signed residual potential was the starting candidate object. Its local
amplitude-growth law is exact for the stated gradient-flow assumptions, but
neither the initial-evidence sign nor a conditional-Gram sign successfully
predicts the global width-benefit boundary. Role loss has not been established
as the sole mediator of the gap. This is a promising conditional observation,
not a completed competition theory.

The new [profiled-feature follow-up](docs/reports/feature_profile.md) provides
stronger intervention evidence: repeated early readout equilibration at fixed
width closes about 94% of the main width gap on 20 fresh starts per optimizer.
The effect survives keeping optimizer moments, a second dataset, and longer
training. Terminal readout refitting cannot rescue the ordinary learned features.
This supports an early coefficient–feature feedback pathway, without assigning
neuron groups or using teacher directions in the intervention.

It does not uniquely identify harmful neuron competition or its mediation of
width: faster readout learning and, for Adam, moment resets can also rescue
training. Intercept controls are weak, and early profiled loss/readout lag fail
to improve cross-setting endpoint forecasts. The next object must explain the
dynamic tracking of the moving optimal readout and predict these boundaries;
static profile diagnostics alone have not met the objective.

MNIST already shows a practical width benefit and counterexamples to universal
interference scores. Its residual field can be measured from labels and
predictions; identifying which suppressed features are necessary remains an
open problem. No teacher-free residual-potential result has yet been established.

Read [the evidence overview](docs/README.md), [current derivation](docs/theory/residual_growth.md),
and [earlier findings and failures](docs/reports/research_history.md).
