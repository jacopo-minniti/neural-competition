# Options, competition, and width: current research context

Last updated: 29 October 2026.

Latest continuation: [minimal model and probabilistic option transitions](theory/minimal_probabilistic_options.md).

This is the maintained statement of the current framework, evidence, and research
objectives. Update it when results change the definitions, supported claims, or
priorities. Detailed derivations, protocols, and raw results belong in the linked
study notes; this document should remain concise.

## Objective

Develop a theory connecting **functional correction options, competition for
shared computational resources, and the learning benefits of width**. Preserve
the current spectral framework while improving the assumptions and dynamics that
make it explanatory across different tasks.

The working hypothesis is that learning one useful structure can commit shared
resources in ways that weaken other useful corrections. Extra width may supply
alternative implementations or permit faster recruitment and recovery. This is
a hypothesis to test, not an interpretation guaranteed by observing a width gap.

The priority is a simple, empirically useful account of one phenomenon: learning
pressure changes useful correction responses through shared trainable features.
Harmful competition is its negative case, not an inevitable consequence of
learning. Retain only ingredients needed to predict these changes and their loss
effects. Harm requires a deficit over the correction budget, not merely an
instantaneous loss of strength. The active phase focuses on controlled heterogeneous-frequency
task families; **MNIST is paused**. First probabilistic transition forecasts now
complement causal investigation; whole-run prediction remains secondary. Earlier local MNIST forecasts remain evidence,
but whole-run prediction and full mediation of natural width remain open.

## Current foundation: spectral options

On a fixed data distribution or audit panel, let J be the prediction Jacobian,
M the declared optimizer metric, and H the output-loss curvature. For n uniformly
weighted examples, define

\[
 D=JM^{1/2},\qquad A=D^THD/n,\qquad
 B_\tau=\frac1nD\left(A+\frac1{2\tau}I\right)^{-1}D^T.
\]

Data weights can instead be absorbed into the coordinates. Classification uses
centered output contrasts. The scale tau is an observation horizon derived from
local GD dynamics, not an additional editing objective.

An **option** is a positive-eigenvalue eigenspace of B_tau; its eigenvalue is its
response strength. Directions are discovered from model derivatives. They are
not assigned neuron roles, teacher features, or hand-selected corrections.
Several parameters can support one option, and options can share implementations.
The spectrum is primary; breadth and mass are summaries, not literal counts.

For current residual demand e, the option value is

\[
 V_\tau(e)=\frac1{2n}e^TB_\tau e.
\]

This equals discounted loss benefit in the frozen local quadratic model. It is
not generally a certificate of recoverable nonlinear loss. Availability and
demand remain separate: a successful model can have a small value because it has
already removed accessible errors. Report residual magnitude, normalized
directional strength, and actual loss alongside raw value.

Use a deterministic-budget member when the question concerns h actual steps:
B_(h,eta)=D phi_(h,eta)(A) D^T/n, where
phi_(h,eta)(a)=[1-(1-eta a)^(2h)]/a and phi_(h,eta)(0)=2h eta.
This is the same spectral formalization with a different time filter. A lower
V_tau need not imply lower gain at h; actual response evolution and original
loss curvature still matter beyond the frozen quadratic approximation.

The instantaneous member of the framework is K=DD^T/n. Under full-batch gradient
flow on the original loss,

\[
 S_t=\frac1n e_t^TK_t e_t=-\dot L_t.
\]

This identity connects options to current learning rate where the gradient-flow
chain rule applies to the declared Jacobian. Discrete ReLU experiments need the
finite-update correction: repeated boundary crossings can invalidate simply
integrating the current one-sided Jacobian, even under step refinement.
It is also the
optimizer-weighted gradient norm squared, so the identity alone supplies no
additional causal explanation of width. Finite-step SGD requires its own noise
and nonlinear-update accounting.

## Active task families and paused transfer setting

The general family is y=s(Z)+Σ_j 1_(R=j)h_j(Z), with unequal context frequencies
or importances. Conditional-mean residual Ce and centered residual (I−C)e give
orthogonal data-space pressures, while their trainable implementations can overlap.
The shared rule and refinements need not use the same attributes. Every component
remains in the objective throughout training; changing relative demand is endogenous.

| Context | Current evidence and limits |
|---|---|
| Discrete contextual toy | Fresh 64-pair cohort: 12 width-4 substantial errors, zero at width 8. All start with ≥95% structural refinement coverage; 15 narrow paths lose it and three recover. Earlier independent cohorts had 11/0 and 10/2 failures, so wider fitting is not guaranteed. |
| Continuous variation | Fresh dense losses .015554/.001113 at widths 4/8. Four of 13 initially covered narrow paths that close recover. Fine variation leaves small errors even with good coarse coverage; exact full-span thresholds are misleading. |
| Separate-attribute refinement | y=2Z₁+2RZ₂. Width-4/8 mean losses .060313/.015781, substantial errors 29/6. Most first large losses of support are common-pressure-induced; the mechanism does not require common and refinement rules to use one input attribute. |
| Refinement hierarchies | N=1/2/4, fixed total refinement frequency, decreasing ratios, natural widths and budgets to 4000. Wider models retain more early coverage and complete more rare refinements; many closures recover. One apparent persistent gap disappears under step refinement. |
| Frequency and negative controls | q=.02 reduces loss cost without eliminating failures. Centering away the shared target retains a width gap, mostly under refinement-induced closures. Two refinements usually recover; the pilot's zero failures becomes two in fresh narrow runs. |
| MNIST — paused | Preserve local bilinear response-forecast successes and weak/transient competition findings. They do not establish ordinary endpoint rescue or width mediation. No new MNIST experiment is part of the active phase. |

These are realizable shallow-ReLU tasks with explicit four-unit witnesses (six
for two refinements), small Gaussian initialization, squared loss and a declared
mean-field optimizer metric. The continuous task uses bounded clusters and
separate training/dense quadrature. The new cohorts use full original-task
training to optimizer time 2000. “Substantial error” is above 5% of initial
centered-refinement energy; raw losses and threshold sensitivities are retained.

## Minimal working model: options, demand, and conditional transition probabilities

Keep **B and residual e as the two evolving state objects**. Task structure,
width, optimizer and budget condition their evolution. Do not require a separate
resource distribution in the predictive core. The [implementation-state study](theory/functional_implementation_state.md)
remains supporting theory: functional particles reconstruct network dynamics,
gate moments reconstruct current options, and matched-moment counterexamples
show that current options do not determine their future. Exact particle
transport is not a simplifying forecast; moment closure fails at gate crossings.
Hidden implementations are marginalized in the new probabilistic account.

The [new study](theory/minimal_probabilistic_options.md) uses existing directional
support Q, finite-budget strength s, and remaining gap d. Q is the infinite-budget
limit of directional spectral response, not a new definition of an option.
Support reopening, strong response and completed correction remain distinct.

Two historical-data-selected laws were frozen before 1,728 fresh runs:

- Early demanded support closure, conditional on initial coverage, uses only
  task alignment and relative width: p=1-exp[-.8918 exp(-2.1955 a)
  (m/(2+2N))^(-1.9689)], for the declared time-20 window. This fitted exponent
  is not universal. More task predictors slightly improve familiar-family
  scores; the smaller model transfers better to new families.
- First reopening after a closed, demanded time-20 state uses
  p(T)=1-exp[-lambda(x) log(T/20)]. A log-linear rate uses Q, log strength,
  log remaining gap and four fixed task/width covariates. The hidden-rate
  mixture interpretation is an assumption, not a new measured resource state
  or a theorem about neural hazard. Deadlines are 50/100/500/2000.

New seeds, rank-power frequency profiles, N=3, alignment .5 and width factor 3
provide transfer tests. Closure Brier is .1364 versus .1763 for the constant
baseline; reopening .0940 versus .1187 for task/width alone. The separate new-
family scores are .1539 and .1131. Probabilities are imperfectly calibrated:
mean reopening prediction .4714 versus observed .5466, with most timing error
at early deadlines. No refit uses these fresh outcomes.

Width strongly improves closure prediction: dropping it worsens Brier by .0356
(seed-cluster 95% interval [.0300,.0409]). Its incremental reopening improvement
after conditioning on state is .0016 ([-.0018,.0050]), although it helps on the
transfer subset. Closed-state selection differs by width, so this is not proof
of mediation or evidence that width cannot help recruitment. Dropping gap from
the support-reopening predictor also costs little; demand is still necessary to
judge harm. The gate-group-count proxy slightly improves Brier but worsens log
loss and late-deadline Brier. It does not justify restoring the full intermediate
state. Further reduction of the recovery predictor needs replication.

These are sampled, conditional **first-passage** forecasts, not a complete joint
option process, sustained-survival probabilities or final-loss predictions.
The gamma waiting-rate representation would imply eventual reopening and must
not be extrapolated to deny permanent failure. Keep the previous discrete
obstruction and continuous recovery cases. [Raw records and reproduction](theory/probabilistic_options/README.md)
include frozen forecasts, exact sources, seed-cluster uncertainty, all ablations,
calibration and final losses. No new causal intervention is claimed here.

## Retained hierarchy evidence and causal limits

Fresh 48-seed banks cover N=1,2,4 refinements, frequency ratios 1/.25/.1,
shared versus separate attributes, sufficient narrow widths 2+2N and twice that,
and selected fourfold widths. All components remain present. Budgets reach 4000.
Targets have explicit narrow-width witnesses. The attribute-sharing manipulation
is not assumed to equal implementation overlap; gate-resource participation is
measured separately and is often higher in the easier shared-attribute setting.

For four refinements with ratio .1 and separate attributes, initial coverage is
already nearly complete. Demanded closures of the rarest refinement occur in
27/48, 9/48 and 0/48 runs at widths 10/20/40. All recover structurally, but only
30/48, 39/48 and 48/48 master that refinement by time 4000. Relative gap errors
are .1268/.0736/.0359, despite tiny total losses. Initial missing span alone is
insufficient; initial strength differences are not thereby ruled out.

Matched-option constructions have identical current predictions and full K but
different signed moments and later response paths. Early natural-state rotations
can preserve even all moments and still change outcomes through different gate
trajectories. Effects have both signs and restrictive eligibility; late mixing
is often ineffective. These are causal tests of missing state information, not
a proven rescue method or complete mediation of the natural-width benefit.

A numerical negative is consequential: the apparent N=4 ratio-.25 narrow plateau
falls from .001061 at step .05 to 5.94e−8 at .0125. It is not robust evidence for
a continuous-time trap. Keep all resolutions. Earlier discrete obstruction,
continuous recovery, harmful pressure knockout, unsuccessful copying, restrictive
recycling and weak recovery certificates remain linked historical evidence.
No MNIST run was added. See [raw records and reproduction](theory/implementation_state/README.md).

The [previous toy phase](theory/heterogeneous_option_resources.md) supplies the
same-state pressure interventions and same-width directional recycling evidence.
Its theoretical response susceptibility and conditional width bounds remain
valid under their assumptions. The resource construction exposes hidden state
and the limits of moment closure; the smaller predictive model integrates that
uncertainty out rather than claiming a deterministic closure.

## What the framework must explain

1. **Observed learning dynamics.** Relate option strength and residual loading to
   measured loss reduction during training, including differences between widths.
   Separate current loss, absolute learning rate, and fractional error reduction.
2. **Success and failure.** Distinguish a solved model with little remaining
   demand from a model with substantial error and weak or missing responses.
   Explain why a weakened correction sometimes recovers and sometimes remains
   inaccessible over the training budget.
3. **Hypothesis-consistent variation.** Derive conditions under which changing
   width, task structure, exposure, initialization, or optimizer should strengthen,
   weaken, or remove the proposed effect. Test both positive regimes and controls.

Compare widths on common residual patterns as well as their own residuals. Equal
loss does not imply equal demand. Correlations along a trajectory can reflect
time and shrinking errors; report within-run, matched-stage, and paired-width
comparisons with their relevant baselines. High correlation alone does not show
that option preservation causes the width advantage.

## Retained local model and earlier MNIST validation

Retain the local bilinear coefficient–feature model, full residual directions,
original loss, optimizer and physical ReLU response evaluation. Freeze masks
within each local interval; refresh once midway through a 20-step interval from
**predicted parameters**, and at the declared change of pressure. This is a
parameter-level surrogate with no fitted coefficients, not a low-dimensional
closure or a computational shortcut. The refresh interval is a numerical
resolution, not a new task constant or latent recovery state.

The [earlier local study](theory/local_response_model.md) established the
response-matching algebra and why physical response evaluation matters. The
[finite-budget continuation](theory/finite_budget_competition.md) strengthens it:

- On ten fresh MNIST seeds covering all ten focal digits, a midpoint refresh
  reduces finite correction-contrast error from .237 to .091 of a zero-effect
  baseline. It predicts 118/120 signs, with dependent rows grouped by seed.
  An eight-seed focused replication gives .117 error ratio and 96/96 signs.
- The probe matches initial predictions, demand and curvature across pressure
  histories using fixed output offsets and soft background targets. It tests
  capability consequences; original-label recovery is reported separately.
- Signed copies (2,-1), with matched initial function and **full optimizer
  kernel**, reduce feature movement as well as smooth feedback. They reduce
  absolute option drift in 18/20 broad-cohort states, but hurt 20-step
  correction in 19/20 and worsen mean ordinary full-task recovery.

**Causal scope.** Three of 20 broad-cohort states have immediate correction
impairment; forecast-selected preservation partially rescues all three. Late-only
copy controls support a pressure-history contribution. None remains impaired at
five steps. One different narrow state develops a small 20-step deficit, correctly
predicted and rescued in the controlled probe. That digit-specific deficit does
not recur in eight fresh starts, and the model correctly predicts its absence.
The focused cohort has three more immediate impairments, again gone by five steps.
These are causal, mostly transient pressure effects—not ordinary MNIST endpoint
rescue or mediation of its width advantage.

A retrospective nested comparison shows why further simplification fails here:
frozen quadratic response misses changing curvature; exact CE with a frozen
Jacobian still misses consequential response evolution. The existing coupled
model captures both without adding depletion, recruitment or semantic neuron-role
variables. A scalar option-growth closure remains rejected.

## Earlier conditional width result and its separate scope

The [finite-budget note](theory/finite_budget_competition.md) derives a preservation bound under bounded inputs/unit norms,
diffuse activation margins, a supported finite-dimensional corrective subspace,
and the declared 1/sqrt(width) parameterization. Restricted kernel drift is
O(time/sqrt(width)); if the squared-loss residual stays in that subspace, the
bound gives a finite protected correction window. This supplies a conditional
route from width to maintained learning rate, not a ranking of arbitrary trained
widths. It does not transfer unchanged to mean-field optimizer scaling.

For a stable corrective mode with susceptibility decreasing with width, the
pressure-induced log-loss penalty equals twice its integrated response damage.
Width helps only when damage overlaps the useful correction window; creation
or sufficiently fast recovery can remove or reverse the benefit. Common mode
shape and ordered susceptibility are additional assumptions, not consequences
of the norm bound alone. Wider MNIST models do not uniformly exhibit less drift.

Seek task conditions that predict this adverse **finite-budget** evolution:
shared rules with later-important refinements, overlapping implementations and
inadequate timely recovery. Keep initial capability, residual demand and width
scaling separate. The current experiments support a local predictor on real
images, but do not establish widespread persistent competition in real-world
tasks. Preserve the discrete toy's obstruction and continuous toy's recovery
counterexamples; neither should be generalized solely to fit MNIST.

## Cumulative account of learning

Develop cumulative quantities that describe how training has used strong options
on the residuals actually present. For full-batch gradient flow, the starting
account is already exact:

\[
 \mathcal U(T)=\int_0^T S_t\,dt=L(0)-L(T).
\]

For squared loss, define kappa_t=e_t^T K_t e_t/||e_t||^2. While L is positive,

\[
 \int_0^T\kappa_t\,dt=\tfrac12\log\frac{L(0)}{L(T)}.
\]

For bτ=Vτ/L, the new squared-loss relation κ≥bτ/[2τ(1−bτ)] bounds accumulated
log-loss progress from the actual residual-loaded response path. This follows
from Cauchy–Schwarz, allows mode rotation, and is not an initial-state forecast.
The plotted time average of L−Vτ is descriptive; its sparse numerical integral
is not the exact finite-update ledger.

These connect accumulated residual-response strength to absolute and fractional
loss reduction. The initial loss gap plus the difference in accumulated learning
between widths gives the final gap. Mode-resolved accounts must allow directions
to rotate; matching eigenvectors by sorted index is insufficient.

Extend this account to finite steps, stochastic updates, and relevant residual
components without assigning unexplained changes to options. Summing discounted
V_tau values is not automatically an exact loss ledger. A cumulative identity
using observed gradients or future geometry is an account of training, not an
independent forecast or proof of competition. Training-loss reduction also does
not by itself establish generalization; retain held-out evaluation.

The [finite-update study](theory/observed_learning_and_sampling.md) now supplies
the exact ledger. For actual update delta, prediction remainder
r=f(theta+delta)-f(theta)-J delta, and output-loss Bregman cost C,
loss reduction is -g^T delta + <e,r> - C. For SGD, parameter work separates
into eta g^T M g, signed sampling work, and recorded arithmetic discrepancy.
On eight fresh MNIST confirmation pairs, summed signal over steps 100–800
overstates realized gains by 6.3x/4.0x at widths 32/128; convex cost accounts
for most of the excess. Identical-checkpoint batch-512 interventions reduce
cost and improve one-step gain relative to batch 128 in all eight seeds.
This is local causal evidence about sampling, not mediation of the width gap.

Two measurement conditions are now explicit. A size-b random panel's own
instantaneous response has finite-population bias
(1/b-1/N) times the trace of the (N-1)-normalized gradient covariance in optimizer
coordinates; at MNIST step 800, size-64 measurements
average 41.6 times full-training signal. Unbiased variance correction remains
very noisy and does not repair the old same-panel discounted B_tau forecasts.
For task components, cumulative self-response can be cancelled by cross-response;
use joint residual loading or the existing conditional Schur response. The toy
gate correction has an O(step) cumulative bound under additional uniform margin
regularity, supported more closely by dense continuous quadrature than by atoms.

## Evidence and maintenance

- [Minimal probabilistic options](theory/minimal_probabilistic_options.md), with
  [frozen fresh forecasts, transfer tests and simplification comparisons](theory/probabilistic_options/README.md).
- [Functional implementation state](theory/functional_implementation_state.md), with
  [hierarchies, moment-matched interventions, and numerical counterexamples](theory/implementation_state/README.md).
- [Heterogeneous demand and shared resources](theory/heterogeneous_option_resources.md),
  with [fresh toy interventions, natural-width transitions, and negative controls](theory/toy_resource_competition/README.md).
- [Finite-budget competition and width](theory/finite_budget_competition.md), with
  [fresh causal tests, failed replication and timing controls](theory/causal_response/README.md).
- [Local response model and validation](theory/local_response_model.md), with
  [locked protocols, negative results, raw forecasts and outcomes](theory/response_feedback/README.md).
- [GD-derived spectral foundation](theory/gd_residual_options.md).
- [MNIST option/residual trajectories](theory/option_mismatch_during_sgd.md) and
  [response transport and retention](theory/option_slack_and_transport.md).
- [Task-induced polarization](theory/task_induced_polarization.md) and
  [dynamic conditional options](theory/dynamic_options_and_renewal.md).
- [Delayed demand and recovery](theory/delayed_demand_option_recovery.md), with
  [protocols and raw evidence](theory/option_transport_resilience/README.md).
- [Earlier findings and counterexamples](reports/research_history.md).
- [Observed learning and sampling](theory/observed_learning_and_sampling.md),
  with [reproducible ledgers, batch interventions, and boundary audits](theory/observed_learning/README.md).

When progress changes this context, update the supported claims and remaining
gaps in place, advance the date, and link the evidence. Preserve negative results
and original protocols in their study records. Keep descriptive tracking, causal
explanation, and prospective prediction distinct as each improves.
