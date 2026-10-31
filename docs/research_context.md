# Current research context

Last curated: 30 October 2026.

## Current model

For prediction Jacobian (J), optimizer metric (M), loss curvature (H), and
uniform audit measure over (n) examples, define

\[
D=JM^{1/2},\qquad A=D^THD/n.
\]

The discounted finite-response operator is

\[
C_\tau=\frac1nD\left(A+\frac1{2\tau}I\right)^{-1}D^T.
\]

Its eigenspaces are **capability modes**, with eigenvalues giving response
strengths. For a declared residual demand \(e\), an **option** is a nonzero
correction \(q_E=\Pi_Ee\) in an eigenspace, together with its available
response. Its value is \(\beta_E\|q_E\|^2/(2n)\). The total value,

\[
V_\tau(e)=\frac1{2n}e^TC_\tau e
\]

is the frozen local quadratic loss benefit. An option is closed when its demand
remains but the response is zero; an unneeded mode is not a current option.
Fixed directions remain useful as demand-independent capability probes. For
\(h\) steps of gradient descent on the frozen local quadratic, use the
deterministic-budget filter

\[
B_{h,\eta}=D\phi_{h,\eta}(A)D^T/n,
\qquad
\phi_{h,\eta}(a)=\frac{1-(1-\eta a)^{2h}}a.
\]

Capability and demand remain distinct. A solved model may have little remaining
demand; a failing model has meaningful residual demand but inadequate useful
response over its remaining budget.

## Option dynamics and competition

The predictive dynamics retain the network parameters but simplify the shallow
ReLU network only within short gate-stable intervals. Both layers evolve under
the original loss and optimizer, including the bilinear coefficient–feature
term. Physical predictions and Jacobians are evaluated with the forecast
parameters and actual ReLU gates. A midpoint mask refresh is used for validated
20-step MNIST forecasts.

Different functional directions interact because their derivative
implementations share parameters. For functional directions (u,v) and pressure
(p), the smooth susceptibility is

\[
\mathcal T(u,v;p)=u^T\dot K[p]v,
\qquad K=DD^T.
\]

Diagonal transport changes strength; off-diagonal transport changes coupling
and orientation. Either sign is possible. ReLU boundary changes are measured by
finite counterfactual updates rather than forced into this smooth formula.

Competition is therefore dynamic: pressure changes the response operator, and
harm occurs only if the adverse change overlaps later demand for long enough to
reduce finite-budget correction.

## Width

The current priority is [protected reserve and tolerable common
learning](model/competition_reserve.md), rather than detailed future-step
forecasting. Units inactive on the entire common support supply an exact
operator lower bound under common-only gradient training, from arbitrary states.
Under a stronger coherent-pressure approximation, a scalar inversion maps
common prediction gain to transported capability. A monotone structural special
case makes critical progress the last surviving route's closing exposure mapped
into common gain; a distribution over common demand gives joint loss laws
without independent-refinement assumptions. Full rare-task updates can invalidate
the initial floor. Natural widening changes both alternative routes and their
normalization, so monotonicity is conditional, not universal.

The earlier [conditional forecasting tool](model/power_law_competition.md)
has no Gaussian or small-initialization premise. A multinomial exposure law
supplies joint pressure uncertainty;
the retained coupled local dynamics propagate residuals and both layers. Exact
one-step laws, an analytic shared-count binomial special case, and approximate
multi-step physical response forecasts are distinguished. The older Gaussian
coherent-pressure law below is a restricted analytic case, not the general premise.

Fresh tests use 24 power-law refinements, then transfer to 32 with a steeper
frequency law and different association strength. The richer synthetic tasks
contain co-occurring correlated factors, continuous attributes, context jitter,
nuisance inputs, and mixed observation coordinates. Narrow widths have exact
capacity witnesses. The same predictor improves expected-response forecasts
across three initialization families; this is conditional robustness rather than
proof that initialization stops affecting training.

Width can affect initial coverage, the number and diversity of implementation
routes, survival under pressure, and timely recruitment. Under the toy's
mean-field parameterization, individual feature velocities remain order one;
there is no automatic feature freezing with width. Initial response reliability
improves through averaging. The new
[probabilistic chapter](model/probabilistic_competition.md) derives later closing
and opening probabilities in a coherent-pressure regime: a pressure clock makes
transported implementations independent exactly at fixed exposure, and the
feature flow determines their Gaussian gate-event probabilities. Complete loss
of a specified refinement response has probability b(s)^m; two-time laws exclude
initial absence and include reopening. Finite-strength reliability follows from
the same transported block distribution.

This is an exact restricted model, not a fixed-training-time law for arbitrary
tasks. A separate full-objective invariant region proves that pressure-induced
closure can enforce a nonzero loss floor above representational sufficiency.
Quantifying entry probabilities under ordinary all-data training still requires
clock and perturbation control. The small deterministic audit checks one natural
training construction, not a population width curve.

The strongest controlled evidence is dynamical: fresh discrete runs start with
broad refinement support, but narrow paths close more often and recover less
reliably. Same-state pressure branches cause some closures, and same-width
directional recycling rescues some correction failures. These findings do not
yet identify a complete mediator of the natural width effect.

## Current empirical position

- **Current reserve:** 24-factor power-law tasks with correlated geometry,
  continuous variants, and an exact three-unit capacity witness distinguish
  present capability from protection against common learning. Protected and
  completion reserve rank later half-response losses with AUC .907/.964, versus
  .689 for current strength; 32-factor transfer gives .841/.902 versus .704.
  These pooled scores are descriptive, not calibrated probabilities, and are
  not uniformly best in every stratum. Width-4 versus width-16 mean loss-event
  fractions are 6.09% versus .049% (transfer 6.62% versus 1.29%). Some full-training
  responses violate the common-only floor. These tasks impose a stronger shared
  refinement structure than the earlier co-occurring family. See the
  [reserve evidence](evidence/competition_reserve/README.md).

- **Many power-law refinements:** coupled 16-step expected-response prediction
  has error ratios .485/.205 versus persistence on prototype/associated tasks,
  and .244/.105 after 32-factor transfer. Rare prototype availability losses are
  underestimated. Most short-window resolution closures recover by step 16,
  ruling out an absorbing model in these regimes. Natural-width effects on
  final loss remain modest and mixed. Common pressure weakens prototype response
  slightly but creates response in the associated task; neither sign alone
  establishes harmful learning. The uncentered-input and frozen-pressure
  failures are retained in the [evidence record](evidence/power_law_competition/README.md).

- **Discrete contextual tasks:** replicated width-dependent persistent failures
  above known representational sufficiency; later-needed responses can be lost
  during early learning.
- **Continuous tasks:** missing responses can recover; permanent exhaustion is
  false in general.
- **Multiple-refinement hierarchies:** width reduces demanded closures and
  improves timely rare-refinement mastery, but structural recovery alone does
  not guarantee correction within budget.
- **MNIST:** the local bilinear model accurately predicts controlled short-horizon
  intervention contrasts. Harmful effects are mostly small and transient, and
  mediation of the ordinary width advantage is unproved.

## Open causal question

Establish when task frequency, shared implementation, option transport, and
finite recovery combine into a meaningful loss penalty, and determine how much
of the natural width benefit passes through that chain. The maintained model
does not include a fitted stochastic closure. Current protected reserve is a
decomposition of the existing kernel, not a latent resource. The Gaussian
implementation distribution in the older special case is only its initialization
law pushed through parameter dynamics. More general distributional approximations
require finite-width error control, especially for rare failure events. Earlier
alternatives remain recoverable from commit `92b7904`.
