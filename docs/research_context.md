# Current research context

Last curated: 29 October 2026.

## Current model

For prediction Jacobian (J), optimizer metric (M), loss curvature (H), and
uniform audit measure over (n) examples, define

\[
D=JM^{1/2},\qquad A=D^THD/n.
\]

The discounted finite-response operator is

\[
B_\tau=\frac1nD\left(A+\frac1{2\tau}I\right)^{-1}D^T.
\]

Its positive eigenspaces are **options** and their eigenvalues are response
strengths. For residual demand (e),

\[
V_\tau(e)=\frac1{2n}e^TB_\tau e
\]

is the frozen local quadratic loss benefit. For (h) actual gradient steps,
use the deterministic-budget filter

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

Width can affect initial coverage, the number and diversity of implementation
routes, survival under pressure, and timely recruitment. Under the toy's
mean-field parameterization, individual feature velocities remain order one;
there is no automatic feature freezing with width. Initial response reliability
improves through averaging, while later protection requires assumptions about
retained mean response and correlated feature evolution.

The strongest controlled evidence is dynamical: fresh discrete runs start with
broad refinement support, but narrow paths close more often and recover less
reliably. Same-state pressure branches cause some closures, and same-width
directional recycling rescues some correction failures. These findings do not
yet identify a complete mediator of the natural width effect.

## Current empirical position

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
does not include a separate implementation distribution or a fitted stochastic
closure. Those later alternatives remain recoverable from commit `92b7904` but
are outside the current theory.
