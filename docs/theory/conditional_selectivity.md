# One network: preserving conditional selectivity

23 October 2026. Follow-up to [innovation survival](innovation_survival.md).
Preferred research hypothesis, a conditional reduced-model result, and a small
ordinary-network falsification probe. The reduced-model width law below is **not
yet a theorem for the jointly trained network**.

## 1. The proposed simplification

The earlier model separated an already known default from a correction branch.
The natural extension is to separate **parts of the task in function space**, while
letting one ordinary network learn everything.

The high-level task assumption is:

> A small number of conditional distinctions are necessary beyond the main
> effects. Main effects learn quickly. Early feature learning can make the
> network insensitive to those conditional distinctions before they are fitted.

A conditional distinction means that a predictor is useful differently in
different contexts. Examples include a subgroup-specific slope, an interaction
between two attributes, or a contextual exception to an otherwise adequate
rule. This does not require a mixture-of-experts architecture, explicit routing,
teacher neurons, or a fixed output sign.

The essential object is **conditional learning rate**: how effectively can the
current network change the required interaction after compensating for changes
to the main effects? Under additional dynamical assumptions, its distribution
gives a smooth version of “probability of losing all routes times damage.”

This extension trades an exact elementary activation boundary for a more useful
single-network distinction between current representation and ability to learn.
It retains a concrete task structure so it is not a generic reachability object.

## 2. Task assumptions before network assumptions

Let \(R\in\{0,1\}\) be a context, \(\Pr(R=1)=q\), and let \(Z\) be a predictive
attribute. For the clean first family, assume \(R\perp Z\),
\(\mathbb E h(Z)=0\), \(\mathbb E h(Z)^2=1\). Consider

\[
 g(R,Z)=g_{\rm main}(R,Z)+d\,s(R,Z),\qquad
 s(R,Z)=\frac{(R-q)h(Z)}{\sqrt{q(1-q)}}.
 \tag{1}
\]

Here \(g_{\rm main}\) lies in a specified finite-dimensional space \(\mathcal C\)
of main effects, and \(s\perp\mathcal C\), \(\|s\|_{L_2(P)}=1\). The choice of
\(\mathcal C\) comes from the task semantics or an independently specified
additive model, never from which functions a failed network happened to learn.

For example, if
\(g(R,Z)=A+D R h(Z)\), then its best additive predictor is
\(A+qD h(Z)\). The conditional part is
\(D(R-q)h(Z)\), and

\[
 d=D\sqrt{q(1-q)},\qquad
 \boxed{\mathcal D=\tfrac12d^2=\tfrac12D^2q(1-q).}
 \tag{2}
\]

This is the exact excess half-MSE from losing the interaction while still
learning the additive predictor. It replaces the fixed-baseline damage
\(qD^2/2\): one jointly trained network is allowed to absorb part of the
exception into its main effects.

The same construction works with dependent inputs by projecting a **specified**
conditional predictor onto \(\mathcal C^\perp\). The interpretation then depends
on the chosen task decomposition. Not every data law has a useful low-dimensional
one. The strongest simplifying assumption is that one interaction, or a few
approximately decoupled interactions, dominate the remaining error.

Rarity can create a weak initial interaction signal, but rarity alone does not
prove delayed learning. The assumptions needed for that dynamical step are
explicit below. In particular, rare-context main effects can themselves be slow.

## 3. One network, all layers trainable

Start with an ordinary two-layer network

\[
 f_\theta(x)=c+\frac1m\sum_{r=1}^m a_r\sigma(w_r^\top x+b_r),
 \tag{3}
\]

where all parameters, including \(c\), train. Nothing is preassigned to a main
effect or an exception. A learned intercept is optional; it is not a supplied
default. The following local object also makes sense for deeper networks.

Use squared loss and gradient flow
\(\dot\theta=-G_\theta\nabla L\), with a positive semidefinite optimizer metric.
Ordinary gradient descent has the identity metric after choosing parameter
scales. For (3), the experiments use particle learning rates \(m\) and intercept
rate one, which yields a mean-normalized kernel. A frozen positive
preconditioner can also be included. Momentum and Adam's evolving state require
an enlarged dynamical model; they are not covered simply by calling \(G\) a
metric.

Let \(c_1,\ldots,c_k\) be an orthonormal basis for \(\mathcal C\). Define the
metric-scaled parameter gradients of task coefficients:

\[
 j_i=G_\theta^{1/2}\nabla_\theta\langle f_\theta,c_i\rangle,
 \qquad j_s=G_\theta^{1/2}\nabla_\theta\langle f_\theta,s\rangle.
\]

Put the \(j_i^\top\) in the rows of \(J_C\), and form
\(K_{CC}=J_CJ_C^\top\), \(K_{Cs}=J_Cj_s\), and \(K_{ss}=\|j_s\|^2\).
The proposed object is

\[
 \boxed{\kappa_s(\theta)
 =\|\Pi_{\ker J_C}j_s\|^2
 =K_{ss}-K_{sC}K_{CC}^{\dagger}K_{Cs}.}
 \tag{4}
\]

It is the squared best interaction response from a unit metric-sized update
that leaves the main-effect coefficients unchanged to first order:

\[
 \kappa_s=\max_{\|u\|\le1,\ J_Cu=0}\langle j_s,u\rangle^2.
\]

This is a familiar Schur complement, not a new algebraic construction. Its value
here is the **task-defined interaction** and the separation of direct useful
response from changes that can only be obtained by disturbing already learned
main effects. It includes readout changes, hidden-weight changes, and coordinated
compensation across neurons. No “one neuron, one role” assumption is needed.

### Why activation or feature counts are insufficient

A feature can have zero interaction moment while its trainable incoming weights
have a nonzero interaction derivative. On the four-point context task, take
one neuron with preactivation equal to the context and another with constant
positive preactivation. Neither current feature has an interaction moment.
The first neuron's weight on \(Z\) creates a context-specific response; the
second can cancel its additive part. In the executable check,
feature interaction energy is zero but \(\kappa_s=.0445545>0\).

This is precisely why the fixed-readout activation criterion should not be
transplanted unchanged into a fully trainable network.

## 4. The assumptions that would make this predictive

The proposed package has four substantive parts. Each can fail separately.

**A. A simple conditional remainder.** After the main effects, the remaining
Bayes predictor has one dominant contrast \(s\), or a few weakly coupled ones.
This identifies both the demand and its risk scale from the distribution.

**B. Main effects adapt faster than the conditional remainder.** Around the
relevant early state, \(K_{CC}\) is well conditioned and relaxes main-effect
errors faster than the interaction evolves. Redundant representation of main
effects or relatively fast readout adaptation can make this plausible. It is
not automatic from small \(q\), and must be checked along trajectories.

**C. Early training selectively removes sensitivity.** The main-effect-driven
flow changes the distribution of conditional responses: some particles retain
substantial response; others become nearly insensitive. This is stronger and
more predictive than saying all gradients get smaller. Uniform attenuation
reduces the population rate and cannot generally be repaired by width.

For ReLU networks, a natural realization is loss of context-dependent gate
patterns. If all gates are constant over the relevant support, both readout and
hidden derivatives are affine functions, so an interaction orthogonal to affine
functions has zero first-order response. A neuron need not be dead: it may be
very active and useful for main effects yet unable to express their interaction.

**D. Sensitivity does not regenerate immediately.** Over the prediction horizon,
the post-early conditional geometry is sufficiently stable. Exact zero tangent
response is not by itself an irreversible obstruction: nonlinear movement,
gate crossings, noisy minibatches, or another layer can restore it. For ReLU,
gate margins and bounds on movement provide a concrete way to test stability.
For smooth activations, use small integrated sensitivity, not exact extinction.

These four assumptions define the mechanism. To obtain a particularly simple
**width law**, one more statistical closure is needed: particles approximately
follow a width-stable shared early field, so their retained responses are
approximately independent conditional on that field. This is an approximation
to justify, not a consequence of independent initialization at all future times.

## 5. From conditional learning rate to risk

Suppose the residual is contained in \(\mathcal C\oplus\operatorname{span}\{s\}\).
Let its main-effect coefficients be \(r_C\), and its interaction coefficient
be \(b=\langle f-g,s\rangle\). The exact coefficient dynamics are

\[
 \dot r_C=-K_{CC}r_C-K_{Cs}b,\qquad
 \dot b=-K_{sC}r_C-K_{ss}b.
 \tag{5}
\]

Fast main-effect equilibration gives
\(r_C\simeq-K_{CC}^{\dagger}K_{Cs}b\), hence

\[
 \dot b\simeq-\kappa_s b,
 \qquad
 \boxed{L_s(T)\simeq L_s(\tau)
   \exp\left[-2\int_\tau^T\kappa_s(t)\,dt\right].}
 \tag{6}
\]

This is an interaction-loss prediction; total risk is approximately the same
only when the main-effect and omitted residuals are small.

The approximation has a measurable error. Set
\(\eta=r_C+K_{CC}^{\dagger}K_{Cs}b\). Then, under the stated residual closure,

\[
 \dot b=-\kappa_s b-K_{sC}\eta.
 \tag{7}
\]

Variation of constants bounds the forecast error by the accumulated
\(|K_{sC}\eta|\), damped by the later \(\kappa_s\). Residual components outside
the chosen task space add another explicit coupling term. Thus a failed scalar
forecast can be attributed to failure of the fast-main-effect approximation,
omitted task modes, or a wrong forecast of future sensitivity; it should not be
repaired by silently redefining the exception.

The integral in (6) must be **forecast before continuation**, for example from a
stable post-early kernel or a reduced population flow. Measuring the entire
future trajectory gives an explanatory identity, not a prediction.

## 6. A smooth probability-times-damage law

For a mean-normalized two-layer network, the kernel is an average of particle
contributions. At a fixed state, (4) can be written

\[
 \kappa_s=\frac1m\sum_{r=1}^m
  \|j_{s,r}-\beta^\top J_{C,r}\|^2
  +\text{contribution of shared parameters},
\quad \beta=K_{CC}^{\dagger}K_{Cs}.
 \tag{8}
\]

Local derivatives in (8) are of the unnormalized particle contribution; metric
factors are included consistently. The shared-parameter term, such as an
intercept used for compensation, must not be discarded. In general \(\beta\)
depends on all particles, so the summands are **not iid**.

For the clean reduced law, assume the shared term is negligible or deterministic,
\(\beta\) and the early population transport are width-stable, and the retained
local strengths are iid \(\xi_r\ge0\). Approximate the later rate by
\(\kappa_m=m^{-1}\sum_r\xi_r\), and assume early interaction learning was small,
so \(L_s(\tau)\simeq\mathcal D\). With \(H=T-\tau\),

\[
 \boxed{\mathbb E L_s(T)
 \simeq \mathcal D
 \left[\mathbb E_\xi e^{-2H\xi/m}\right]^m.}
 \tag{9}
\]

This is exact for the stated iid constant-rate scalar surrogate. Its application
to the neural flow depends on the preceding approximations. A deterministic
shared sensitivity \(\kappa_0\) multiplies the right side by
\(e^{-2H\kappa_0}\), and can eliminate the extinction floor altogether.

If \(\xi=\lambda\) with probability \(p\) and zero otherwise, (9) becomes

\[
 \mathcal D\left(1-p+p e^{-2H\lambda/m}\right)^m.
 \tag{10}
\]

For fixed width and sufficiently long horizon this tends to
\(\mathcal D(1-p)^m\). For fixed horizon and large width it tends to
\(\mathcal D e^{-2Hp\lambda}\). These are different limits: keeping one route
alive does not guarantee that it can train quickly under mean normalization.

For general \(\xi\), no alive/dead threshold is necessary. The lower tail of its
distribution controls finite-time failures. Under suitable moments,

\[
 \log\frac{\mathbb E L_s(T)}{\mathcal D}
 =-2H\mathbb E\xi+\frac{2H^2\operatorname{Var}\xi}{m}+O(m^{-2}).
 \tag{11}
\]

Expected risk is nonincreasing with width in this iid surrogate: apply Jensen
to the average of the leave-one-out sample means. This is not a monotonicity
theorem for neural training, where the strength distribution itself can change
with width.

The concise prediction is: **width reduces the risk of an unusually depleted
conditional response, but cannot restore a universally depleted population**.
If responses are correlated through a shared random early trajectory \(Z_0\),
the proper expression is a mixture of conditional laws,
\(\mathcal D\,\mathbb E_{Z_0}[(\mathbb E[e^{-2H\xi/m}\mid Z_0])^m]\).
An early trajectory that eliminates every response can leave a failure floor
even at infinite width. Substituting an invented “effective width” would hide
that distinction.

## 7. How to obtain the strength law prospectively

The distribution of \(\xi\) is not a parameter to fit from final success rates.
For a first derivation, run the mean-field particle dynamics driven by the
specified main-effect target:

\[
 \dot\theta_r=-\mathbb E[
   (f_{\mu_t}(X)-g_{\rm main}(X))\,\nabla_{\theta_r}\phi_{\theta_r}(X)].
 \tag{12}
\]

Here \(\phi_{\theta_r}=a_r\sigma(w_r^\top x+b_r)\), and the shared intercept
is evolved as well. This is shorthand for the ordinary particle gradient field,
not a detached fixed baseline. Its pushforward of the initialization law,
combined with the task moments in (8), predicts the strength distribution.

Its use requires a bound on the omitted interaction forcing during this early
period, and control of finite-width deviations. A small interaction residual
coefficient alone is insufficient if the flow amplifies the perturbation near
a gate boundary. Establishing a stable separation of trajectories is the
hard mathematical step. Existing mean-field results motivate the approach;
they do not automatically deliver it for this task or at arbitrary long times.

For initial empirical work, estimate the early transport on separate pilot seeds
and freeze it before forecasting new widths and task variants. Compare predicted
\(\xi\) distributions as well as risk. Measuring \(\kappa\) at a checkpoint is
a weaker but legitimate continuation forecast; it does not establish (12).

## 8. Small experiment in one ordinary network

Source: [explore_conditional_selectivity.py](../../scripts/studies/explore_conditional_selectivity.py).
Results: [JSON](conditional_selectivity_results.json),
[paired arrays](conditional_selectivity_results.npz).

Use four population points \((R,Z)\), \(R\sim\mathrm{Bernoulli}(.1)\),
\(Z\in\{-1,1\}\) balanced and independent, target \(2+2RZ\). The specified main
space is \(\operatorname{span}\{1,R,Z\}\), and the interaction damage is .18.
Train incoming weights, hidden biases, readouts, and an output intercept from
independent small Gaussian initialization, with no assigned neuron signs or
roles. Width three has the explicit exact representation

\[
 2+[\mathrm{ReLU}(c+z)-\mathrm{ReLU}(c-z)],\qquad c=2R-1,
\]

using a third constant hidden unit if the output intercept is omitted. Thus all
tested widths can represent the task. The code verifies the witness.

128 paired starts, nested widths 3/6/12, population gradient descent, step .05,
horizon 1000. A checkpoint at time 20 forecasts continuation by freezing the
measured kernel. This time was selected as a simple early checkpoint before
seeing outcomes, not optimized for accuracy. Controls and the full-kernel
comparison were added after the first run; this is exploratory evidence.

| Width | Mean final half-MSE, learned intercept | Without output intercept |
|---:|---:|---:|
| 3 | .0360156 | .131094 |
| 6 | .00367188 | .00765625 |
| 12 | 7.88e-30 | 1.31e-8 |

With the learned intercept, width-three outcomes comprise 91 near-zero runs,
25 near .09, 2 near .18, and 10 near .20. Width six has 124 near zero, 3 near
.09, and 1 near .20. These are observed numerical plateaus, not retrospectively
defined task roles. In particular, **the common damage .18 is not a universal
failure cost in the unrestricted network**. Some trajectories miss only part
of the conditional structure; others also miss part of the main effects.

The scalar frozen-rate forecast at time 20 has mean absolute interaction-loss
error .00882, .00101, and 4.36e-11 for widths 3,6,12. At width three its predicted
mean .02593 is close to observed interaction loss .02566, but individual errors
are substantial: agreement of means is not good calibration.

The full four-mode frozen-kernel forecast uses
\(\tfrac12\|e^{-HK(20)}r(20)\|^2\), with no fast-main-effect reduction. Its
mean absolute total-loss errors are .00916, .00192, and 1.63e-7 with the learned
intercept. Without the intercept, its width-three error is only 1.75e-9. That
stronger result forecasts stable deficient tangent spaces, but also includes
failures to learn the default. It does not validate the simpler scalar law.

Numerical rank diagnostics at tolerance 1e-12 show 13 width-three starts with
zero initial conditional rate and 25 at time 20; 17 starts newly lose it, and
all 17 retain loss above .01 at the final horizon. No early-zero run reaches
near-zero final loss. This tolerance is a floating-point rank diagnostic,
**not a theoretical viability threshold**. Future tests should record actual
gate-pattern changes and margins rather than rely on numerical zero tests.

Two crucial negative/limiting results:

- Removing the large default constant, keeping the interaction and its damage
  unchanged, gives width-three mean loss .03375 versus .03602. The paired
  improvement .00227 has an approximate 95% normal interval of ±.01145 around
  that mean. **This does not establish that pressure to learn the default
  caused the width gap.** The intercept may absorb that pressure, or another
  mechanism may cause the conditional degeneration.
- Current feature interaction moments can all vanish while the trainable
  network retains positive conditional learning rate, as the explicit
  counterexample in Section 3 verifies.

Halving the step and extending training fourfold on the same first 32
width-three starts both retain mean loss .0196875, equal to the original subset.
A central-finite-difference audit of the complete metric-scaled kernel agrees
within 9.19e-13. These checks cover the learned-intercept arm; they are not a
claim of optimizer or distribution robustness. Only four input atoms are used.
The saved results record the source hash and all main settings.

The appropriate conclusion is **a width benefit and consequential conditional
degeneration in one fully trainable network, with incomplete evidence for its
proposed early-default cause**. The iid width law (9) has not been fit or tested
against this cohort; doing so now from these endpoints would be circular.

## 9. Predictions and next discriminating work

The proposal is most useful if it predicts the following together:

1. At fixed interaction damage, stronger early *feature-level* main-effect
   pressure shifts the retained sensitivity distribution toward zero. Changing
   a constant label mean need not create such pressure when an intercept absorbs
   it; the negative control above makes this distinction necessary.
2. Earlier exposure or upweighting of the interaction changes that distribution,
   while changing only the evaluation weight changes its damage. Evaluate all
   interventions under the original task distribution.
3. Width mainly removes the lower tail of conditional learning rates. Uniform
   shrinkage of every route, or a common trajectory destroying all routes,
   predicts weak or saturating width gains.
4. Fast adaptation of main-effect predictions should improve the accuracy of
   the scalar reduction. If it does not, the task has multiple important slow
   contrasts or the geometry is changing too much.
5. A reserved random feature, gate-preserving update, or small targeted
   perturbation should rescue a narrow network when it restores conditional
   rate. A comparable intervention that changes activity without restoring the
   rate should fail. The feature-only counterexample says why this control matters.
6. A short-lived zero rate followed by ordinary recruitment refutes permanent
   route loss. Forecast integrated sensitivity on a specified horizon instead.

The next decisive experiment should use a nonconstant main-effect target and
a rare contextual interaction, with common modes demonstrably fast. Learn both
in one network on a bounded continuous distribution. Forecast the sensitivity
law from main-effect-only dynamics on separate seeds; then compare widths,
early interaction upweighting, and readout-rate changes on new seeds. Report
the main-effect tracking error in (7), gate regeneration, and the strength
distribution, not only final error. The binary survival law is only justified
if sensitivity itself shows a well-separated near-zero component and stable
positive component.

## 10. What is established and what remains a hypothesis

The task decomposition, damage calculation, Schur-complement geometry, and
coefficient equations are exact under their explicit definitions. The Laplace
formula is exact for its iid scalar surrogate. Fast-main-effect elimination,
stable post-early sensitivity, and propagation of the initialization law through
a shared early field are hypotheses/approximations for ordinary neural training.
The probe shows both their promise and a real failure of a simple causal test.

This is not yet a stronger theorem than the fixed-sign erasure model. It is a
more natural **research hypothesis for a single network**. Its elegance comes
from three quantities with separate jobs:

\[
 \boxed{\text{interaction damage }\mathcal D,\quad
        \text{retained sensitivity law }\nu_\tau,\quad
        \text{learning horizon }H.}
\]

The earlier survival probability is the zero-versus-positive special case of
the sensitivity law. The essential research problem is to derive how early
training changes that law, while allowing all the recovery routes that a
fully trainable network actually has.

Relevant established ingredients: the shared particle-flow approximation is
motivated by [Mei, Montanari & Nguyen (2018)](https://arxiv.org/abs/1804.06561),
whose mean-field results do not imply the independence closure at arbitrary
finite widths and long times. The possibility that early alignment removes
useful diversity is supported in specific settings by
[Boursier & Flammarion (2025)](https://www.jmlr.org/papers/v26/24-1523.html).
Neither establishes this task-specific width law or its novelty.

## Reproduce

```sh
.venv/bin/python scripts/studies/explore_conditional_selectivity.py
.venv/bin/ruff check scripts/studies/explore_conditional_selectivity.py
```
