# Width as dynamic option value: reachable moments and release-time viability

23 October 2026. Exploratory theory proposal. The moment certificate and the
occupancy identities below are elementary consequences of the stated assumptions.
The substantive unproved step is to derive the release-time viable sets for an
actual neural training flow before observing its finite-width outcomes.

## 1. Proposed mechanism

The working claim is not that width universally reduces simultaneous gradient
conflict. It is:

> A task can contain representation demands that are structurally necessary at
> the endpoint but locally disfavored until other demands have been learned. A
> finite network fails when too few neurons remain capable of serving one of
> these delayed demands at its release time. Width helps by supplying redundant
> particles, so premature commitments do not exhaust every route to that demand.

This separates four objects that should not be conflated:

1. **Structural demand:** what properties any low-risk predictor must have.
2. **Release:** when a demand first becomes favorable under the evolving residual.
3. **Viability:** which current neurons can still serve it by the target horizon.
4. **Damage:** the objective error forced when viable coverage is insufficient.

The signed-role experiment is one realization. The intended scope is broader:
latent directions in multi-index regression, class-contrast features, mixture
components, subgroup-specific features, or hierarchical features that become
learnable at different stages.

## 2. A distribution-free loss certificate

Let \(g(x)=\mathbb E[y\mid x]\) and use population squared excess risk

\[
\mathcal E(f)=\frac12\|g-f\|_{L_2(P_X)}^2.
\]

Choose a vector of probe functions \(s(x)\in\mathbb R^q\), with
\(\Sigma=\mathbb E[s(x)s(x)^\top]\), and define the linear moment map

\[
M(h)=\mathbb E[h(x)s(x)].
\]

Projection onto the span of the coordinates of \(s\) gives, for every square-
integrable \(h\),

\[
\|h\|_2^2\ge M(h)^\top\Sigma^\dagger M(h).
\]

Consequently, if a set \(\mathcal K\subseteq\mathbb R^q\) contains every moment
vector reachable by a candidate model class, then

\[
\inf_{M(f)\in\mathcal K}\mathcal E(f)
\ge
\frac12\operatorname{dist}_{\Sigma^\dagger}^2(M(g),\mathcal K).
\tag{1}
\]

This is teacher-free: \(M(g)=\mathbb E[ys(x)]\) is defined from the data law.
The choice of probes is useful only when individual neurons have constrained
moment geometry. The Gaussian second-Hermite certificate is the example already
established in this repository. Higher Hermite tensors, density-score moments,
and vector-valued class-contrast moments are possible alternatives.

For losses other than squared error, (1) remains a local or restricted risk
certificate when excess risk lower-bounds a squared prediction norm. For example,
cross-entropy has such a bound on logit-contrast regions where its Hessian is
uniformly positive. Separable limits and unrestricted common-logit directions
require separate treatment.

## 3. The future-option deficit

Fix a training state at time \(t\), an algorithm, and a horizon \(T\). For neuron
\(r\), let \(\mathcal C_{r,t\to T}\subseteq\mathbb R^q\) be the set of moment
changes that this neuron can still produce by \(T\) under the stated dynamics
and admissible stochasticity. It should exclude roles that have become unreachable
on that horizon, not merely directions that are inactive at time \(t\). A dormant
unit can therefore retain a nontrivial option set.

The aggregate reachable moment set is the Minkowski sum

\[
\mathcal C_{t\to T}^{(m)}
=\mathcal C_{1,t\to T}+\cdots+\mathcal C_{m,t\to T}.
\]

Define the **future-option deficit**

\[
\boxed{
\mathfrak D_{t,T}^{(m)}
=\frac12\operatorname{dist}_{\Sigma^\dagger}^2
\left(M(g-f_t),\mathcal C_{t\to T}^{(m)}\right).
}
\tag{2}
\]

Whenever every actual continuation satisfies
\(M(f_T-f_t)\in\mathcal C_{t\to T}^{(m)}\), equation (1) implies

\[
\mathcal E(f_T)\ge\mathfrak D_{t,T}^{(m)}.
\]

The object has the basic option-value property: adding a neuron while leaving the
existing option sets unchanged enlarges the Minkowski sum, so \(\mathfrak D\)
cannot increase. The marginal value of an extra neuron is the reduction in (2),
not its immediate gradient alignment. An option can be valuable while dormant
and while following it would temporarily increase the current loss.

Equation (2) is general but not automatically useful. If the reachable sets are
defined from observed future paths, it is retrospective. The research problem is
to derive conservative \(\mathcal C_{r,t\to T}\) from a reduced population model,
basin geometry, and a release rule specified before finite-width outcomes.

## 4. A tractable stagewise-basin reduction

Assume the task supplies \(J\) moment roles. A role can be a latent subspace or
class contrast; it need not be an individual teacher neuron. For role \(j\):

- \(\mathcal B_j\) is a recruitment basin in particle state space;
- \(r_j\) is the number of viable atoms required for the desired moment rank;
- \(\varepsilon_{j,1}\ge\varepsilon_{j,2}\ge\cdots\ge0\) are the certified
  loss costs of successively missing atoms;
- \(\tau_j\) is its release time.

One possible prospective definition starts from the best directional utility in
the role cone \(\mathcal H_j\):

\[
u_j(t)=\sup_{h\in\mathcal H_j,\ \|h\|_2\le1}
       -D\mathcal L(f_t)[h],
\tag{3}
\]

Absolute utility is not enough: a weak role can be favorable from the start and
still lose every neuron to a stronger use. Let \(\mathcal A_j(t)\) be the roles
that compete for the same particles and let \(\kappa_j(t)\ge0\) encode a
commitment or activation barrier. Define the recruitment margin and release by

\[
\rho_j(t)=u_j(t)-\max_{k\in\mathcal A_j(t)}u_k(t)-\kappa_j(t),
\qquad
\tau_j=\inf\{t:\rho_j(t)\ge0\}.
\tag{3a}
\]

In the signed-role fixture this can reduce to a gradient sign change. In a
spectral allocation problem it is an eigenvalue crossing: the delayed role can
have positive absolute utility before release but negative relative utility.
Particle-specific basin calculations should replace (3a) when different neurons
face different alternatives.

For the first tractable theory, assume \(\rho_j\) has one release rather than many
noisy crossings. Let \(\Phi_{0,t}^{\bar\mu}\) denote the characteristic flow
under a width-stable population trajectory \(\bar\mu_t\). Define the initial
conditions that remain recruitable at release,

\[
V_j=\left\{\theta_0:
\Phi_{0,\tau_j}^{\bar\mu}(\theta_0)
\text{ can reach }\mathcal B_j\text{ by }T\right\},
\qquad p_j=\mu_0(V_j).
\tag{4}
\]

The number \(p_j\) is the **release-time viable mass**. It combines basin volume,
pre-release suppression, initialization scale, optimizer dynamics, and the delay
before release. It is not the fraction of units currently active in the role.

Under an independent-characteristic approximation, the viable count satisfies

\[
N_j^{(m)}\sim\operatorname{Binomial}(m,p_j).
\]

If the moment blocks are orthogonal and \(N_j\) viable atoms remove the first
\(N_j\) spectral costs, (2) reduces to the **release-time occupancy deficit**

\[
\boxed{
\mathfrak D_m
=\sum_{j=1}^J\sum_{i>N_j^{(m)}}\varepsilon_{j,i}.
}
\tag{5}
\]

The signed Gaussian bound is recovered with two roles, positive and negative,
and \(\varepsilon_{\pm,i}=\lambda_{\pm,i}^2/2\). An orthogonal multi-index model
with one required atom per latent direction has \(r_j=1\).

### 4.1 The release-time option Gramian

Raw counts are too special for a latent role of dimension greater than one. A
continuous version is obtained by linearizing only the *post-release* phase. Work
in whitened moment coordinates for a delayed role, and let
\(b_j=\Sigma_j^{-1/2}M_j(g-f_{\tau_j})\in\mathbb R^{q_j}\) be its residual demand
at release. Let \(J_{rj}\) map an admissible small displacement of neuron \(r\)
at release to its terminal contribution in this moment block. This can be a
transported Jacobian through the remaining dynamics; it is not necessarily the
current network Jacobian. Define the **release-time option Gramian**

\[
G_j^{(m)}
=\sum_{r=1}^m J_{rj}J_{rj}^{\top}.
\tag{5a}
\]

The network parameterization and learning-rate convention determine the scale
inside \(J_{rj}\). For a fixed total post-release displacement budget \(B\), the
linearized future-option deficit is

\[
\mathfrak D_{j,B}^{\mathrm{lin}}
=\frac12\min_{\sum_r\|u_r\|^2\le B}
\left\|b_j-\sum_rJ_{rj}u_r\right\|^2.
\tag{5b}
\]

Under a nested-width comparison that preserves the existing columns and their
scale, adding a neuron appends feasible coordinates, so (5b) cannot increase.
If exact correction is possible, the minimum displacement energy is

\[
E_j^*(b_j)
=
\begin{cases}
b_j^\top\left(G_j^{(m)}\right)^\dagger b_j,
  &b_j\in\operatorname{range}(G_j^{(m)}),\\
+\infty,&\text{otherwise}.
\end{cases}
\tag{5c}
\]

Thus early competition has a quantitative meaning: it removes columns, makes
them collinear, or attenuates them before the role is released, increasing (5b)
or (5c). A dormant unit retains option value if its transported Jacobian is
nonzero.

This is a role-restricted, future-transported analogue of a feature or tangent
Gram matrix. Its purpose is not to relabel generic kernel conditioning as
competition. The relevant event is deterioration of the Gramian *before a
prospectively defined release time* for a moment block that certifies loss.

### 4.2 An exact latent-subspace width law

There is a solvable model that is broader than signed scalar roles. Suppose a
delayed role is a \(q\)-dimensional latent subspace. At its release, each neuron
independently remains viable with probability \(p\). Conditional on viability,
its future option vector is isotropic, \(z_r\sim N(0,I_q)\); otherwise \(z_r=0\).
With the usual mean-preserving scaling, take

\[
G_m=\frac1m\sum_{r=1}^m z_rz_r^\top,
\qquad K\sim\operatorname{Binomial}(m,p).
\tag{5d}
\]

Conditional on \(K=k\), the option span is a uniformly oriented subspace of
dimension \(\min(k,q)\). Therefore, for any fixed residual demand \(b\), the
expected certified moment-loss component that is unreachable even with unlimited
post-release time is exactly

\[
\boxed{
\mathbb E\mathfrak D_m^{\mathrm{rank}}
=\frac{\|b\|^2}{2q}\,
  \mathbb E\left[(q-K)_+\right].
}
\tag{5e}
\]

The target can be represented by \(q\) deliberately placed neurons, yet an
ordinary random width-\(q\) initialization has too few viable options with high
probability whenever \(p<1\). The characteristic random-initialization width is

\[
m_{\mathrm{cover}}\asymp \frac{q+\log(1/\delta)}{p},
\tag{5f}
\]

up to constants. The exact expected value of one additional neuron is

\[
\mathbb E\mathfrak D_m^{\mathrm{rank}}
-\mathbb E\mathfrak D_{m+1}^{\mathrm{rank}}
=\frac{\|b\|^2}{2q}\,p\,
  \Pr\{\operatorname{Bin}(m,p)<q\}.
\tag{5g}
\]

This is an explicit account of overparameterization that is not an expressivity
claim: strategic width \(q\) suffices, whereas ordinary width must be inflated by
roughly \(1/p\) because only a release-time fraction \(p\) remains usable.

Rank is only the infinite-horizon part. Conditional on \(K=k\), a standard
Gaussian singular-value bound gives, with probability at least \(1-\delta\),

\[
\lambda_{\min}(G_m)
\ge
\frac{1}{m}
\left(\sqrt{k}-\sqrt q-\sqrt{2\log(1/\delta)}\right)_+^2.
\tag{5h}
\]

If the released residual follows the locally linear gradient flow
\(\dot b=-G_m b\), then

\[
\mathcal E_j(T)
\le \frac12e^{-2\lambda_{\min}(G_m)(T-\tau_j)}\|b_j\|^2.
\tag{5i}
\]

For \(K\approx mp\), the rate in (5h) is approximately

\[
p\left(1-\sqrt{q/(mp)}\right)_+^2.
\tag{5j}
\]

This predicts a sharp onset around \(mp=q\), poor conditioning just above the
onset, and saturation at a population rate controlled by \(p\). It also exposes
an important boundary of the mechanism. If early training smoothly attenuates
*every* option by the same transport \(z_r=S\xi_r\), width improves sample
conditioning but cannot remove the population factor \(s_{\min}(S)^2\). A large
mean-preserving width benefit therefore requires selective commitment (a small
surviving fraction), anisotropic loss of option diversity, or width-dependent
finite-particle feedback—not merely uniform gradient shrinkage.

## 5. Exact width identities in the occupancy model

Taking expectations in (5) gives

\[
\boxed{
\mathbb E\mathfrak D_m
=\sum_{j=1}^J\sum_{i=1}^{r_j}
  \varepsilon_{j,i}\,\Pr\{\operatorname{Bin}(m,p_j)<i\}.
}
\tag{6}
\]

This is an exact loss-floor prediction under the stagewise-basin assumptions.
For equal per-missing-role cost \(\varepsilon_j\),

\[
\mathbb E\mathfrak D_m
=\sum_j\varepsilon_j
\sum_{k=0}^{r_j-1}(r_j-k)
\binom{m}{k}p_j^k(1-p_j)^{m-k}.
\tag{7}
\]

Adding one particle yields the exact marginal option value

\[
\mathbb E\mathfrak D_m-\mathbb E\mathfrak D_{m+1}
=\sum_j p_j\,
\mathbb E\!\left[
\varepsilon_{j,N_j^{(m)}+1}
\mathbf 1\{N_j^{(m)}<r_j\}
\right].
\tag{8}
\]

It is nonnegative and has diminishing returns when the spectral costs decrease.
Width is most valuable for roles that are simultaneously expensive
(large \(\varepsilon\)), rare at release (small \(p_j\)), and not yet safely
covered. This is a more specific prediction than "wider is better."

If \(p_j>(r_j-1)/m\), a binomial Chernoff bound gives

\[
\Pr\{N_j^{(m)}<r_j\}
\le
\exp\{-mD((r_j-1)/m\,\|\,p_j)\}.
\tag{9}
\]

The characteristic width scale is therefore \(m_j\approx r_j/p_j\), with a
logarithmic surcharge to cover many roles simultaneously. In the symmetric
one-atom case \(p_j=1/J\),

\[
\Pr\{\text{some role missing}\}
\le J(1-1/J)^m\le J e^{-m/J},
\]

so \(m\gtrsim J\log(J/\delta)\) gives complete coverage with probability
\(1-\delta\). This is the familiar coupon-collector effect. Existing Gaussian
multi-index theory already proves a closely related \(k\log k\) width requirement
when neurons converge to their nearest orthogonal index direction. The proposed
extension is to replace static basin probability by the dynamic viable mass in
(4), and to weight missing coverage by an objective certificate.

This cleanly separates expressivity from optimization. A hand-assigned width
\(m_0=\sum_jr_j\) network can represent the target, while an ordinarily initialized
network of that width has a large probability of exhausting at least one role.
Extra particles change the probability of finding the representation, not the
existence of the representation.

## 6. Finite-particle feedback is probably essential

The independent-characteristic model is a reference, not the final theory.
Neurons share the residual, so an initial shortage of one role can change the
field, induce further conversions, and make the shortage self-reinforcing.
Conversely, enough particles can stabilize the population trajectory near a
mean-field path that retains positive viable mass.

A deliberately crude audit of the existing 30-degree/(1,-3) cohort illustrates
the issue. If the terminal width-32 positive fraction is used as a fixed survival
probability for eight independent particles, the predicted versus observed mean
positive-role deficit \((4-N_+)_+\) at width 8 is:

| Optimizer | Width-32 positive fraction | Independent prediction | Observed width-8 deficit |
| --- | ---: | ---: | ---: |
| SGD | .434 | .847 | 1.750 |
| Momentum | .420 | .920 | 1.800 |
| Adam | .214 | 2.303 | 3.400 |

Terminal positive fraction is not a valid estimate of (4), so this is not a
formal model rejection. It does show that "width merely draws more independent
tickets from one fixed probability" is too weak a default explanation. The
theory should allow the viable mass itself to depend on empirical-measure
fluctuations.

The loss-weighting side of the proposal has somewhat better retrospective
support. In the main ordinary-width-8 sign-survival cohort, the within-optimizer
Pearson correlation between the terminal signed spectral floor and population
MSE is .836 for SGD, .878 for momentum and .907 for Adam. The floor averages
31.0%, 30.1% and 37.0% of population MSE, respectively. At ordinary width 32 it
is zero in every SGD/momentum run and nonzero in only one Adam run. These are
endpoint correlations, not evidence for a release-time Gramian, but they suggest
that weighting missing options by an observable moment obstruction is materially
better than studying role counts alone.

A useful refinement is a **separatrix margin**. Let \(A_j\) be the boundary
between states that remain viable and states committed elsewhere, and define

\[
\delta_j
=\inf_{0\le t\le\tau_j}
\operatorname{dist}
\left(\Phi_{0,t}^{\bar\mu}(V_j),A_j(t)\right).
\tag{10}
\]

If \(\delta_j>0\) and the particle vector field is sufficiently regular,
finite-width propagation-of-chaos bounds can transfer the population viable
mass to the particle system. The expected scale is a concentration condition
of the form

\[
m p_j-r_j
\gg
\sqrt{m p_j(1-p_j)}
+m\,\mathrm{Err}_{\rm field}(m,T,\delta_j).
\tag{11}
\]

Equation (11) is not proved here. It states the intended technical program.
When the population path stays far from a separatrix, width should give ordinary
concentration. When \(\delta_j\) is small, \(O(m^{-1/2})\) field fluctuations can
change basin membership and produce a sharp, optimizer-sensitive width effect.
If the population viable mass is exactly zero, width alone should not rescue the
task in the width-stable limit.

## 7. Natural idealized settings

### 7.1 Gaussian multi-index and hierarchical targets

Let

\[
x\sim N(0,I_d),\qquad
g(x)=G(v_1^\top x,\ldots,v_k^\top x)
\]

with a low-dimensional latent subspace. Hermite moment tensors of \(g\) expose
latent directions without requiring \(g\) to be a ReLU teacher. A ridge neuron
contributes a rank-one symmetric tensor in each Hermite degree. For orthogonally
decomposable moments, missing latent directions force a spectral/tensor-tail
error through (1).

This setting is meaningful beyond the present fixture: multi-index models are a
standard abstraction of low-dimensional latent structure, and recent theory
shows sequential or staircase learning of their directions. The static
orthogonal case calibrates (6). Correlated directions, unequal Hermite degrees,
or hierarchical targets such as a main effect plus an interaction provide a
setting in which release times and pre-release capture can be derived.

A particularly clean first derivation is centered quadratic regression,

\[
g(x)=x^\top A_*x-\operatorname{tr}(A_*),\qquad x\sim N(0,I_d),
\tag{12}
\]

with normalized quadratic ridge features. This is a Gaussian multi-index model,
but its roles are observable eigenspaces of the label moment rather than signed
teacher units. The population loss is proportional to Frobenius error in the
represented matrix. If the residual moment matrix has a fixed eigenbasis,

\[
R(t)=\sum_i s_i(t)v_iv_i^\top,
\]

then the projected angular flow of a unit direction has the Oja form

\[
\dot w=(I-ww^\top)R(t)w.
\tag{13}
\]

Writing \(q_i(t)=(w(t)^\top v_i)^2\), equation (13) gives the exact log-odds
identity

\[
\frac{d}{dt}\log\frac{q_j(t)}{q_k(t)}
=2\{s_j(t)-s_k(t)\}.
\tag{14}
\]

Suppose \(k\) is the initially dominant role, \(j\) is delayed, and a unit can
serve \(j\) by horizon \(T\) only if
\(\log(q_j(T)/q_k(T))\ge h\). Define the pre-release competition debt and the
post-release recovery credit

\[
C_{jk}=2\int_0^{\tau_j}(s_k-s_j)\,dt,
\qquad
R_{jk}=2\int_{\tau_j}^{T}(s_j-s_k)\,dt.
\]

For isotropic initialization, \(q_j(0)/q_k(0)=Z_j^2/Z_k^2\) with independent
standard Gaussians. Hence the release-time viable probability is explicit:

\[
\boxed{
p_{j\mid k}(T)
=\frac{2}{\pi}\arctan
  \exp\left[-\frac{h+C_{jk}-R_{jk}}{2}\right].
}
\tag{15}
\]

Once the population residual trajectory supplies \(s_i(t)\), this predicts
\(p\) before simulating finite widths. Substitution into (5e)--(5j) gives the
whole width curve. When the net debt is large,
\(p\asymp e^{-(h+C_{jk}-R_{jk})/2}\), so the coverage width grows exponentially
in accumulated pre-release disadvantage. The derivation is exact under the
fixed-eigenbasis angular flow. In unconstrained quadratic matrix factorization it
describes finite-horizon slowness rather than an irreversible failure, because
nonzero latent coordinates can eventually recover. Nonlinear specialization or
a true basin boundary is needed to turn this soft viability threshold into
permanent option loss. That distinction is itself a useful experimental test.

### 7.2 Mixtures and subgroup-specific features

For separated mixture components, a feature or expert can be useful mainly on
one component. Early gradients may be dominated by a common or majority feature;
after that component is fitted, subgroup residuals become the remaining demand.
Probe moments can be component-conditional means or covariances. Viable basins
are regions of feature space that can still specialize to each component.

### 7.3 Multiclass contrasts

For vector outputs, replace scalar signs by output-contrast cones. A role is a
feature together with a direction in the logit subspace orthogonal to the common
logit. Target class-conditional moment matrices can impose rank requirements.
This is a plausible bridge to classification; raw counts of positive and negative
scalar readouts are not.

## 8. Prospective predictions

The framework makes the following falsifiable predictions before choosing a
specific successful task:

1. Width benefit is controlled by expensive rare roles: large
   \(\varepsilon_{j,i}\) and small release-time viable mass \(p_j\).
2. A scalar role should show a threshold near \(r_j/p_j\); a
   \(q_j\)-dimensional role should require effective viable width
   \(m p_j\gtrsim q_j\), followed by a conditioning regime and then saturation.
3. Equalizing initial role counts can fail when pre-release survival, rather than
   initial occupancy, is the bottleneck.
4. Delaying release by reducing the learning rate or strengthening an earlier
   demand should lower \(p_j\) and move the width threshold upward in update time.
5. Increasing initialization scale can reduce commitment, increase viable mass,
   and shrink the width effect, even with the same target moments.
6. If a derived population flow has \(p_j=0\), finite width should not show a
   persistent monotone rescue merely from adding particles.
7. If a role's certified cost is negligible, preserving it can change internal
   trajectories without materially changing loss.
8. With fixed aggregate scaling, the intervention value of preserving options
   should fall as width moves safely above every coverage threshold.
9. Uniform attenuation of all role options should mainly slow every width. A
   large width-specific rescue requires heterogeneous survival, anisotropic
   collapse of the option Gramian, or width-dependent residual feedback.
10. In the fixed-eigenbasis quadratic model, the logarithm of the width threshold
    should be affine in the integrated residual disadvantage in (15).

Several of these agree qualitatively with existing observations, but that is not
validation: the quantities \(p_j,\tau_j,\delta_j\) have not yet been predicted
prospectively in those experiments.

## 9. First theory-led experimental program

Do not begin with another arbitrary teacher sweep. First derive the reduced
population dynamics and viable sets for nested settings.

1. **Analytic dynamic calibration:** centered Gaussian quadratic regression with
   a fixed residual eigenbasis. Derive \(s_i(t)\), the release crossing, and the
   viable mass from (15), then predict the rank and conditioning curves
   (5e)--(5j) before finite-width runs.
2. **Boundary case:** compare unconstrained quadratic matrix factorization, where
   every nonzero latent coordinate can eventually recover. The theory predicts a
   finite-horizon width effect but no permanent option loss. This separates
   ordinary spectral slowness from hard competition.
3. **Hard-specialization extension:** use a nonlinear Gaussian multi-index or
   mixture target whose population angular dynamics has genuine basins. Derive
   their release-time mass rather than estimating it from endpoints. Here the
   same Gramian theory predicts persistent failure probability.
4. Vary a structural parameter that changes accumulated competition debt without
   changing minimal representational rank: eigenvalue ratio, latent correlation,
   hierarchy coefficient, or relative signal strength. The width threshold must
   move according to the predicted \(p_j\).
5. Record release-time option Gramian spectra, basin occupancy, moment tails, and
   population loss. The primary test is prediction of the width-by-task curve,
   not whether one intervention helps after the fact.

The dynamic setting is successful only if the object predicts both a regime with
a substantial width benefit and a regime where representationally redundant
width does little. A post hoc basin definition or a viable mass estimated from
the same finite-width endpoints would not test the theory.

## 10. Relation to existing theory

This proposal combines established ingredients rather than claiming them as new:

- two-layer networks as interacting particles and their mean-field limits
  ([Chizat and Bach, 2018](https://arxiv.org/abs/1805.09545),
  [Mei, Misiakiewicz and Montanari, 2019](https://proceedings.mlr.press/v99/mei19a.html));
- early alignment and its possible loss of feature diversity
  ([Boursier and Flammarion, 2025](https://www.jmlr.org/papers/v26/24-1523.html));
- sequential/staircase feature learning
  ([Abbe, Boix-Adsera and Misiakiewicz, 2023](https://proceedings.mlr.press/v195/abbe23a.html),
  [Dandi et al., 2024](https://www.jmlr.org/papers/v25/23-1543.html));
- static direction coverage in orthogonal Gaussian multi-index models
  ([Simsek, Bendjeddou and Hsu, 2025](https://proceedings.mlr.press/v258/simsek25a.html));
- feature imbalance under competing predictive signals
  ([Pezeshki et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)).

The candidate contribution is the combination of a *dynamic release-time viable
mass*, an objective-valued reachable-moment deficit, and a finite-width coverage
law. The key novelty question is not whether coupon-collector coverage or moment
bounds exist separately, but whether a release-time version can be derived for
ordinary feature-learning dynamics and predict when redundant width prevents a
loss of future representational options.

## 11. Current assessment

The broad object is the reachable-moment deficit (2). Its most promising
computable form is the release-time option Gramian (5a), evaluated through the
budgeted deficit (5b). The hard-survival model (5d)--(5j) and discrete occupancy
law (6)--(8) are its first solvable reductions. Together they capture the desired
distinction:

- the narrow network may be globally expressive;
- ordinary dynamics can shrink its finite-horizon reachable moment set;
- width supplies redundant directions and improves release-time rank and
  conditioning;
- missing coverage has a certified loss consequence;
- the benefit depends on task geometry and release time, not width alone.

Equation (15) already derives \(p_j\) from an integrated residual trajectory for
the fixed-eigenbasis quadratic flow. The immediate mathematical target is to
close that residual ODE and test its width curve, then derive \(p_j\), \(\tau_j\),
and a nontrivial separatrix margin for one genuinely basin-forming Gaussian
multi-index task. Until the latter derivation is complete, the persistent-failure
version remains a structured conjecture, not an explanation established by the
existing experiments.
