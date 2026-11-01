# Frequency-weighted pressures, finite gate changes, and width

**Status: exploratory branch, not the primary extension.** The direct jump to
parity changed several premises at once. Conditional SGD trajectory forecasting
is outside the current research target. The current work returns to successful
reserve predictions through [controlled relaxations](../../../docs/model/reserve_relaxation.md).
The exact diagnostics and failed assumptions below remain useful evidence.

31 October 2026. This extends the [shared-implementation dynamics](shared_implementation_competition.md)
to the [parity dataset](../../../docs/parity_dataset.md). It retains the spectral definition
of options and treats protected reserve as a conditional special case. See the
[executed evidence and limitations](../evidence/frequency_pressure/README.md).

## Result and scope

The useful generalization is **many signed residual pressures acting on coupled
neural implementations**, not a replacement of common pressure by a sum of
frequency-only destruction rates. Frequencies weight exposure and change target
correlations; current residuals, gates, and coefficients determine pressure.
Learning can create, strengthen, weaken, or recruit a response.

The foundation below gives exact decompositions of current pressure, loss
progress, and finite response change; a conditional Gaussian approximation to
ordinary SGD; and a loss-rate bridge in ordinary training time. The conditional
forecast is tested prospectively on independent future SGD streams. It evolves
the full parameter state and is not a compact analytic width law or an efficient
alternative to training. A causal explanation of the entire natural-width loss
gap remains open.

Two experimentally rejected shortcuts are important:

1. Smooth gate-cell response transport does not predict finite changes here.
   Gate crossings dominate them, even with stable learning rates.
2. Faster learning at larger width does not imply more reliable preservation of
   each width's own earlier residual response. These are different demands.

## 1. Distribution, factors, and task

Let independent factors satisfy X_i ~ Bernoulli(q_i), 0 < q_i < 1, and let
Y = sum_i X_i mod 2. The observed encoding is initially the raw bit vector with
an intercept; centering is an explicit control. These are known generative
factors, not an assumption about the network's learned features or neuron roles.

None of the identities requires a power law. The initial study uses
q_i = .4/i for eight factors, then tests flatter, geometric, equal-frequency,
centered-input, and initialization controls. Every condition has one fixed
distribution throughout training. Changing q also changes total activity,
label balance, and the available correlations. In particular,

\[
 P(Y=1)=\frac{1-\prod_i(1-2q_i)}2.
\]

Equal-activity and equal-label-balance controls address different confounds;
they are not interchangeable. Eight factors are a first enumerable test, not
evidence for a natural long tail of arbitrary size or dependence structure.

## 2. A complete pressure decomposition without double counting

Define the orthonormal Bernoulli product functions

\[
 \phi_S(x)=\prod_{i\in S}\frac{x_i-q_i}{\sqrt{q_i(1-q_i)}},
 \qquad \phi_\varnothing=1.
\]

Under the product distribution E[phi_S phi_T] = 1{S=T}. On complete support,
every current residual e = Y-f has the exact decomposition

\[
 e=\sum_S r_S\phi_S,\qquad r_S=E[e\phi_S],\qquad
 L=\tfrac12E[e^2]=\tfrac12\sum_S r_S^2.                 \tag{1}
\]

The ordinary target already has interaction terms. Its coefficients are

\[
 c_S=\tfrac12\mathbf1_{S=\varnothing}
 -\tfrac12(-2)^{|S|}
 \prod_{i\in S}\sqrt{q_i(1-q_i)}
 \prod_{k\notin S}(1-2q_k).                            \tag{2}
\]

Proof: expand 1-2Y = product_i[(1-2q_i)-2(X_i-q_i)]. Thus
frequency changes signal geometry as well as exposure. If all q_i=.5, only the
full-order nonconstant target coefficient remains. Low-order pressure is not a
universal prerequisite for successful parity training.

For any fixed optimizer metric, parameter velocity is linear in the residual.
Consequently the velocities induced by all terms in (1) add exactly. In
contrast, sum_i q_i E[velocity | X_i=1] double-counts co-occurring examples.

For a compact factor-level picture we declare

\[
 e_0=r_\varnothing,\qquad
 e_i=\sum_{S\ni i}\frac{r_S\phi_S}{|S|},\qquad
 e=e_0+\sum_i e_i.                                    \tag{3}
\]

Interactions are allocated equally to participating factors. This is a symmetric
attribution convention, not a unique physical ownership of an interaction.
The e_i are generally not orthogonal. The unallocated modal decomposition (1)
remains available to check conclusions that depend on this convention.

## 3. Coupled neurons and the frequency entry point

Retain the mean-field shallow ReLU model and declared optimizer metric:

\[
 f=c+\frac1m\sum_j a_j[w_j^T\bar x]_+,\quad
 M_w=M_a=mI,\quad M_c=1,\quad \bar x=(x,1).
\]

This is minibatch gradient descent with an explicit width-dependent metric;
the study does not silently substitute Euclidean SGD in the same physical
parameters. There is no momentum, weight decay, gate freezing, curriculum,
factor knockout, or neuron intervention.

Writing G_j=1{w_j^T xbar>0}, define

\[
 b_j(e)=E[eG_j\bar X],\qquad
 \dot w_j=a_j b_j(e),\quad
 \dot a_j=w_j^Tb_j(e),\quad\dot c=E[e].                \tag{4}
\]

For an original raw bit coordinate,

\[
 \dot w_{ji}=a_jq_iE[eG_j\mid X_i=1].                 \tag{5}
\]

Frequency is explicit, but (5) is a coordinate-update identity, not a scalar
learning rate for a skill. The same readout coefficient, residual, and gate
couple multiple coordinates and multiple corrections. Substitution of (3) into
(4) gives b_j(e)=sum_g b_j(e_g), including the mean component. Their signed
effects may oppose or reinforce one another.

## 4. Immediate interference and capability transport are different

On a finite population let E=P^(1/2)e, D=P^(1/2)JM^(1/2), K=DD^T, and let U
contain P^(1/2)phi_S. Then U is orthogonal, E=Ur, and

\[
 \dot r=-\mathcal K r,\qquad \mathcal K=U^TKU,\qquad
 -\dot L=r^T\mathcal K r.                             \tag{6}
\]

The modal loss rate is r_S (mathcal K r)_S. It may be negative even when total
loss decreases. Gradient overlap produces immediate transfer or interference.
The cancellation statistic

\[
 \chi=1-\frac{r^T\mathcal K r}{\sum_S r_S^2\mathcal K_{SS}}
\]

is defined only when the denominator is positive. Positive chi means net
cancellation in this decomposition; negative chi means net cooperation.
Factor-allocation cancellation is a different statistic. Neither is by itself
response destruction or a causal explanation of endpoint loss.

For a fixed functional probe d and horizon tau, retain
C=K(K+lambda I)^(-1), lambda=1/(2 tau), beta_d=d^TCd. Inside a gate cell,

\[
 \dot C[v]=\lambda(K+\lambda I)^{-1}\dot K[v]
                    (K+\lambda I)^{-1},\qquad
 \dot\beta_d=\sum_g d^T\dot C[v_g]d.                 \tag{7}
\]

This generalizes the earlier pressure susceptibility to any number of
components. Negative terms are differential adverse pressure effects; positive
terms are creation or strengthening. A probe is not automatically a spectral
option. Current residual loading and finite correction consequences are still
required before inferring harm.

The source code independently checks (1), (4), (6), and (7) against finite
differences and automatic differentiation. Crucially, the experiment also
checks where the smooth approximation (7) fails.

## 5. Finite gate changes cannot be omitted

For one actual SGD step write delta z_j=delta w_j^T xbar and

\[
 R_j=[z_j+\delta z_j]_+-[z_j]_+-G_j\delta z_j.
\]

Exactly, delta f = ell+B+G, where ell=J delta theta and

\[
 B=\frac1m\sum_j\delta a_jG_j\delta z_j,\qquad
 G=\frac1m\sum_j(a_j+\delta a_j)R_j.
\]

Therefore the loss decrease on the unchanged population is exactly

\[
 L_k-L_{k+1}=E[e\ell]+E[eB]+E[eG]-\tfrac12E[(\delta f)^2]. \tag{8}
\]

This is evaluated on actual unfrozen training updates. The gate term can have
either sign. Discrete inputs do not provide an anti-concentration assumption
near learned boundaries; a nominally small learning rate is not enough to make
it negligible. In this study the gate term often cancels a large part of the
tangent-predicted improvement. Smooth second-order forecasts are consequently
retained as failures, not repaired by fitting an effective common-pressure rate.

There is an exact finite kernel ledger as well. For a fixed unit probe d put
u=P^(1/2)d, h_j=sum_x u_x[z_j(x)]_+, and
B_j=sum_x u_x G_j(x)xbar. The neuron contribution is
(h_j^2+a_j^2||B_j||^2)/m. Its endpoint change splits exactly into

\[
 \frac1m\left[
 h_{j,1}^2-h_{j,0}^2
 +\frac{a_{j,1}^2-a_{j,0}^2}{2}(\|B_{j,1}\|^2+\|B_{j,0}\|^2)
 +\frac{a_{j,1}^2+a_{j,0}^2}{2}(\|B_{j,1}\|^2-\|B_{j,0}\|^2)
 \right].                                             \tag{9}
\]

These are activation-feature, amplitude, and derivative-gate terms. Neuron
blocks are implementations, not option counts.

To allocate a finite actual update to pressure sources, the implementation also
integrates smooth transport on the straight parameter path and allocates each
gate jump by the sources' signed margin displacements. Their sum is exactly the
observed kernel change, including jumps. Shares may be negative or exceed one
when opposing pressures nearly cancel. This is a declared path attribution,
not unique causal mediation. Endpoint and path allocations need not agree
component by component when amplitudes move during a crossing.

## 6. Ordinary loss decay separates geometry from residual sorting

Define the effective instantaneous loss-decay rate

\[
 \rho=\frac{E^TKE}{E^TE},\qquad -\frac{d\log L}{dt}=2\rho. \tag{10}
\]

Here t is gradient-flow time; ordinary SGD is indexed by steps, with its exact
finite loss accounting supplied by (8). For s=E/||E|| and smooth flow,

\[
 \dot\rho=\sum_g s^T\dot K[v_g]s
             -2\{s^TK^2s-(s^TKs)^2\}.                \tag{11}
\]

The first term is response transport under all pressures. The second is
nonnegative residual sorting: learning preferentially removes components with
strong response. A declining current-demand rate can therefore occur with a
constant or improving capability operator.

The same distinction holds for b=s^TCs:

\[
 \dot b=s^T\dot Cs
       -2\{s^TCKs-b\,s^TKs\}.                        \tag{12}
\]

The bracket is nonnegative because C is an increasing scalar function of K.
This explains why the response to each model's moving residual must not be
read as a count of destroyed options.

No smoothness assumption is needed for a symmetric endpoint split. Write
rho_ab=s_a^T K_b s_a. Then

\[
 \rho_{11}-\rho_{00}
 =\tfrac12(\rho_{01}-\rho_{00}+\rho_{11}-\rho_{10})
 +\tfrac12(\rho_{10}-\rho_{00}+\rho_{11}-\rho_{01}).    \tag{13}
\]

These are exact geometry and demand-rotation allocations between ordinary
checkpoints. They include all gate changes. They do not uniquely identify
which pressure caused a long-run loss difference.

The existing resolvent bound also survives: for b=s^TCs,
rho >= lambda b/(1-b). Integrating (10) bounds loss progress if b is controlled
along the actual trajectory. Initial b alone is not a future-loss forecast.

## 7. A tested conditional law for the full pressure mixture

In optimizer coordinates, let V(X,Y;theta) be the negative per-example gradient,
mu=E[V], and Sigma=Cov(V). For ordinary IID minibatches of size B,

\[
 E[\delta\theta\mid\theta]=\eta\mu,\qquad
 \operatorname{Cov}(\delta\theta\mid\theta)=\eta^2\Sigma/B. \tag{14}
\]

The moments are exact. The approximation replaces the conditional minibatch
gradient by mu+Sigma^(1/2)Z/sqrt(B), retaining the entire covariance and all
shared-unit correlations. Every forecast step recomputes residuals, gates,
moments, and parameters at the predicted state. Nothing is held frozen across
the forecast window. Physical validation uses independent ordinary minibatches.

On a finite population an equivalent construction uses weights

\[
 w_x=p_x+\sqrt{p_x/B}Z_x-p_x\sum_z\sqrt{p_z/B}Z_z,
\]

whose mean is p and covariance is (diag(p)-pp^T)/B. These can be negative:
they are a Gaussian gradient-noise approximation used only for predictions,
never a modified data distribution for the reported training runs.

This replaces single-direction common pressure by state-dependent joint
pressure moments. A gate margin has a Gaussian one-step approximation with
mean z+eta mu_z and variance eta^2 sigma_z^2/B; conditional closing and opening
probabilities follow from its normal tail. For response or joint closure, the
correlations with readout changes and other margins must also be retained.
Multiplying independent unit-survival probabilities is unjustified.

The central empirical assumption is approximate Gaussianity of the relevant
aggregate gradient projections at the declared batch size, including near gate
boundaries. It is not a theorem for rare pattern counts or small batches. The
forecasts use the enumerated distribution; outside this small task, moments
would have to be estimated and their estimation error included.

## 8. Width and option-event predictions: what follows conditionally

At IID initialization K=bias+(1/m)sum_j Z_j has the earlier 1/m kernel-variance
identity. During training all units share residuals and minibatches. For any
exchangeable scalar implementation contributions A_j,

\[
 \operatorname{Var}\left(\frac1m\sum_jA_j\right)
 =\frac{\operatorname{Var}(A_1)}m
  +\frac{m-1}{m}\operatorname{Cov}(A_1,A_2).           \tag{15}
\]

Thus width may improve initial coverage, creation, survival, or recovery, but
there is no general exponent or monotone retention law without additional
control of the mean and correlations. A strong mean response in an irrelevant
direction is insufficient. The experiments retain a common fixed interaction
probe across widths as well as each width's own current residual.

For a fixed beta probe and full ordinary-SGD transition kernel, define
a_k=E[beta_{k+1}-beta_k | theta_k]. Then

\[
 \beta_T=\beta_0+\sum_{k<T}a_k+M_T,                   \tag{16}
\]

with a martingale M. Conditional forecast paths estimate weakening, recovery,
and first-passage probabilities in actual steps. Expected counts sum marginal
event probabilities without an independence assumption; joint events require
the joint paths. A bounded-increment martingale bound is possible given
prospective drift and variance bounds, but those bounds have not been established
here. Do not turn (16) into an unvalidated closure hazard.

The measured 10% decline is practical response weakening, not structural zero.
The current forecasts use instantaneous kernel response, not an intrinsic option
count. Demanded spectral closure, sustained impairment, and causal loss harm
remain distinct targets.

The old inactive-unit reserve theorem is usually vacuous for this decomposition:
nonzero ANOVA pressures generally span the whole input support. A neuron
inactive on that entire support contributes no useful response there. Approximate
low exposure, signed transport, alternative routes, and finite recovery can
still matter, but an exact protected floor cannot be presumed.
The constant output-bias block remains protected; it supplies no floor for the
nonconstant product-basis corrections orthogonal to it.

## 9. Historical forecast target explored in this branch

This branch evaluated

\[
 \Delta_H L(k)=E[L(\theta_k)-L(\theta_{k+H})\mid\theta_k],
 \qquad H=1000\text{ ordinary SGD steps}.              \tag{17}
\]

It was tested at ordinary checkpoints for each width, with prospective
predictions, held-out conditional outcomes, and uncertainty clustered by
original seed. Per-factor errors, exact finite gate terms, and fixed-demand
response were retained as diagnostics. This does not make optimizer trajectory
simulation the desired macro theory.

The independent-seed experiments validate the conditional numerical baseline
for (17), but this is outside the current scope. It does not establish the
desired macro theory of options, width, and consequential loss. The next primary
measure is instead the aggregate endangered correction mass over learning
phases, as described in the [reserve-relaxation ladder](../../../docs/model/reserve_relaxation.md).
In particular, the standardized bipolar parity formulation documented there
restores the necessary opposing geometry and unlearned initial pressure,
confirming that response weakening and width protection operate naturally across
sequential learning episodes without discrete trajectory simulation.
