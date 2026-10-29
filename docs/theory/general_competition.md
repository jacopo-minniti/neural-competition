# Competition as suppression of useful learning opportunities

23 October 2026. Exploratory synthesis, not an established universal mechanism.

The Gaussian teacher is a test fixture, not the intended scope. Existing evidence
rejects defining competition as gradient disagreement, positive pair cost, or
shrinkage alone. Those quantities can increase during successful learning.
A useful formalization must distinguish local interaction, loss of useful
representation, and causal consequences over time.

## 1. A distribution-free interaction operator

For any input law P, differentiable prediction loss L=E ell(f(x),y), and additive
feature model f=sum a_r h_r, the readout Hessian blocks are

    G_rs = E[h_r h_s Hess_f ell(f,y)].

For squared loss this is the feature Gram matrix. For softmax cross-entropy it
is weighted by diag(p)-pp^T. Off-diagonal blocks describe coupling, but coupling
is not inherently harmful. In a deep model replace features by parameter-block
Jacobians: J_r^T H_loss J_s. The full parameter Hessian additionally includes
residual-weighted second derivatives of f. The block choice and parameter metric
must be stated; arbitrary rotations or rescaling change parameter-based measures.

For any fixed candidate function h, the opportunity to add it is
D L(f)[h]=E[grad_f ell(f,y) dot h]. This is defined without a teacher, Gaussian
inputs, or a prescribed decomposition of the target. Use held-out data to test
usefulness; selecting h from evaluation labels would introduce selection bias.

A useful feature may have negative directional derivative yet be inaccessible
through the current parameterization. If J is the prediction Jacobian, a desired
prediction change v is locally feasible only in range(J). The projection of
-grad_f L onto that range describes accessible first-order descent under a
prediction-space norm; it does not describe ordinary SGD's parameter metric.
Saturated/dead units, rank loss and collapsed feature directions can remove
opportunities. Width can provide additional directions, but no monotonicity
claim follows for independently trained networks or fixed compute.

## 2. Operational meaning of harmful competition

Call a mechanism harmful competition only after specifying:

1. A useful learning opportunity, defined before observing the endpoint.
2. How learning other features changes its accessibility or progress.
3. A paired intervention on that interaction which improves a common held-out
   objective at a specified horizon, beyond movement/rate/history controls.

This distinguishes immediate update obstruction from delayed suppression or
conversion of representation. A baseline-versus-intervention risk contrast is
a causal effect of that intervention, not a unique decomposition of the width
advantage. The previous temporal experiments often found cooperation. The sign
experiment found adverse target demand even without peers. Both must remain
counterexamples to claiming that every suppressed opportunity is peer-caused.

The existing profile loss R(W)=min_A L(Phi(W)A) is a useful distribution-free
representation object. It is defined for cross-entropy too (use regularization
or acknowledge a possibly unattained infimum on separable data). For squared
loss, L=R+readout lag exactly. For other losses that Pythagorean identity does
not hold. On regular smooth regions, the envelope derivative still separates
representation movement from coefficient adaptation. Static R alone previously
failed to predict endpoints; the question is how particular opportunities become
inaccessible during ordinary training.

## 3. Signed spectral obstruction beyond Gaussian inputs

Assume x has a positive twice-differentiable density p on R^d, suitable boundary
decay for two integrations by parts, and the square-integrability conditions
below. Define S(x)=Hess p(x)/p(x), T_g=E[g(x)S(x)]. Integration by parts gives
T_f=E[Hess f] in the distributional sense, including ReLU kinks.
For a unit a ReLU(w^T x+b),

    T_unit = a w w^T E[delta(w^T x+b)].

The multiplier after a is nonnegative. Consequently a scalar-output shallow
ReLU network with k+ positive and k- negative active units has at most k+
positive and k- negative eigenvalues in T_f, for arbitrary such p. The target
g(x)=E[y|x] need not be a neural teacher. T_g=E[y S(x)].

Let s=svec(S), where off-diagonal entries are multiplied by sqrt(2), so vector
norm equals symmetric-matrix Frobenius norm. Set C=E[s s^T], assumed finite,
and Delta=svec(T_g-T_f). Hilbert-space projection gives

    E[(g-f)^2] >= Delta^T C^dagger Delta
                 >= ||T_g-T_f||_F^2 / lambda_max(C).

For singular C, Delta lies in its range: a null direction has zero inner product
with s almost surely. The first inequality is obtained by projecting g-f onto
the linear span of the coordinates of s. The second follows spectrally on that
range, provided lambda_max(C)>0. Thus

    E[(g-f)^2] >= [sum_{i>k+}(lambda_i^+(T_g))^2
                   +sum_{i>k-}(lambda_i^-(T_g))^2] / lambda_max(C).

The spectral-tail step follows from symmetric-matrix eigenvalue approximation
under inertia constraints. For standard Gaussian inputs C=2I on symmetric
matrices, recovering the previous factor 1/2. Subtract irreducible conditional
noise to interpret this as excess squared risk when labels are noisy.

This is a general-density theorem, not a claim about all distributions. Uniform
inputs require boundary terms or a different test operator; discrete datasets
have no such density. Estimating S in high dimensions may be difficult and the
lambda_max bound may be loose. Empirical plug-in eigenvalues without uncertainty
are not rigorous population certificates. The training effect can transfer even
where this particular certificate is unavailable.

The score-function identity and spectral feature extraction are established:
[Janzamin et al., FEAST (2015)](https://proceedings.mlr.press/v44/janzamin2015.html).
The relevant extension here is combining that identity with signed inertia and
the input-dependent moment metric. No literature novelty claim is made.
[Gradient starvation](https://arxiv.org/abs/2011.09468) supplies another formal
example of competing feature learning under cross-entropy; its assumptions and
conclusions should not be transplanted to these squared-loss runs.

## 4. Test and next decisions

The companion exploratory probe uses Gaussian, product Gaussian-mixture, and
uniform inputs with covariance I. It uses ten paired seeds 300–309 per optimizer,
ordinary balanced widths 8 and 32 and permanently protected width 8, 10k steps.
All settings retain a realizable eight-feature target to isolate input-law changes.
This tests transfer of the phenomenon, not the general-density theorem, and it
does not test teacher-free learning. Results and exact executed source are saved.

The next natural-data test should define complementary candidate functions on a
separate discovery split, measure their incremental held-out value after fitting
the current representation, then intervene on early learning dynamics at fixed
width. Use prediction-space movement controls and preserve optimizer history.
A cross-entropy/classification study should use output contrasts and Jacobian
subspaces rather than scalar output-sign quotas. Neither Gaussian density
estimation nor a known teacher is needed for that operational definition.
