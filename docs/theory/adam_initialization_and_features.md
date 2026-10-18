# Adam: output polarity, feature access and the timing of learning

The starting observations come from [robustness](../reports/robustness.md) and
[Adam transfer](../reports/adam_cross_batch.md): zero-output narrow Adam models
often acquire only negative outgoing coefficients; random outgoing initialization
strongly helps; early harmful Adam transfers are already rare; and wider models
can fit much better with more accumulated pair cost.

Those observations suggest a failure of joint feature/readout development,
rather than one universal kind of destructive pair interaction. The distinction
below yields separate, falsifiable explanations.

## What recent alignment theory actually licenses

**Paper result and scope.** Boursier and Flammarion,
[Early Alignment in Two-Layer Networks Training is a Two-Edged Sword](https://www.jmlr.org/papers/v26/24-1523.html)
(JMLR 2025), quantify early directional alignment under small initialization
and gradient flow. Their Theorem 2 constructs a regression example where
alignment obstructs interpolation at arbitrary width. This supports checking
effective feature diversity, rather than assuming many neurons supply many
useful directions. Their initialization scales both layers and satisfies
|a_r(0)|≥||w_r(0)||; our zero-a, order-one incoming initialization violates
that condition. Their result is not an explanation proved for default Adam,
our teacher distribution, or our learning rates. See their §2 and Theorems 1–2.

**Our deduction.** A small initial network output is not the same thing as
the small-parameter regime in that theorem. Nor may we import its sign
preservation into this experiment. With z_r=(w_r,b_r) and augmented input
(x,1), ReLU homogeneity gives, under ordinary equal-rate gradient flow,

\[
z_r^\top\nabla_{z_r}L=a_r\partial_{a_r}L,
\qquad \frac{d}{dt}(a_r^2-\|z_r\|^2)=0.
\]

Our initial invariant is negative, so a_r can cross zero while z_r stays
nonzero. This identity does **not** trap outgoing signs. Momentum, finite
steps and Adam do not generally preserve this invariant. A measured negative
sign trap must therefore be explained dynamically, not asserted from this
gradient-flow conservation law.

## Why the mean target can organize the initial motion

At a_r=0, incoming gradients vanish and the full-data outgoing gradient is

\[
g_{a_r}(0)=-\frac{1}{\sqrt m}\mathbb E[y\phi_r],
\qquad \phi_r=\operatorname{ReLU}(w_r^\top x),
\]
\[
\mathbb E[y\phi_r]
=\mathbb E[y]\mathbb E[\phi_r]+\operatorname{Cov}(y,\phi_r).
\]

The teacher-8/2027 mean is approximately −.662; inputs are Gaussian,
biases initially zero, and E[phi_r]=||w_r||/sqrt(2*pi)>0. A shared negative
mean term can dominate many initial feature correlations. The exact first
Adam outgoing update from empty moments is

\[
\delta a_r=\eta\,
\frac{\mathbb E_B[y\phi_r]}
{|\mathbb E_B[y\phi_r]|+\epsilon\sqrt m}.
\]

Thus, when epsilon is negligible, Adam largely removes differences in their
initial magnitudes. SGD instead weights these correlations by eta/sqrt(m).
Both have zero incoming movement on the very first update. This is a
concrete reason to track the timing of sign changes and hidden-feature
motion separately. It does not imply Adam keeps the initial sign forever.

**New check of existing states.** Every initial full-data target correlation
is negative in the retained seeds 0 and 3 at widths 8/10/16/32. This holds
for the incoming features shared by the core optimizer profiles. It is a
full-data statement, not a reconstruction of every first minibatch. The
successful wide models therefore cannot be explained simply by finding an
initial feature with a positive full-data correlation that the narrow models
lacked. Such correlations also do not determine the signs of a joint
least-squares readout, whose feature Gram matrix matters.

## A close 2026 paper suggests a check, which these cases partly reject

**Paper result and scope.** Sandbrink, Martinelli, van Meegen, Gerstner and
Brea, [Teacher Geometry Shapes Learnability in Teacher-Student Networks](https://arxiv.org/html/2609.09595v1)
(9 September 2026 preprint), distinguish interior failures from neurons
whose gates become always off or always on over finite data. Their §6 finds
benefits from faster readout learning and slower hidden-bias learning;
switching to Adam alone does not resolve their failures. Their analytical
reductions assume small systems, infinite Gaussian data and fast readout.
Their simulations include an output bias and a two-stage convergence
procedure, unlike our minibatch, fixed-budget protocol. Use this as a
mechanism and intervention template, not a transferred success-rate theorem.

**Our check.** For Gaussian inputs, set k_r=−b_r/||w_r||. The population
activation probability is Pr(Z>k_r); on the actual finite sample, inspect
the signs of w_r^T x_i+b_r directly. Always-off, always-on and low-coverage
neurons should be distinguished. A large raw bias is not sufficient, since
the incoming norm also matters.

We evaluated retained states at initialization, 10k, 50k and 100k updates.
We also froze their hidden features and solved the unrestricted least-squares
readout on all 8,192 training points, without adding an output bias. Selected
100k examples, all using ordinary training MSE:

| Initialization / optimizer | Width / seed | Actual MSE | Best fixed-feature readout MSE | Always-off neurons |
| --- | --- | ---: | ---: | ---: |
| zero / Adam .001 | 8 / 0 | .104451 | .104314 | 0 |
| zero / Adam .001 | 8 / 3 | .104649 | .104594 | 0 |
| zero / Adam .001 | 32 / 0 | 2.29e-7 | 1.19e-7 | 7 |
| zero / Adam .001 | 32 / 3 | 2.00e-6 | 7.28e-7 | 0 |
| zero / Adam .01 | 32 / 0 | 5.87e-8 | 3.36e-8 | 23 |
| outgoing std 1 / Adam .001 | 8 / 0 | 1.87e-4 | 1.78e-4 | 0 |

Both narrow Adam-.001 examples have every neuron active on at least 7.4%
or 18.5% of the training data, respectively; none is always on. An optimal
signed readout barely changes their error. This rules out both complete gate
death and a readout-only repair as explanations for **these two saved failures**.
It does not rule out early gate dynamics, missing rare regions, directional
collapse, or different failures in other seeds.

Both widths do learn features. For seed 0, the optimal readout floor falls
from .794 at initialization to .104 at width 8, and from .391 to 1.19e-7
at width 32. The wide success requires substantial change in the hidden
feature span; its initial random features alone cannot fit this target.
The narrow failure also makes substantial representational progress before
stopping at an incomplete representation.

Conversely, one successful width-32 Adam-.01 model has 23 inactive neurons:
only nine remain active on training data. A final dead-neuron fraction alone
would rank this successful case poorly. Track which useful directions were
acquired before units became redundant. Final nominal width is not effective
feature coverage.

The data, provenance and reproduction command are in
[the saved-state check](checks/README.md). These are purposively retained
examples. They must not be pooled into a new estimate of cohort frequencies.
The readout oracle is a local diagnostic, not training or a causal rescue.

## Predictions that distinguish the remaining explanations

**Joint feature/sign development.** Track output-sign crossings together with
feature coverage and the fixed-feature readout floor. If a wide Adam model
escapes by developing missing features, its readout floor should fall during
that development. Merely counting positive coefficients is insufficient,
especially when some positive units are inactive.

**A race between readout and incoming motion.** On paired saved states,
separately change the readout rate and the hidden-bias rate, then check
sign crossings, normalized kink locations, residual feature demand and fit.
If this explains the failures, improving the timing should rescue narrow
models without requiring C_plus to decrease. The final readout-oracle result
predicts that a late readout-only refit will not be enough in the two failures
above; an earlier intervention could still alter later feature learning.

**Mean-driven early allocation.** Use teachers with the same hidden directions
but outgoing coefficients constrained to give zero mean, renormalizing target
variance. Separately test a trainable output bias initialized at zero, which
preserves inclusion of the original teacher function but enlarges the model.
If the shared mean drives early polarity concentration, reducing that demand
should change the initial sign pattern and narrow-Adam failure rate. It need
not eliminate hard feature learning. Simply centering labels while retaining
our no-output-bias architecture can change whether width K represents the
target; that would confound this test.

**A common mechanism across optimizers remains open.** The most defensible
candidate is reduced shortage or misallocation of useful features, measured
using [residual access](residual_access_and_tests.md), rather than universal
reduction of pair overlap. This must be tested on SGD and momentum failures
as well, including successful narrow controls and wider counterexamples.
