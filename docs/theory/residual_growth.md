# Signed residual potentials and the loss of feature roles

18 October 2026. This replaces explicit splitting as the intervention used to
motivate the next experiments. The [training report](../reports/residual_growth.md)
contains ordinary random-initialization results, including unsuccessful
predictions. No hidden neurons are copied, inserted, or assigned to teachers.

## 1. An object governing neurons already present during training

Write a bias-free ReLU network as

\[
f(x)=\frac1{\sqrt m}\sum_r a_r\operatorname{ReLU}(w_r^Tx)
    =\sum_r c_r\phi_{u_r}(x),\qquad
u_r=w_r/\rho_r,\quad \rho_r=\|w_r\|,\quad c_r=a_r\rho_r/\sqrt m.
\]

The signed empirical feature measure μ=Σcᵣδᵤᵣ describes both the feature
directions and the positive/negative roles they currently carry. Define its
**residual potential**

\[
\boxed{H_f(u)=\langle y-f,\phi_u\rangle
       =h_y(u)-\int K(u,v)\,d\mu(v).}
\]

Here h_y(u)=E[yφᵤ] and K(u,v)=E[φᵤφᵥ]. For isotropic Gaussian inputs and
unit directions, K(u,v)=κ(arccos(uᵀv)), with
κ(t)=[sin t+(π−t)cos t]/(2π). This field is not a collection of interaction
counts: it is the first variation of the population loss with respect to
signed feature mass. It changes as other neurons learn.

For ordinary equal-rate gradient flow on half-MSE, away from ρ=0,

\[
\dot a_r=\frac{\rho_r}{\sqrt m}H_f(u_r),\qquad
\dot\rho_r=\frac{a_r}{\sqrt m}H_f(u_r),
\]
\[
\boxed{\dot c_r=\frac{a_r^2+\rho_r^2}{m}H_f(u_r)},\qquad
\dot u_r=\frac{a_r}{\sqrt m\rho_r}\nabla_{\mathbb S}H_f(u_r).
\]

Thus sign(aᵣ)H_f(uᵣ)<0 contracts the magnitude of the neuron's current signed
role; a positive sign expands it. The derivation uses positive homogeneity,
not balanced initialization, an infinitesimal split, or frozen hidden weights.
The radial identity also holds for finite-data full gradients, with empirical
expectations. With trainable hidden biases, use augmented weights and inputs;
the closed isotropic Gaussian kernel no longer applies unchanged. Momentum and
Adam have optimizer history/preconditioning, so these gradient-flow identities
are not exact dynamics for those optimizers.

The loss itself is the interacting-feature functional

\[
L(\mu)=\tfrac12\|y\|^2-\int h_y\,d\mu
       +\tfrac12\iint K(u,v)\,d\mu(u)d\mu(v).
\]

This is closely related to the existing interacting-particle/mean-field view
of two-layer networks; see [Chizat and Bach's presentation of particle gradient
flows](https://lchizat.github.io/PGF.html). We are not claiming novelty for this
functional or a finite-width convergence theorem from its continuum limit.

## 2. A sign condition for suppression, with explicit assumptions

Suppose y=Σβⱼφᵥⱼ. At a near-zero initial prediction, a neuron with the correct
direction vᵢ and correct output sign is driven by

\[
\chi_i=\operatorname{sign}(\beta_i)(G\beta)_i,
\qquad G_{ij}=\langle\phi_{v_i},\phi_{v_j}\rangle.
\]

Its own target component contributes |βᵢ|Gᵢᵢ>0. Opposing-output components
contribute negatively when their features coactivate. If their combined
contribution exceeds the own-feature term, **χᵢ<0: even an already correctly
oriented small neuron is pushed away from its required output polarity**.
That is a concrete competitive effect of the simultaneous target demands.

For the four independent pair planes used in the experiments, each teacher
pair has coefficients (1,−r) and full angle 2θ. Omitting the common positive
target-normalization factor, the weaker positive feature's initial evidence is

\[
\chi_+=\tfrac12-r\kappa(2\theta)+\frac{3(1-r)}{2\pi}.
\]

The last term is the contribution of other pairs' nonzero means, despite their
orthogonal input directions. At θ=30°, its sign changes at

\[
r_{\rm raw}=
\frac{1/2+3/(2\pi)}{\kappa(\pi/3)+3/(2\pi)}\simeq1.25.
\]

If a learned output bias has absorbed the target mean, raw G is replaced by
the centered covariance C=G−μμᵀ. Cross-pair covariances then vanish, giving

\[
\chi_+^{\rm centered}=v-rq,\qquad
v=\tfrac12-\tfrac1{2\pi},\quad
q=\kappa(2\theta)-\tfrac1{2\pi}.
\]

For θ=30°, q>0 and the threshold is about 2.345. At θ=60°, q<0, so opposite
outputs reinforce this centered feature correlation. These statements predict
the local direction of a specified feature's readout learning. They do not
predict that an entire network succeeds exactly on one side of either threshold.

At a nonzero network, the corresponding quantity is sign(βᵢ)H_f(vᵢ).
The peer contribution can either inhibit or release a role. For example,
learning a negative target component can remove its negative contribution
from the residual and allow previously suppressed positive roles to grow.
Accordingly, opposing target demands and direct harmful peer updates are
distinct causal statements. The experiments establish the former local
calculation; they do not establish that the peers' net influence is harmful.

## 3. Why signed role availability can make width consequential

The eight teacher hyperplanes in our nondegenerate geometries are distinct:
four carry positive ReLU coefficients and four carry negative coefficients.
Exact representation on the Gaussian population requires at least four
positive-output and four negative-output nonconstant student neurons.

To see the necessity, the distributional Hessian of a ReLU ridge is supported
on its activation hyperplane, with coefficient proportional to a·wwᵀ. A
positive teacher kink needs a positive student contribution on that same
hyperplane. Negative units cannot supply it; one student hyperplane cannot
cover two distinct teacher hyperplanes. The analogous argument applies to
negative kinks. An output intercept does not change this conclusion. This is
a necessary condition for exact population representation, not a lower bound
on finite-sample or approximate error.

At width 8, every neuron must ultimately occupy the right type of role. At
width 32, many neurons can carry the dominant polarity while four others
suffice for the minority polarity. This supplies a possible width mechanism:
early competitive evidence can remove useful roles, and redundancy makes
that loss more tolerable. We test whether role counts actually change during
training, including a control starting with exactly four signs of each kind.

This is not a proof that width prevents polarity collapse. H_f, its angular
flow, and the surviving directions depend on the full trajectory. Our ordinary
initialization has a²−||w||² generally negative, so gradient flow permits
output-sign crossings. No sign-preservation theorem traps these runs forever.
Likewise, having enough neurons of each polarity is not sufficient: their
directions and amplitudes still need to represent the target.

## 4. A plausible conditional operator that failed empirically

Before the polarity follow-up, we derived a different candidate. If a shared
span C has already been optimally fitted, the residual signals at target
features are Sβ, where

\[
S=G_{TT}-G_{TC}G_{CC}^{-1}G_{CT}.
\]

This conditional Gram operator is positive semidefinite but can have negative
off-diagonals. For two flanks around one center,
S₁₁=1/2−2κ(θ)² and S₁₂=κ(2θ)−2κ(θ)². With outputs (1,r), the weaker
signal becomes negative at r>S₁₁/(−S₁₂), when S₁₂<0. It is an exact
conditional suppression calculation and needs no neuron splitting.

However, the tested negative-sign case at 30° with outputs (1,3) was easy
for narrow networks, whereas the positive-sign case (1,−3) was difficult.
The hypothetical fitted shared span was not a justified description of every
ordinary trajectory. We reject Sβ as an established predictor of the width
benefit in this family. The original [protocol](residual_growth_protocol.md)
and [polarity follow-up protocol](polarity_followup_protocol.md) preserve the
chronology rather than rewriting the prediction after seeing results.

## 5. What has and has not become a stronger explanation

The [new experiments](../reports/residual_growth.md) find much larger ordinary
training gaps in the opposing-output family, accompanied by loss of minority
polarity roles. The gaps survive 100k updates in the five extended starts.
That is stronger evidence than the previous small-detail splitting experiment.

The raw-evidence threshold also fails as a universal boundary: width helps
even at opposing strengths 1 and 1.2, where χ₊ is positive. Output-bias
controls have mixed outcomes, and balanced initial signs do not prevent later
role loss. The state-dependent signed measure and residual field remain
useful objects; neither initial χ nor a count of negative potentials alone
is an adequate theory of the final width benefit.

Initialization is also part of the regime: increasing the initial readout
scale reduces the observed opposing-output gap and makes the same-sign
control harder. A data-only sign condition cannot account for that sensitivity.

This is best described as a **candidate competition between signed feature
roles**, with a precise local growth law and a structural capacity consequence.
It is not yet a demonstrated explanation through direct harmful
neuron-to-neuron updates. The missing link is a prediction of the joint
evolution of H_f and signed feature occupancy, validated on ordinary trajectories.

Related literature helps locate, but does not prove, this mechanism:
[early alignment](https://www.jmlr.org/papers/v26/24-1523.html) can reduce
feature diversity and obstruct fitting under particular small-initialization
assumptions; [gradient starvation](https://papers.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)
formalizes feature imbalance under cross-entropy. Our finite-data squared-loss
models and initialization do not inherit those papers' guarantees.

## Teacher-free extension (proposed)

The residual potential needs data labels and current predictions, not a known
teacher decomposition. For a vector-output model, define

\[
\mathbf H_f(u)=\mathbb E[\phi_u(x)r(x)],\qquad
r=y-f(x)\ \text{(squared loss)},\quad
r=y-\operatorname{softmax}(f(x))\ \text{(cross-entropy)}.
\]

For an existing hidden activation h_r and outgoing vector a_r in the MNIST
normalization, ordinary gradient flow gives

\[
\dot a_r=\mathbb E[h_r(x)r(x)]/\sqrt m,\qquad
\tfrac12\frac{d}{dt}\|a_r\|^2
=a_r^T\mathbb E[h_r(x)r(x)]/\sqrt m.
\]

For cross-entropy, remove the outgoing vector's common-logit component when
interpreting a class-contrast role. Negative radial force contracts that role;
this need not be harmful. Momentum and Adam require tracking the actual update
separately from the gradient force. Empirical averages estimate the field; a
held-out probe can check whether conclusions are specific to training examples.

Near uniform predictions and balanced labels, component k is proportional to
the feature's class-k mean activation minus its average activation across
classes. Thus a sign condition is observable without teacher features. What is
missing is a guarantee that the suppressed role is required: the four-positive /
four-negative teacher quota has no direct MNIST analogue. One prospective test
would define candidate features/class contrasts on a separate discovery split,
check their complementary predictive value, and test whether early suppression
precedes persistent loss of that information and is relieved by width.

This is a proposed transfer route, not a completed experiment or an established
causal explanation of the existing MNIST width gap.
