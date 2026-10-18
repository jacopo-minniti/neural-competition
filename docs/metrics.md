# Metrics, signs and interpretation

There is no single scalar definition of competition privileged across all
experiments. We use different measurements for conflicting data demands,
interactions between neuron updates and accumulated training costs. Their signs
and denominators answer different questions.

The [allocation study](reports/allocation.md) now measures three additional
objects proposed in the theory notes: duplication-invariant positive/negative
residual work, corrective cosine, and gain from relative neuron step weights
beyond uniform shrink. It distinguishes the nonnegative fitted quadratic
regret from signed held-out exact-forward benefit, and includes sign-filter,
four-group, shuffled-coefficient and shortened-step controls. None gives a
universal width ordering in the completed experiments. Detailed definitions,
CE extensions, small-step limits and caveats are in that report.

## Model and loss conventions

\[
f_\theta(x)=\sum_{r=1}^m u_r(x;\theta_r),\qquad
u_r=\frac{a_r\operatorname{ReLU}(w_r^\top x+b_r)}{\sqrt m},\quad
\theta_r=(w_r,b_r,a_r).
\]

At input dimension 20, each neuron block has 22 coordinates. Loss is half-MSE,
L_B=E_B[(f−y)²]/2. Reported fitting MSE is twice this loss. Expectations in
diagnostics are empirical averages over the specified source batch, reference
set, test set or diagnostic batch draws.

## Raw cross-batch alignment

At frozen parameters, g_B=grad L_B and

\[
X(B,C)=g_C^\top g_B,\qquad
L_C(\theta-\eta g_B)-L_C(\theta)=-\eta X+o(\eta).
\]

X<0 predicts harm to C from a sufficiently small SGD step on B. The threshold
is **zero, not the mean of X**. Probability p_minus=Pr(X<0) measures frequency;
E[(-X)_+] measures severity and E[(X)_+] beneficial transfer. Signed E[X] combines
the two. Cosines remove the pair's gradient norms but answer a different
normalization question.

For independent batch draws from the same fixed empirical objective,
E[X]=||g_train||². The mean is therefore sensitive to training phase and
gradient magnitude, not only compatibility. Net agreement

\[
A=\frac{\|g_{\rm train}\|^2}{\mathbb E_B\|g_B\|^2}
\]

compares full-gradient signal with total batch-gradient energy. Its population
range is [0,1] when defined; the denominator is estimated from probes, and
finite-sample off-diagonal mean dots need not equal the exact full-gradient
numerator. A can be smaller for the better-fitted wider model. At vanishing
gradients, ratios and signs become unstable or undefined.

For a nondegenerate X, skewness is E[(X−mu)³]/sigma³ and excess kurtosis is
E[(X−mu)⁴]/sigma⁴−3. Skewness measures asymmetry around the mean. Kurtosis
measures standardized tail/outlier weight around the mean, with a normal
distribution having excess zero. Neither distinguishes the negative tail
below zero from unusually large beneficial values. Lower kurtosis does not
by itself mean less interference or less positive reinforcement. Moment
estimates can be sensitive to the number of probe batches.

## Optimizer-aware cross-batch transfer

Write a candidate optimizer update as delta_B=−eta h_B. Then

\[
T(B,C)=g_C^\top h_B,\qquad \Delta L_C\simeq-\eta T(B,C).
\]

Positive T predicts help and negative T harm. For the recorded update delta,
the equivalent harmful sign is g_C dot delta > 0. Actual finite loss change
must be evaluated separately; curvature and ReLU gate crossings can reverse
the first-order prediction at a full step.

- SGD: h_B=g_B.
- Classical momentum: h_B=beta v_previous+g_B, with the saved pre-step buffer.
- Adam: update both pre-step moments with g_B, then apply bias corrections:

\[
m'=\beta_1m+(1-\beta_1)g_B,\quad
v'=\beta_2v+(1-\beta_2)g_B^2,\quad
h_B=\frac{m'/(1-\beta_1^{t+1})}
{\sqrt{v'/(1-\beta_2^{t+1})}+\epsilon}.
\]

Here t is the number of completed updates before the candidate. Adam defaults
are beta1=.9, beta2=.999, epsilon=1e-8; the experiments have no weight decay or
AMSGrad. Candidate batches share the same frozen previous moments; they do not
successively change each other's history. Adam's first moment alone is not
its actual adaptive direction. In general T(B,C)≠T(C,B), so we use all ordered
off-diagonal batch pairs, rather than one triangle. E[T] need not be a squared
gradient norm.

The latest cross-batch Adam diagnostic computes the mathematical candidate
direction in float64 at saved float32 states. The every-update loss accounting
instead uses actual rounded float32 before/after parameters, evaluated in
float64. Their numerical conventions and local/finite scopes are explicit.

## Attributing data conflict to neurons

For raw gradients q_r=g_{C,r} dot g_{B,r}; for optimizer transfer
t_r=g_{C,r} dot h_{B,r}. Their sums recover X or T exactly. Merely summing
neurons adds no information. Taking positive/negative parts before summing does:

\[
P=\mathbb E\sum_r[q_r]_+/S_g,\quad
N=\mathbb E\sum_r[-q_r]_+/S_g,\quad S_g=\mathbb E\|g_B\|^2.
\]

Optimizer P_step/N_step replace q by t and normalize by
sqrt(E||g_C||² E||h_B||²). Their difference is normalized mean transfer.
Global negative mass instead applies the negative part after summing neurons.
For example, (-2,3) and (0,1) both give global dot 1, but have negative neuron
mass 2 and 0. Separate P/N expose hidden conflicting data demands. They do
not measure an inner product between different neurons' parameter vectors or
direct interference between r and s on one batch.

Normalizations remove one overall scale; they do not equalize learning dynamics
or make the values invariant to ReLU parameter rescaling. Raw and normalized
mean scores can weight different training seeds differently when averaged.

## Exact prediction changes and cancellation

Hold all proposed block updates fixed. Define each exact finite change

\[
d_r(x)=u_r(x;\theta_r+\delta_r)-u_r(x;\theta_r),\qquad d=\sum_r d_r.
\]

It is also the prediction change from updating only neuron r, retaining the
rest of the existing network. No gradient is recomputed after another neuron
is moved. These changes include simultaneous incoming/outgoing motion and
ReLU gate crossings.

Partial cancellation is C=1−E|sum d_r|/E sum|d_r|. It increases when opposing
movements partially offset. The energy ratio is

\[
R=\frac{\mathbb E d^2}{\mathbb E\sum_r d_r^2}.
\]

R<1 means net cancellation of squared movement; R>1 means reinforcement on
balance. It is compatible with C>0 because some terms can oppose while net
cross terms remain positive. Neither ratio establishes whether movement helps
the residual. R_perpendicular first projects each movement vector orthogonally
to the evaluation-set residual. Ratios are undefined if denominators vanish.

For residual e=f−y, pointwise isolated loss change is h_r=e d_r+d_r²/2;
joint change is h=e d+d²/2. H_pred=E([h]_+) and G_pred=E([-h]_+) apply signs
pointwise before averaging. Dividing by mean source-batch gain U_source
measures collateral effects per unit useful source correction. This H_pred/G_pred
notation is distinct from the per-neuron batch-averaged H/G below. Accurate
examples use a common mask at both widths (usually |e|≤.05 at both models).

## Same-batch pair interference and severity

For evaluation set V, write

\[
A_r=\mathbb E_V[e d_r+\tfrac12d_r^2],\qquad
J_{rs}=\mathbb E_V[d_rd_s].
\]

The same quantity was called I_rs in the initial pair study. It is exactly
L_rs−L_r−L_s+L_0; the two-neuron loss change is A_r+A_s+J_rs. Adding s after r
has marginal loss change A_s+J_rs. J>0 reduces benefit relative to the sum
of separate updates. This includes harmless diminishing returns.

A **strict event** requires A_r<0, A_s<0 and A_s+J_rs>0. Frequency conditions
on both neurons helping alone and counts ordered r→s cases. Aggregate severity
divides summed positive marginal harm over events by summed standalone benefit
of the second neuron over all eligible ordered cases. Event severity restricts
that denominator to events. Neither is an additive cost of the full update:
alternative pairs overlap in neurons. Lower per-pair frequency can coexist
with the same number of events per batch because width creates more pairs.

The [matched-loss follow-up](reports/strict_matched_loss.md) applies these
strict frequency/severity definitions to the original teacher-8 Adam cohorts,
retaining pair counts and raw harm alongside the normalized outcomes.

The broader cost uses every unordered pair once:

\[
C_+=\sum_{r<s}[J_{rs}]_+,\quad C_-=\sum_{r<s}[-J_{rs}]_+,\quad
G=\sum_r[-A_r]_+,\quad H=\sum_r[A_r]_+.
\]

S_plus is sum C_plus / sum G over diagnostic batches or actual updates within
a seed. S_credit and S_net use C_minus and C_plus−C_minus with the same
denominator. These are **ratios of sums**, not averages of unstable batch
ratios. A value .144 means 14.4 units of unfavorable interaction per 100 units
of standalone helpful gains. Signs before averaging are at the batch-averaged
A/J level here, unlike the pointwise collateral-harm calculation above.

Frequency may be measured over all pairs, or only pairs with A_r,A_s<0; the
denominator must be named. Sign tests typically use tolerance 1e-10 times the
evaluation set's current half-MSE. Continuous cost magnitudes are unthresholded.
Frequency, normalized severity, net credit and absolute cost are complementary.

## Exact whole-update and accumulated accounting

For this additive network and squared loss, including all gate crossings,

\[
\Delta L_V=H-G+C_+-C_-.
\]

There are no higher-order cross-neuron terms missing from this exact identity;
each d_r already includes its finite within-neuron nonlinear change. Across
actual updates on a fixed reference set V, the identity telescopes to final
minus initial loss. Source minibatches change, so their sum does not telescope
to one fixed objective.

Raw accumulated penalty K_plus(T)=sum_t C_plus,t answers how much local cost
was experienced over the budget. S_plus_cum=K_plus/sum_t G_t expresses that
cost per unit standalone helpful activity. Costs can exceed initial loss
because training repeatedly improves and harms examples. They are not remaining
error or the amount final loss would fall after an intervention.

Equal accumulated U=sum_t G compares costs at common standalone activity;
U is not net progress. Matched-loss bins compare phase, with ratios of summed
costs/gains within each bin. The main 200k summary averages 12 log-spaced bin
ratios equally over reference MSE .2→.02, requiring complete paired coverage.
Raw totals, equal U and matched loss answer different questions; no one is
uniformly the fairer choice. Failed narrow Adam runs create missing coverage.

## MNIST extension

The [MNIST study](reports/mnist.md) keeps one hidden layer but uses ten output
scores. Each neuron owns (w_r,b_r,a_r), with a_r a ten-vector. Squared loss is
half the sum of ten squared score errors, averaged over examples. Replace
scalar e*d_r and d_r*d_s by vector inner products; all additive-model quadratic
identities, including the exact pair budget, survive finite updates unchanged.

For cross-entropy, define A_r by the exact isolated loss contrast and J_rs by
the exact four-loss contrast. Strict events have the same interpretation, but
pair sums generally omit higher-order effects. Define C_all=Delta L−sum_r A_r
for an exact whole-update budget H−G+C_all. It includes every interaction order
and is not pair-only C_net. Main MNIST G/H/C_all sums measure every actual
update on a fixed reference; pair statistics sample time and, for CE, pairs.
The report retains inverse-inclusion weights, excluded matched targets,
sampling checks and direct evaluations on the entire training set.

## Temporal causal peer response

The [temporal intervention report](reports/temporal_branches.md) introduces a
different measurement: damp one earlier sender update, continue training on
identical future minibatches, and compare later standalone harm at all other
neurons with the ordinary trajectory. Positive B_H is damped-minus-ordinary
recipient harm, so the ordinary sender prevented harm; negative B_H means
damping that sender reduced later harm. It is not the same-step C_plus or a
lag correlation. Useful gains and actual loss are evaluated alongside H.

The primary pulses match removed prediction RMS across widths and include a
uniform-step control of matching RMS. Report raw finite effects, per ordinary
recipient G, and relative to the pulse's initial absolute loss perturbation.
The pulse-normalized ratio is not a percentage of training loss explained.
Only sampled short windows are summed; effects from different hypothetical
branches cannot be added into one full-training counterfactual trajectory.

## Curvature and finite-step controls

Batch sharpness uses E[g_B^T H_B g_B]/E||g_B||², computed by separately saved
HVP numerators and denominators. The oldest `batch_sharpness` field averaged
individual quotients; it is a different legacy statistic. Neither is the
full-data top Hessian eigenvalue. The SGD reference 2/lr is not applied to Adam.
Local quadratic corrections omit gate-crossing effects; exact finite forwards
and shortened steps are used to check the approximation.

## Neuron splitting as motivation, not a measured mechanism

A width expansion can duplicate neurons without changing the initial function,
adjusting outgoing coefficients for the number of copies and 1/sqrt(width).
For identical copies, opposite parameter perturbations cancel the linear term:

\[
\phi(\vartheta+\epsilon v)+\phi(\vartheta-\epsilon v)-2\phi(\vartheta)
=\epsilon^2v^\top\nabla^2\phi(\vartheta)v+o(\epsilon^2).
\]

The surviving change can be useful if aligned with the needed correction. For
a ReLU neuron inside a fixed activation region, mixed incoming/outgoing motion
can supply this second-order prediction term; moving only incoming weights with
fixed outgoing coefficient has no such term there. Gate changes add nonsmooth
effects. This motivates extra degrees of freedom, but no controlled splitting
experiment was run and useful cancellation is not guaranteed.

## Literature-guided alternatives under consideration

[Pair cost and competition](theory/pair_cost_and_competition.md) derives a
duplication counterexample for C_plus, a residual-based finite-step budget and
a proposed local test of whether relative neuron step weights improve on the
best uniform shrink. [Residual access](theory/residual_access_and_tests.md)
develops complementary feature/readout and optimizer-rate diagnostics. These
are research proposals; the definitions and measured results above remain
the conventions for the completed studies. The
[width note](theory/width_escape_and_scaling.md#a-useful-caution-about-splitting-in-our-exact-model)
also identifies a stationary fixed-gate case where the second-order split
motivation has no negative direction.
