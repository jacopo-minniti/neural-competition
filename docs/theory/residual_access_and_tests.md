# Measures of residual access, and experiments that could distinguish mechanisms

The proposed common idea is **limited access to useful corrections**, followed
by optimizer-dependent use of those corrections. This is broader than positive
pair cost, but also more demanding: feature access, direction choice and finite
step size need separate measurements. The proposals below are derived for our
model. They have not been shown to order all optimizers consistently.

Throughout, ||v||²=mean_i v_i² on a stated evaluation set, e=f−y, and
L=||e||²/2. Choose intervention parameters on a source set and assess transfer
on a separate reference or test set; an oracle chosen on the evaluation set
is only a diagnostic upper bound.

## 1. Is the problem the readout, or the features it has available?

For Phi_ir=ReLU(w_r^T x_i+b_r)/sqrt(m), define

\[
L_{\rm readout}=\tfrac12\min_a\|\Phi a-y\|^2,
\qquad L-L_{\rm readout}\ge0.
\]

The first quantity is the error floor of the current hidden representation;
the second is a readout optimization gap. This cleanly separates two
interpretations of Adam's negative output coefficients. The
[saved-state check](adam_initialization_and_features.md) already finds a high
readout floor, and a small gap, in two narrow Adam failures. It does not yet
establish the corresponding pattern across the original cohort or SGD.

An additional, teacher-specific oracle can identify an unserved correction.
Normalize each nonzero teacher feature to empirical norm one, writing
y=sum_j c*_j psi_j. With the current function held fixed, adding one such
feature with a freely signed coefficient alpha gives

\[
\min_\alpha\left[\tfrac12\|e+\alpha\psi_j\|^2-L\right]
=-\tfrac12\langle e,\psi_j\rangle^2.
\]

Thus D_teacher=(1/2)max_j <e,psi_j>² is the best one-feature gain available
from this oracle. D_teacher/L is in [0,1] when L>0. It is not a practical
feature-discovery algorithm or a general definition of competition; it asks
whether a known missing target direction could still help.

There is a simple certificate. At readout stationarity, <e,f>=0, hence

\[
\|e\|^2=-\langle e,y\rangle
\le\left(\sum_j|c^*_j|\right)\max_j|\langle e,\psi_j\rangle|.
\]

A nonzero residual must therefore correlate with some teacher feature, even
if all existing readout gradients vanish. Away from stationarity replace
the left side in this bound by ||e||²−|<e,f>|, which can be nonpositive and
then uninformative. This is an elementary deduction from realizability and
linear readout structure, not an imported convergence theorem.

The oracle can be implemented without changing the old function: rescale old
outgoing weights for the new width and add a zero-output unit with the chosen
teacher feature before optimizing its coefficient. Compare against random
new features and a same-width perturbation. A large oracle gain with failed
ordinary training would locate a feature-discovery obstruction. A low
normalized gain may instead mean the residual is distributed across many
features; inspect the vector of correlations, not just its maximum.

## 2. Is the residual locally accessible, and how fast can the optimizer move it?

**Paper basis.** Liu, Zhu and Belkin,
[Loss landscapes and optimization in over-parameterized non-linear systems and neural networks](https://arxiv.org/abs/2003.00307)
(2020 preprint; Applied and Computational Harmonic Analysis, 2022), use
PL* conditions and tangent-kernel conditioning to obtain optimization
guarantees without convexity. Their §4 relates a positive lower kernel
eigenvalue to a gradient domination condition; their convergence results
require this control over a suitable region, not merely at one snapshot.
We borrow that diagnostic connection, not its sufficient-width guarantees.

**Our derivation and an essential limitation.** Let F be the n-by-p prediction
Jacobian, g=F^T e/n, and D a positive definite preconditioner frozen at the
current state. For delta=−eta Dg define

\[
K_D=FDF^\top/n,\qquad
\mu_e(D)=\frac{e^\top K_De}{e^\top e}
=\frac{g^\top Dg}{2L}.
\]

Under the corresponding full-gradient flow,
−d log(L)/dt=2 mu_e(D). This residual-weighted rate can be informative even
when the minimum eigenvalue of K_D is zero. It does not certify that the
same rate will persist after features or moments change.

In our main architecture p=22m. Even m=128 gives only 2,816 parameters for
8,192 training outputs. The full-training kernel is necessarily singular
at **every** state. Initially, a=0 additionally leaves only the m readout
Jacobian columns nonzero. It would be meaningless to diagnose these runs
by seeking a positive full-training lambda_min(K). A smaller reference set
can conceal that rank obstruction and must not be presented as a certificate
for all training examples.

To separate available directions from their speed, also use

\[
\chi=\frac{\|\Pi_{\operatorname{range}(F)}e\|^2}{\|e\|^2}\in[0,1].
\]

Chi is the fraction of residual energy in the accessible tangent span; an
unrestricted least-squares step can remove it in the linearized problem.
It is unchanged by an invertible change of parameter coordinates. Report singular values,
the projection tolerance and residual weights on weak directions: a formal
span can contain the residual while requiring an enormous parameter change.
Use a metric-based trust region or ridge version when judging realizable
finite steps, and verify the exact forward response.

An exact duplicate split can leave range(F) unchanged while creating
second-order possibilities; chi cannot detect those possibilities. Compare
it with the gate-crossing and split diagnostics in
[the width note](width_escape_and_scaling.md). Neither chi nor mu is itself
a direct neuron–neuron interaction measure.

**Prediction.** If wide models better cover the currently missing residual,
they should have a lower readout floor, more tangent access, or faster
residual-weighted response at the relevant stage. Which quantity improves
depends on whether the mechanism is feature supply or conditioning. The
global gradient agreement A can decrease simultaneously: its denominator
is stochastic gradient energy, whereas mu normalizes by loss. Matching MSE
does not match residual direction, so inspect its spectral weights too.

## 3. Include optimizer history without pretending Adam is one fixed kernel

For the actual candidate direction h_B, use the conditional first-order rate

\[
\Gamma(B)=\frac{g_{\rm train}^\top h_B}{2L_{\rm train}}.
\]

It can be negative. Average over independent candidate batches with the
saved history frozen, and retain its gain/harm distribution. For SGD its
conditional mean is mu_e(I); for momentum/Adam it also contains history.
This is a rate interpretation of existing optimizer-transfer measurements,
not a claim that they already prove a common width trend.

More explicitly, Adam's next direction can be written

\[
h_B=D'_B\left(
\frac{\beta_1m_t}{1-\beta_1^{t+1}}
+\frac{(1-\beta_1)g_B}{1-\beta_1^{t+1}}\right),
\]

where D'_B is the diagonal inverse denominator after updating the second
moment with candidate B. If that denominator is frozen independently of B,
the gradient contribution has a positive semidefinite kernel F D F^T/n;
the saved first moment adds a history-dependent vector. Actual D'_B depends
on B, so one fixed symmetric kernel cannot in general describe all candidate
Adam transfers. The ordered asymmetry measured in our report is expected.

At the same parameters and saved history, compare g_B, the updated first
moment, that moment with a common frozen denominator, and the full next
Adam direction. Run function-step-matched and source-gain-matched comparisons
separately. Evaluate Gamma, held-out harm, corrective cosine and the
[benefit of reweighting neuron steps](pair_cost_and_competition.md).
This can identify filtering versus preconditioning versus candidate-dependent
adaptation. It does not assert that a history reset during training preserves
the subsequent trajectory.

## What to predict before committing to a larger experiment

The following outcomes would discriminate mechanisms at a failed saved state:

| Observation | Explanation it would support | Intervention that should help if it is causal |
| --- | --- | --- |
| Large L_readout; small readout gap; weak tangent access; large teacher-feature demand | Missing useful representation directions | Fresh features or gate-changing expansion |
| Useful tangent access but weak/negative Gamma; optimizer direction differs from full gradient | Poor use of available directions or stale history | Controlled direction/history intervention |
| Large gain from relative neuron step weights beyond the best uniform shrink | Avoidable allocation cost within the proposed update | Reweight/block updates while retaining existing neuron functions |
| Large finite-step rho, but little extra gain from relative weights | Mostly global step overshoot | Uniform learning-rate reduction |

These are hypotheses; mixed outcomes are possible. In particular, low early
Adam harm frequency is compatible with later feature failure, and a lower
pair penalty does not establish better access to missing target directions.

The [temporal peer-update experiment](../reports/temporal_branches.md) now
performs short causal continuations in all original settings. It changes one
earlier sender movement and measures later recipient harm, with prediction-
matched pulses and uniform controls. This tests whether H is downstream of
peer movement; it does not optimize the allocation measure above or identify
the fraction of width benefit mediated by H.

## Priority for the next work

The [objective-scale audit](../reports/effect_magnitude.md) now separates
statistical patterns from practical size. Broad SGD/momentum costs are large
enough to matter locally; low-rate Adam strict reversals are tiny, and early
cumulative pair scores have little stable extra forecasting value. A next
metric should be evaluated in loss units, prospectively, against an early-loss
baseline. The audit's P−N−Q residual-work budget and bounds provide one
candidate; the proposed allocation test should retain best-uniform-shrink
controls before interpreting recoverable gain as neuron competition.

The [matched-loss strict-event check](../reports/strict_matched_loss.md) is
now complete for the original teacher-8 cohorts. Default Adam's per-pair
frequency and aggregate severity fall with width in the shared loss range,
but total event counts rise, and raw reference harm rises in the std-1
initialization control. This supports keeping the access/allocation tests
below separate from a search for one universally decreasing pair score.

1. **Reanalyze saved states first.** On narrow failures, narrow successes and
   wider counterexamples, compare readout floors, teacher residual demand,
   tangent access and actual-direction rates. Use matched early loss and
   explicitly named failure stages. The retained subset can motivate mechanisms;
   cohort claims require the corresponding states for all intended seeds.
2. **Test the proposed allocation measure locally.** Compare relative block
   weights with best uniform shrink, shortened directions, exact forwards and
   independent evaluation batches. Keep fixed-count grouping controls. A
   persistent wide advantage across optimizers would justify investigating a
   shared competition mechanism; failure to agree would be informative too.
3. **Intervene on the feature/sign timing.** Vary readout and bias rates
   separately, early enough to affect feature development, with the saved
   histories and future minibatches controlled. Measure whether feature floors
   improve before fit, and whether C_plus can rise during a successful rescue.
4. **Expand failed states while preserving their function.** Compare live
   splits, fresh zero-output units, canceling signed pairs and same-width
   perturbations as specified in the width note. Include reset-only controls.
   A rescue by a fresh feature but not an infinitesimal live split would fit
   the fixed-gate calculation, rather than refute all width mechanisms.
5. **Then test regime and teacher dependence.** Control feature-learning
   strength across widths and alter teacher mean/sign/angle structure while
   retaining realizability. Report attainable phase coverage and failures;
   conditioning only on runs reaching low loss would select narrow successes.

For a theory paper, a plausible target is a conditional statement: width
can improve access to residual-correcting directions, while a specified
optimizer and step regime determine the cost of using them. The local
identities above can support lemmas and interventions. A universal monotone
competition scalar, or a causal theorem for final fitting, is not currently
supported. These measures are new proposals for this repository, not claims
of novelty relative to the entire optimization literature.
