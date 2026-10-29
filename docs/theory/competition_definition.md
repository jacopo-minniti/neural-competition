# An operational definition with quantitative damage forecasts

23 October 2026. Candidate: **coupling-induced optimization damage**. This defines
one mechanism precisely; it is not a universal definition of all feature starvation.
The [MNIST tests](../reports/coupled_modes.md) support short-horizon forecasts and
show where they fail. No Gaussian inputs or teacher decomposition are required.

## 1. What is being held fixed?

Specify the objective L, a partition of parameters into neuron blocks, the
optimizer metric P, an initial state and a prediction horizon T. Competition
cannot be an intrinsic scalar of the represented function alone: rotating or
mixing feature coordinates can remove apparent coupling while representing the
same function. Neuron grouping, optimizer and time horizon belong in the definition.

At the current state, use a positive-semidefinite local curvature H. For an
empirical squared loss and locally linear predictions, H=JᵀJ/n. More generally
use generalized Gauss–Newton curvature Jᵀ H_loss J/n. The exact nonlinear Hessian
also contains residual-weighted second derivatives; omitting them is an
approximation whose predictive accuracy must be measured.

Write D=blockdiag(H_11,...,H_mm). First assume D,H are positive definite on the
identified subspace. Define

\[
C=D^{-1/2}HD^{-1/2},\qquad C_{rr}=I.
\]

C describes interaction after normalizing each block's own response. Singular
blocks require a stated restriction/pseudoinverse; numerical eigenvalue cutoffs
are approximations, not invisible theoretical assumptions.

For a parameter change v, define its **destructive overlap**

\[
\chi(v)=1-\frac{v^THv}{v^TDv}.
\]

For squared loss this compares ||Σ J_r v_r||² to Σ||J_r v_r||². Positive χ means
the blocks partly cancel in prediction space. It does not alone establish damage:
the direction might be irrelevant to the unresolved target. This distinction
prevents treating every overlap, shrinkage or conflict as harmful competition.

## 2. Definition of damage and its reference dynamics

Let the local quadratic have minimizer theta*, x=D^(1/2)(theta-theta*), and
q=C^(1/2)x. Then L-L*=||q||²/2. For gradient flow with block-diagonal positive
metric P, put B=D^(1/2) P D^(1/2). The actual local dynamics are

\[
\dot q=-Aq,\qquad A=C^{1/2}BC^{1/2}.
\]

Define the **decoupled reference** by q_dot=-Bq. It removes C while retaining B
as the isolated-block speed matrix. It is implemented on the SAME quadratic
objective by the positive-definite parameter preconditioner

\[
P_{\rm cf}=D^{-1/2}C^{-1/2}BC^{-1/2}D^{-1/2}.
\]

When C=I, P_cf=P: there is no intervention. Since B is block diagonal and
C_rr=I, tr(A)=tr(B). Thus the reference preserves total local curvature rate;
it redistributes speed across modes rather than multiplying every speed by an
arbitrary constant. This is a precise reference convention, not a uniquely
privileged counterfactual. It changes update directions, costs extra computation,
and does not match every parameter or prediction displacement.

**Definition.** At a specified horizon, signed coupling cost is

\[
\Delta(T)=L(\theta_{\rm ordinary}(T))-L(\theta_{\rm cf}(T)).
\]

Positive Delta is optimization damage relative to this reference; negative
Delta is cooperation. For nonlinear empirical branches, freeze P_cf at the
branching state for this local intervention. If using a different schedule, say
so explicitly. Held-out damage is a separate contrast on held-out L; training
damage does not imply worse generalization.

For the local quadratic this gives the quantitative forecast

\[
\boxed{\Delta_{\rm quad}(T)=\tfrac12 q_0^T
  \left[e^{-2AT}-e^{-2BT}\right]q_0.}
\]

It uses the initial state, current derivatives and labels, not future losses.
Discrete gradient descent replaces each exponential by the corresponding
matrix power: (I-eta A)^(2k) and (I-eta B)^(2k), with a stable eta. The forecast
is exact for that quadratic model, not for arbitrary neural training.

The definition is a specific optimization-intervention effect. A successful
prediction does not establish that all benefits of that preconditioner are
uniquely attributable to a natural-language notion of neuron competition.

In the current MNIST experiments, the intervention is constructed from the
conditional **readout** curvature, with one readout vector per neuron as a block.
Hidden-layer updates remain ordinary. The full-Jacobian forecast then predicts
the response of all parameters to that intervention. This is not an experiment
removing every cross-block interaction in the entire network.

## 3. Interpretable special case: normalized block learning

If P=D^-1, then B=I. Let C v_i=lambda_i v_i, and E_i be the initial quadratic
loss in mode i. Then

\[
L(T)=L_*+\sum_i E_i e^{-2\lambda_iT},\qquad
\Delta(T)=\sum_i E_i(e^{-2\lambda_iT}-e^{-2T}).
\]

Define the nonnegative slow-mode burden and fast-mode benefit separately:

\[
\mathcal C(T)=\sum_{\lambda_i<1}E_i(e^{-2\lambda_iT}-e^{-2T}),
\quad
\mathcal B(T)=\sum_{\lambda_i>1}E_i(e^{-2T}-e^{-2\lambda_iT}),
\quad \Delta=\mathcal C-\mathcal B.
\]

This answers both questions: **how much destructive overlap?** (1-lambda_i in
each relevant mode) and **how much damage by T?** (the E_i-weighted forecast in
loss units). Small eigenvalues without residual energy predict no damage.
Do not report C while hiding B. Modes with zero reachable curvature contribute
to unlearnable residual, not to a finite slow-mode relaxation time.

For two equally normalized correlated features,
C=[[1,rho],[rho,1]]. Their common mode has lambda+=1+rho; their contrast mode
has lambda-=1-rho. A target requiring the contrast learns 1/(1-rho) times more
slowly in this model. At rho=.95 it is 20 times slower. If initial error E lies
entirely in that contrast, excess loss at T=1 is
E(exp(-.1)-exp(-2))≈0.7695 E. The same overlap accelerates a common-mode target.
This is a conditional prediction, not a universal penalty for similarity.

Integrated slow-mode excess loss over infinite time is
sum_{lambda<1} E_i(1/lambda_i-1)/2 when all relevant lambda_i>0. This is area
between two specified quadratic relaxation curves, not accumulated causal
damage along an unrelated ordinary trajectory.

## 4. Real networks, moving representations and optimizer history

For a general network, form a local prediction model
f(theta+delta)≈f(theta)+J delta. J includes hidden weights, biases and readouts.
Its branch forecasts can be computed with Jacobian-vector products; explicit
full matrices are unnecessary. On another dataset, apply its own Jacobian to
the forecast parameter changes and evaluate the chosen loss. That predicts
held-out risk separately rather than treating training progress as generalization.

For squared loss, the frozen-Jacobian rollout is a quadratic model. For
cross-entropy, its Gauss–Newton quadratic is only local; a fixed-Jacobian logit
model can instead retain the nonlinear cross-entropy. The current experiments
test squared-loss MNIST only. A softmax extension is not an empirical result.

For momentum on a fixed quadratic, each scalar mode obeys
x_{k+1}=(1+beta-eta lambda)x_k-beta x_{k-1}; the initial velocity is required.
Frozen Adam preconditioning can be incorporated into P, but evolving moments,
minibatch noise and stochastic gating require additional state dynamics. The
ordinary metric experiment uses full-gradient GD, not original minibatch SGD
or a validated Adam forecast.

There is a real limitation on horizon. Under squared-loss gradient flow with
fixed metric, exact residual dynamics use a moving PSD kernel K(t). If
||K(t)-K(0)||≤epsilon throughout [0,T], variation of constants and contraction
give ||r(T)-exp(-K(0)T)r(0)||≤epsilon T||r(0)||. In normalized data norm this
implies a loss forecast error at most epsilon T||r(0)||². For a difference of
two branch losses, add the two bounds. This is useful only when kernel drift
can be bounded in advance; measuring its future value is an audit, not prediction.

This also clarifies an important obstruction: K=sum_r K_r with K_r PSD, so
adding a neuron's fixed tangent kernel cannot reduce instantaneous full-gradient
descent on a fixed residual. A claim that wider networks improve by removing
all instantaneous destructive gradient work would contradict this identity.
Competition here concerns mode speeds under a stated resource normalization
and their finite-time objective consequences.

## 5. Relation to width and existing theory

The connection to earlier feature-formation results is more specific than
"conditioning matters." In squared-loss readout gradient flow, coefficient lag
delta=a-a*(W) obeys delta_dot=-G delta-a*_dot. In a fixed Gram eigenbasis with
eigenvalue lambda, a moving optimum forces

    delta_i(t)=exp(-lambda t) delta_i(0)
               -integral_0^t exp(-lambda(t-s)) a*_dot_i(s) ds.

If |a*_dot_i|≤V_i, then |delta_i(t)|≤exp(-lambda t)|delta_i(0)|
+ V_i(1-exp(-lambda t))/lambda. Thus slow modes can amplify tracking lag by
1/lambda, which changes hidden-layer gradients and hence subsequent feature
formation. This requires forcing in those modes; small eigenvalues alone do
not imply large lag. Moving eigenvectors add further terms, so this bound is
conditional, not a completed prediction of sign conversion or final feature loss.

No monotone-width theorem follows from this definition. Width can lower the
unlearnable floor and simultaneously increase the slow-mode burden. The MNIST
probes exhibit this. A width explanation must predict changes in the entire
residual-weighted spectrum and representation dynamics, not just lambda_min.
The sign-survival experiments constrain an additional mechanism: changing
representations and available signed capacity can alter that floor over time.

This framework builds on established kernel dynamics and curvature conditioning:
[Jacot et al. (2018)](https://arxiv.org/abs/1806.07572) and
[Martens and Grosse (2015)](https://proceedings.mlr.press/v37/martens15.html).
The eigenvalue relaxation formula is not claimed as new theory. The research
question is whether an explicitly specified inter-neuron intervention has
predictable objective cost in nonlinear trajectories, and whether its width
dependence explains a meaningful part of observed learning failures.

The present evidence supports a short-horizon candidate, with measured failure
at longer horizons. It does not yet meet the stronger goal of predicting total
competition damage throughout arbitrary real-network training.
