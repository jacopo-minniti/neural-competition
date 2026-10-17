# Interference measurements

There is no single privileged scalar. We distinguish conflicting demands on
parameters, the harm caused by an actual update, and cancellation between neuron
contributions to predictions. These answer related but different questions.

## Conflicting minibatch demands

At the same parameter state, let $g_B=\nabla_\theta L_B$. A small SGD step
on batch B changes batch C's loss to first order by

$$
L_C(\theta-\eta g_B)-L_C(\theta)\simeq-\eta\,g_C^\top g_B.
$$

A negative dot predicts harm to C. We retain raw dot distributions and cosines,
which separate gradient magnitude from orientation. Self-pairs are excluded;
64 probe batches give 2,016 unordered pairs, which are correlated observations.
The fraction of negative pairs is not the probability of an independent trial.

The neuron block is $(w_r,b_r,a_r)$, with 22 coordinates. Its contribution

$$
q_r(B,C)=g_{B,r}^\top g_{C,r},\qquad\sum_rq_r=g_B^\top g_C
$$

attributes conflict to the shared parameters of individual neurons. Summing
`max(-q_r, 0)` measures negative mass that may be hidden by positive contributions
from other neurons. Its raw magnitude depends on loss, gradient scale and width.

## Actual update and finite-step harm

For the recorded optimizer update $\delta=\theta_{t+1}-\theta_t$,
$g_C^\top\delta>0$ predicts harm. We also measure the actual change
$L_C(\theta+\delta)-L_C(\theta)$. This uses the optimizer's actual direction
and extends to momentum/Adam, although the focused collection uses plain SGD.

The observed-minus-linear remainder includes smooth higher-order effects and
ReLU gate crossings. It is not a pure second-order statistic. The consecutive
actual-update cosine is saved as `step_previous_cosine`; for SGD it also reflects
consecutive sampled gradient directions, subject to parameter rounding.

## Cancellation between neuron prediction changes

Write $f=\sum_r u_r$. At a saved step, we can directly compute each finite
contribution change $\Delta u_r(x)$, or its linearization

$$
\delta u_r(x)=\frac{\delta a_r\operatorname{ReLU}(z_r)
+a_r\mathbf1_{z_r>0}(x^\top\delta w_r+\delta b_r)}{\sqrt m},\qquad
z_r=w_r^\top x+b_r.
$$

Their Gram matrix on a fixed input set measures whether neuron movements reinforce
or oppose one another in prediction space. A useful descriptive ratio is

$$
R=\frac{\mathbb E_x[(\sum_r\delta u_r(x))^2]}
        {\sum_r\mathbb E_x[\delta u_r(x)^2]}.
$$

R below one indicates net cancellation relative to the individual movements;
R above one indicates reinforcement. Neither alone says whether training improves:
compensation may protect already-correct predictions. This ratio is different
from conflict between minibatch gradients and is undefined if all movements vanish.

For a neuron/example whose activation gate stays fixed, the exact remaining
prediction term is

$$
\Delta u_r-\delta u_r=
\delta a_r\mathbf1_{z_r>0}(x^\top\delta w_r+\delta b_r)/\sqrt m.
$$

For ReLU this term comes from moving both layers together. Pure incoming-weight
motion with fixed outgoing weights has no quadratic prediction term within a
fixed activation region. Gate crossings supply an additional nonsmooth effect.
The notebook marks crossings and checks the expansion only where it applies.
These diagnostics inspect the existing runs; they are not a neuron-splitting
intervention or evidence that coordinated cancellation causes the width benefit.

## Curvature and interpretation

The original curated snapshots save the mean of four individual
$g_B^\top H_Bg_B/\|g_B\|^2$ quotients for the **half-MSE**, computed by
Hessian-vector products in float64. It is not the full-data top Hessian eigenvalue
or a ratio of expectations. NaN is legitimate when a direction is exactly zero.
The plain-SGD reference 2/lr is illustrative; four batches do not establish an
edge-of-stability regime, and this reference is not applied to other optimizers.

Compare gradients both at equal update counts and at similar training loss.
Checkpoint loss matching is approximate: display the actual losses rather than
silently treating them as identical. Very small losses introduce rounding and
near-zero direction effects. Curated seeds illustrate contrasting outcomes;
they cannot establish population success rates or a causal interference mechanism.


## Follow-up distribution study

The follow-up in [pier_experiments.md](pier_experiments.md) stores curvature
numerators and denominators separately. Its `batch_sharpness_ratio` estimates
E[gBᵀ HB gB] / E[||gB||²], the definition in Pier's paper. The old field
`batch_sharpness` remains the mean of individual ratios for compatibility.
The new study adds full distribution plots, standardized skewness/fourth moments,
seed-level uncertainty, momentum/Adam directions and finite-step interventions.


## Net agreement A and the information in neuron blocks

For independent minibatches at a frozen parameter state,

$$
A=\frac{\mathbb E[g_B^\top g_C]}{\mathbb E\|g_B\|^2}
=\frac{\|\nabla L\|^2}{\mathbb E\|g_B\|^2}.
$$

`scripts/plot_agreement.py` plots this quantity from saved SGD snapshots for widths
8 and 16, with ten seeds and pointwise seed-bootstrap intervals. The full-training
gradient supplies the numerator; 128 probe batches estimate the denominator. Each
state is measured before its recorded update, so step t means t completed updates.
The two widths need not have equal loss at the same step. A's population range is
[0,1] when defined, but its estimated denominator has Monte Carlo error; it is
undefined when both numerator and denominator vanish. Small gradients also make
late ratios numerically sensitive. A higher value alone does not imply a better fit.

Summing the signed neuron contributions gives exactly the original global dot.
Information is gained by keeping neuron identities or applying a nonlinear summary
before summing. In particular,

$$
\sum_r\max(-q_r,0)\ \geq\ \max\left(-\sum_rq_r,0\right).
$$

For example, q = (-2,3) and q = (0,1) both give X = 1, but their summed negative
neuron contributions are 2 and 0. Separate positive/negative totals reveal hidden
incompatible batch demands. They still discard which neurons and which batch pairs
are involved, and are not direct measurements of interactions between distinct
neurons' prediction changes. Testing a neuron mechanism requires those patterns or
interventions, beyond reconstructing the global sum.


## Neuron-level follow-up

See [neuron_interference.md](neuron_interference.md) for P/N, exact individual-neuron prediction changes, partial cancellation, and full-step versus short-step collateral harm.
