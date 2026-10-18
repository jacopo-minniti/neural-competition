# Residual growth and competition without neuron splitting

Prospective design, 18 October 2026. No runs of this study have been inspected
when this file is first written. This follows the request for a signed
mechanism in ordinary training, stronger effects, and no splitting intervention.

## Object

For a bias-free ReLU student f=Σ a_r ReLU(w_r·x)/sqrt(m), write
u_r=w_r/||w_r|| and c_r=a_r||w_r||/sqrt(m). Define the residual potential

H_f(u)=E[(y-f)ReLU(u·x)].

Under equal-rate full-gradient flow,

dc_r/dt=(a_r²+||w_r||²)H_f(u_r)/m.

Thus sign(a_r)H_f(u_r)<0 decreases that neuron's signed-role magnitude.
Unlike a split, this object describes neurons already present in ordinary
training. Gaussian inputs supply an exact ReLU-kernel expression for H_f.

If a shared feature span C has been fitted by least squares, residual target
signals are governed by the conditional Gram operator

S=G_TT-G_TC G_CC^(-1) G_CT,   h=S beta.

Although S is positive semidefinite, its off-diagonal entries can be negative,
so another teacher component can reverse a required feature's growth signal.
For two flanking directions at angles ±theta from one shared center,
S11=1/2-2k(theta)² and S12=k(2theta)-2k(theta)². With outputs (1,r),
the first feature's signal changes sign at r=S11/(-S12), when S12<0.
This is a conditional mechanism, not a claim that every random trajectory
visits the fitted shared span. Width does not automatically alter this operator.

## Tests and falsification

Use four orthogonal feature-pair planes in dimension 20, eight teacher ReLUs,
half-angles 30 and 60 degrees initially, pair coefficients (1,r) with
r=1,3,-3. Compute the full eight-feature/four-center conditional Gram operator,
not an independence approximation across positive ReLU features.

All students start from ordinary independent Gaussian incoming directions and
small signed random readouts. Compare widths 8,16,32; no copies, assigned
clusters, teacher-informed initialization, or mid-training expansion. Both
layers train. Anchor with bias-free models; test biases if promising.
Targets have unit training variance. Use 4,096 training/test examples,
minibatches of 128, 10k steps initially, SGD .3, momentum .03, Adam .003,
five paired seeds. Compare unchanged rates and the prior width-rate mapping
(incoming multiplied by width/8 for SGD/momentum; Adam output rate divided
by sqrt(width/8)). Random initial functions are not matched across widths.

Record raw held-out half-MSE, actual residual correlations with each teacher
feature, and closeness to the candidate shared span at sparse checkpoints.
The precomputed conditional sign is a prediction of growth after fitting that
span, not a label assigned after reading the outcome. If negative-sign settings
do not show stronger ordinary-training obstruction or width advantage than
positive-sign controls, reject this sign as an established predictor of width
benefit. A large mean improvement driven by few seeds must be reported as such.
More negative residual signals do not alone establish worse learning.

Use absolute loss as well as fractions; do not promote a tiny-detail relative
gain to a strong phenomenon. Replication, longer budgets, different rates,
data draws, or additional geometry are follow-ups and must be labeled as such.
