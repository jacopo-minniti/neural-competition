# When positive pair cost is competition, and when it is shared correction

Follow-up: the [allocation experiments](../reports/allocation.md) now measure
residual work, corrective cosine and relative-step allocation regret, including
held-out exact interventions and controls. They do not find a universal width
ordering. The derivations below motivate the objects, rather than guaranteeing
that they support the competition hypothesis.

This note addresses the reversal in [the robustness report](../reports/robustness.md),
the vanishing tiny-step events in [the pair report](../reports/pair_interactions.md),
and the persistent early direction effect in [the momentum report](../reports/momentum.md).
The calculations below are deductions for our additive network and half-MSE,
not additional empirical results or a theorem imported from a different model.

A subsequent [matched-loss check](../reports/strict_matched_loss.md) tests
the stricter frequency/severity question empirically. Default Adam's wide
models have fewer strict events per eligible pair in the shared loss range;
that differs from their whole-training frequency ranking and from C_plus.
The report retains raw event counts and harm to avoid identifying a lower
per-pair rate with fewer total events.

The [temporal peer-update experiment](../reports/temporal_branches.md) also
clarifies the causal meaning of "standalone": an isolated loss-change test
does not make the proposed direction independent of other neurons' history.
Earlier peer movements can change later H. The measured short-pulse response
is substantial relative to the initial pulse impact in narrow momentum, but
small for low-rate Adam and not universally ordered by width.

## Adam's accounting already points outside the pair term

In the full five-seed teacher-8 / Adam-.001 cohort, accumulated reference
standalone harm H falls from .656744 to .180301 at width 8→32, while C_plus
rises from .034504 to .057844. The reduction in H is about .4764 half-MSE
units, versus about .02334 extra pair cost. G also decreases, so these numbers
must be interpreted through the complete identity H−G+C_plus−C_minus.
They locate a candidate mechanism: better individual proposed corrections or
features could outweigh increased overlap between useful corrections.

This makes a prediction worth testing: a metric of residual correction or
available feature directions might improve with Adam width even when C_plus
increases. It does not establish that such a metric actually does improve.

## A counterexample that any replacement should pass

Write empirical inner products as <v,w>=mean_i(v_i w_i). The existing exact
finite neuron changes d_r satisfy

\[
C_{\rm net}=C_+-C_-
=\tfrac12\left(\|d\|^2-\sum_r\|d_r\|^2\right),\qquad d=\sum_r d_r.
\]

Now replace each d_r by q identical contributions d_r/q, preserving both
the original function and the total prediction change. Between different
original neurons the sum of positive and negative pair terms is unchanged.
The new pairs within each group of copies contribute positive cost. Therefore

\[
C'_+=C_+ + \frac{1-1/q}{2}\sum_r\|d_r\|^2,
\qquad C'_-=C_-.
\]

The actual loss change is identical. The standalone budget changes to compensate:
sum A'_r=<e,d>+(1/2q)sum ||d_r||². Consequently neither raw C_plus nor
C_plus/G is invariant to subdividing the same useful movement.

For a scalar example, take e=−1 and total correction d=.2, shared equally
among m neurons. Every isolated update helps, the joint gain is always .18,
but C_plus=.02(1−1/m): zero at m=1 and approaching .02 as width increases.
Calling all of this positive overlap destructive competition would misclassify
an ordinary cooperative correction.

This is a counterexample about prediction changes and their decomposition.
It is realizable by splitting the before/after neuron functions with the
appropriate outgoing coefficients. It does **not** assert that unchanged
SGD or Adam automatically produces these same changes after a split.

**Consequence for our claim.** C_plus remains a valid exact accounting term.
Its ordering across widths cannot be a necessary condition for better
optimization, even after matching loss or standalone activity. A new measure
should be checked on this duplication example before interpreting its width
trend. Arbitrarily regrouping neurons also changes pair accounting.

## Why shortening a step removes strict events

At a differentiable state, away from gate crossings on the evaluation set,
let delta_r=−eta h_r and let F_r be that neuron's prediction Jacobian. Then

\[
d_r=-\eta F_rh_r+O(\eta^2),\qquad
A_r=-\eta g_r^\top h_r+O(\eta^2),\qquad
J_{rs}=\eta^2\langle F_rh_r,F_sh_s\rangle+O(\eta^3).
\]

If g_s^T h_s is strictly positive, the O(eta²) interaction cannot reverse
the O(eta) standalone benefit as eta→0. Strict events can survive near units
whose first-order benefit is zero or very small, but they should become rare
under ordinary small-step conditions. Their disappearance is therefore
expected from the definition, even if the same representation remains hard
to optimize.

Across batches, by contrast, g_C^T h_B can be negative at first order.
Shortening eta preserves that sign. This explains why momentum's early
cross-batch signal can persist at 1% while same-batch strict events disappear:
the two diagnostics have different leading orders. Neither result contradicts
the other.

With uniformly bounded smooth dynamics over fixed physical time T*eta,
sum_t C_plus,t=O(eta), while total first-order progress can remain O(1).
Gate crossings, stationary degeneracies, constant-rate stochastic noise and
momentum scaling require separate treatment. Even with those qualifications,
a finite-step pair penalty alone cannot explain every width advantage that
also exists under gradient flow.

## Separate useful correction from perpendicular movement

For the actual total prediction change d and residual e, define

\[
s=-\langle e,d\rangle,\qquad
d_\perp=d+\frac{s}{\|e\|^2}e.
\]

For nonzero residual, the exact loss budget can instead be written

\[
\Delta L=-s+
\underbrace{\frac{s^2}{2\|e\|^2}}_{\text{curvature along the residual}}
+\underbrace{\tfrac12\|d_\perp\|^2}_{\text{perpendicular movement}}.
\]

This budget uses the total function change, so it survives the duplication
counterexample. It separates shared movement along the desired correction
from motion perpendicular to that correction. The latter always increases
this step's squared loss relative to its linear term; it can still be an
investment in useful future features. It is not a removable fraction of
eventual fitting error.

Here the linear term is in **prediction space**. For a finite nonlinear
parameter step, s need not equal minus the parameter-gradient dot product.

Two useful companions are

\[
\mathcal E=\frac{[s]_+^2}{\|e\|^2\|d\|^2}\in[0,1],\qquad
\rho=\frac{\|d\|^2}{2s}\quad(s>0).
\]

E is the squared corrective cosine, with anti-corrective directions assigned
zero. Rho reports how much of the first-order gain is consumed by the finite
prediction movement; the actual step helps exactly when rho<1. Keep s and
Delta L alongside these ratios: a perfectly aligned tiny step can make little
progress. Ratios are undefined for the relevant zero denominators.

The [objective-scale audit](../reports/effect_magnitude.md) now recovers total
linear work and total squared-movement cost from the old sufficient statistics,
and reports their cumulative ratio. The corrective cosine and parallel versus
perpendicular split are now measured in the [allocation study](../reports/allocation.md).
The earlier R_perpendicular
measures reinforcement **between neuron changes**; it does not already supply
the total-step residual decomposition. Pool raw numerators/denominators within
seed and keep independent-evaluation collateral harm alongside them.

The audit also derives the exact work budget P−N−Q, where P/N separate useful
and anti-residual neuron work. These survive subdividing identical corrections.
Existing sums bound N and reveal a substantial reduction for the widest
low-rate Adam case, but not a monotone reduction at every intermediate
width. This is an objective-unit candidate to test, not a validated mechanism
or a metric chosen to guarantee optimizer agreement.

## A stricter question: would different neuron step weights actually help?

Positive J alone cannot answer this. A local alternative is to ask whether
relative weights on the proposed neuron directions improve on the best
uniform step length.

Use linearized changes ell_r=F_r delta_r, vector b_r=<e,ell_r>, and the
positive semidefinite Gram matrix M_rs=<ell_r,ell_s>. The local objective is

\[
Q(\alpha)=b^\top\alpha+\tfrac12\alpha^\top M\alpha,
\qquad 0\le\alpha_r\le1.
\]

Define the predicted gain from changing relative weights as

\[
\mathcal R=
\min_{0\le\gamma\le1}Q(\gamma\mathbf1)
-\min_{\alpha\in[0,1]^m}Q(\alpha)\ \ge0.
\]

The inequality follows from inclusion of the uniform choices. Comparing
against the best uniform shrink separates global step overshoot from a
potentially better allocation of motion between neurons. Identical equally
useful copies in the example above give R=0, despite their positive J.
The domain includes suppressing a proposed update while keeping that neuron's
existing function.

More generally, replacing any ell_r by q copies ell_r/q leaves the feasible
set of total corrections unchanged: the copies' average coefficient can be
any number in [0,1]. Both optima, and hence R, are invariant to this exact
subdivision. Adding genuinely different directions can change the feasible
set and the result. This distinction supplies a check against merely counting
more neuron pairs.

R is a **local diagnostic proposed here**, now tested in the allocation study,
not a published result from the papers in these notes. It detects avoidable allocation cost, including individually harmful
updates; it is not exclusively a pair interaction measure. More alpha
coordinates provide more fitting freedom, so include fixed-count neuron-group
and random-group controls. Selecting alpha on a source set and evaluating it
on a separate set distinguishes useful transfer from fitting the diagnostic.

For a real intervention, apply alpha to the parameter blocks and recompute
the exact forward loss. Scaling a parameter update does not generally scale
its finite d_r: both layers move and gates can change. Using the existing
finite d_r in Q is exact at binary subset choices, but a fractional alpha
then describes an interpolation of neuron **functions**, not necessarily
the proposed parameter update. Check shortened steps and actual forwards
before claiming this local quadratic diagnostic predicts an intervention.

If R is smaller in wider SGD, momentum **and Adam**, after these controls,
that would support a shared allocation mechanism. If Adam has larger R but
better residual access, the mechanisms differ. Both outcomes are scientifically
useful; neither warrants redefining competition to erase the difference.
