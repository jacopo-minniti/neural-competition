# Prospective specialization mechanism study

Design recorded before running the new training study, 18 October 2026.
This follows the request to derive a mechanism and construct regimes that test
it, rather than search old trajectories for a favorable diagnostic.

## Mechanistic hypothesis

Correlated target features can first be represented by one shared feature.
Moving that indispensable feature toward one constituent disturbs its existing
contribution; two locally available neurons can separate while canceling the
first-order disturbance. The candidate object is the loss operator on
mean-preserving splitting directions, compared with ordinary motion of the
shared feature. Extra width helps conditionally by supplying such directions
where useful features share an insufficient number of neurons. More width alone
does not guarantee either supply or symmetry breaking.

For Gaussian ReLU features, a pair v±=cos(theta)c±sin(theta)d has the exact
innovation relative to its common feature:

relu(v+·x)+relu(v-·x)-2cos(theta)relu(c·x)
  = (sin(theta)|d·x|-cos(theta)|c·x|)_+.

The innovation occupies a wedge of probability 2theta/pi and has squared norm
of order theta^3 near zero angle. At the optimum single merged feature without
bias, ordinary angular motion is locally stable for
sin(theta)<(pi-theta)cos(theta), while an antisymmetric split has a negative
population quadratic loss coefficient for theta>0. We will derive and
numerically check these coefficients before interpreting training outcomes.

## Predictions and failure conditions

1. In the stable merged-feature regime, clustered allocations with only one
   neuron at some target pairs should specialize less readily than a balanced
   allocation, even when total width equals the teacher's eight units.
2. Function-preserving expansion plus small generic symmetry breaking should
   rescue this shortage more than exact unperturbed duplication at a learning
   rate mapping that preserves identical-copy dynamics.
3. The effect should disappear at theta=0 (no distinct features to learn).
   It need not grow monotonically as theta decreases: the required innovation
   shrinks and the splitting growth rate also shrinks.
4. Standard random initialization is a transfer test, not guaranteed by the
   local calculation. Failure there limits the mechanism's scope.
5. Bias training and optimizer changes may enable alternative escape routes.
   The local population result is not a theorem for Adam or finite-sample ReLU.

## Training design

Gaussian inputs in dimension 20; four orthogonal feature-pair planes; eight
teacher units; half-angles 0/15/30/60 degrees; normalized target variance;
both incoming and outgoing weights trained. No label noise in this first study.
Use 4,096 training and 4,096 independent test examples, minibatches of 128,
10,000 updates initially, five paired seeds. Positive equal teacher outputs
is an explicit assumption, not a claim about all target sign structures.

Mechanism-isolating initializations start at the shared-feature representation:
width 8 with counts [1,1,3,3], width 8 with [2,2,2,2], and width 16 produced by
duplicating each neuron of the first state. Per-cluster outgoing sums agree,
so the unperturbed functions agree. Small random zero-sum within-cluster
incoming perturbations break symmetry without selecting teacher contrast axes.
Also include exact unperturbed duplication. Rotation of which clusters are
underallocated is paired by seed. This teacher-informed initialization is a
controlled bridge; it is not presented as ordinary random training.

For standard random initialization, compare widths 8/16 with small random
outgoing weights and ordinary Gaussian incoming directions. Include fixed-rate
and identical-copy-matched layer-rate conventions. Bias-free models anchor the
derivation; bias-training sensitivity checks remove that restriction.

For SGD/momentum, expansion by q uses incoming rate q times the narrow rate
and unchanged readout rate in the a/sqrt(m) parameterization. For Adam, use
unchanged incoming rate and readout rate divided by sqrt(q). The Adam mapping
is approximate when epsilon matters. Exact clones without perturbations provide
a numerical check; mapped rates alone do not equalize generic trajectories.

Keep held-out loss, progress relative to the shared-feature baseline, and
within-pair feature resolution as outcomes. Do not claim direct neuron conflict
from a fitting difference alone. Preserve negative outcomes and all settings.
Any later changes to this design must be identified as follow-up exploration.
