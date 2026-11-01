# Extending reserve through controlled assumption relaxations

31 October 2026. This is the current research direction. Start from the
[successful completion-reserve prediction](competition_reserve.md), preserve its
prediction target, and relax data assumptions separately. The
[first bridge experiment](../evidence/reserve_bridge/README.md) has been executed.
The earlier [parity exploration](../../archive/docs/model/frequency_pressure.md) is retained as a
diagnostic branch; accurate simulation of future SGD is not the research target.

## 1. The result to preserve

The previous success was a state-derived answer to:

> Which currently available corrections will lose substantial response while
> an important learning component is acquired, and how does width change this?

The completion-reserve curve R_r(C;theta) evaluates the existing analytic
dominant-pressure flow after a specified amount C of functional learning. It
includes coefficient/feature coupling and physical gate events. Protected
reserve distinguishes current response from alternatives not exposed to that
pressure. These are more useful starting objects than a new simulator of every
gradient update.

The empirical anchor had a strong, coherent, still-unlearned component, weaker
other updates, opposing observation geometry, and sufficient model capacity.
Their importance must be tested individually, rather than discarded together.

## 2. Why raw parity was too large a jump

The parity experiment changed more than feature frequencies:

- The most frequent empty input had target zero and was already close to its
  initialized prediction. In the retained confirmation banks, its gradient norm
  is only about .3%–.7% of the full gradient norm at initialization, depending on
  width. It has probability .288, not approximately .96 as in the weak-tail
  reserve anchor. Frequency did not produce the required learning pressure.
- With raw bits and an intercept, every input has positive dot product 1 with
  the empty input. The previous opposing-support closure geometry is absent.
- Full parity replaces shared low-order corrections with a nonlinear joint
  rule. Factor occurrence, correction demand, and available alternative routes
  all change at once.

Finite gate changes and residual sorting remain useful lessons from that run.
The full-covariance SGD forecast does not address the missing macro explanation
and is not the baseline around which the next extension is organized.

## 3. First relaxation: co-occurrence with everything else controlled

Let Z_i be zero when factor i is absent and an independent fair sign when it
is present. Its occurrence probability is q_i; q_i follows the old smooth rank
spectrum and Q=sum_i q_i<1. Compare two presence laws:

1. The original exclusive law: at most one factor is present.
2. Independent Bernoulli presence with exactly the same q_i.

A fixed mixture parameter interpolates between these laws while preserving
every marginal. This parameter is fixed within each training run; it is not a
curriculum. The comparison retains the old shared target, initialization,
optimizer, observation geometry, model capacity, and prototype audit.

Use u=e_0, gamma=1.5, s=sin(pi/3), and the original direction construction v_i:

\[
 x(Z)=(1-\gamma\sum_i1_{Z_i\ne0})u+s\sum_i Z_i v_i,
 \qquad y(x)=4[u^Tx]_++t^Tx.                         \tag{1}
\]

Choose t so its factor contribution is the original signed linear attribute.
When no factor or one factor occurs, (1) is exactly the previous task. Multiple
factors combine through the same observation rule and the same global target.
The exact three-unit capacity witness remains valid. The first enumerable
bridge uses four factors, with an explicit regression test against the original
four-factor endpoint; it is not a claim of transfer to arbitrarily many factors.

This is a temporary experimental bridge, not a replacement naturalistic
benchmark. At Q=.04 true co-occurrence is rare. That is intentional for the
first isolated relaxation and limits the strength of its realism claim.

### A structural reason the old prediction can survive

Write W=w^Tu, B_i=w^Tv_i. For gamma>1, a unit is inactive on every nonempty
input in (1) exactly when it is inactive on every signed single-factor input:

\[
 W\ge\frac{s\max_i|B_i|}{\gamma-1}.                  \tag{2}
\]

Necessity follows from the single-factor pairs. For sufficiency, if k>=1
factors occur, preactivation is at most

\[
 (1-\gamma k)W+sk\max_i|B_i|
 \le(1-k)W\le0.
\]

Thus adding co-occurrence does not require a new all-support gate-closing
threshold. It changes the training pressure and provides additional contexts
for recovery. The old completion approximation remains plausible only while
those additional updates are a sufficiently small perturbation. This is the
premise tested next, not presumed from q_i alone.

## 4. Keep the probabilistic target simple

For a declared remaining learning amount C and a current state, keep the old
prediction R_r(C;theta) and the event that response falls below half its current
value at phase end. Across initialization/data banks this predicts an aggregate
event probability for each width. Re-evaluate the state at later learning
phases. There is no fit to closure frequencies or simulation of the full future
training trajectory.

If remaining dominant learning is uncertain with conditional law G, the natural
extension is

\[
 p_r=\int 1\{R_r(C;\theta)<b_r\}\,dG(C),\qquad
 E[N_{\rm weakened}]=\sum_r p_r.                     \tag{3}
\]

A frequency- or demand-weighted endangered mass is sum_r pi_r p_r. These
expectations require no independence between corrections. A joint disappearance
law or a survival law across successive phases requires the appropriate joint
or history-conditional probabilities; independent neuron fates must not be
inserted by assumption.

The executed bridge uses a specified half-completion scenario, not a fitted G
and not calibrated per-state stochastic probabilities. Uncertainty across seed
banks is measured. Response contrasts remain probes of demanded capabilities,
not an intrinsic count of spectral options. Their structural absence with
remaining demand is stronger than the measured half-response event.

## 5. What has survived, and where the boundary appears

The same reserve predictor, without refitting, retains high ranking accuracy
and the width dependence of response weakening after introducing independent
co-occurrence. A subsequent shared-uniform presence control preserves all q_i
but makes about 45% of nonempty examples multi-factor; the prediction transfers
there too. It also survives bounded observation variation in that regime.
As Q increases and other update pressure becomes substantial, prediction
quality deteriorates. Removing the opposing geometry eliminates the event in
this cohort, as the old model predicts.

These findings are in the [evidence record](../evidence/reserve_bridge/README.md).
They justify one additional step beyond the successful anchor. They do not yet
justify arbitrary broad pressure fields or transfer to raw parity.

Two useful applicability checks are already measurable: how much update pressure
comes from outside the proposed dominant component, and whether that component
has stable geometry and meaningful remaining loss. The recorded integrated
outside/head ratio is retrospective; it explains a boundary but is not itself
a prospective forecast. The instantaneous ratio is recorded before each phase
to support the next applicability test.

## 6. Standardized parity: direction detection and a forecast boundary

The four-bit parity study uses unequal-frequency binary factors, standardized
inputs, and bipolar targets. Its frequent empty state has nonzero initial error,
and its geometry places the empty input opposite the nonempty inputs. This is a
more interactive target than the bridge, but it also changes the encoding and
target, so it is not a controlled frequency-only comparison.

The leading right singular vector of the incoming-weight update matrix provides
a candidate pressure direction without knowing the empty state. Across widths,
mean alignment with that known direction ranges from 0.738 to 0.935, and mean
coherence is above 0.71. A coherent direction does not establish the full
single-input model: at width 4, the SVD-based completion forecast predicts one
half-response loss while eight are observed, and the known-direction forecast
also predicts one. No half-response losses are observed at wider widths in this
sample, which does not establish a general protection threshold. The result and
its limitations are documented in the [parity evidence](../evidence/natural_parity/README.md).

## 7. Loss and the next primary measure

For this bridge, the functions phi_i=Z_i/sqrt(q_i) are orthonormal under both
presence laws, because signs are independent. Their target coefficients are
c_i sqrt(q_i), c_i in {-1,1}. If the entire derivative implementation of a
factor contrast is absent, the shallow model's prediction has zero coefficient
there, giving the exact current-loss lower bound

\[
 L\ge\tfrac12\sum_{i\text{ structurally missing}}q_i.  \tag{4}
\]

This gives a clean eventual connection from probabilities of missing
corrections to an expected loss floor. It does not turn half-response weakening
into a loss floor. Exact global missing-factor events were absent in the initial
development cohort, so (4) is currently a conditional link rather than an
empirically useful loss predictor. Recovery and sustained weakness must be
tested before extrapolating the phase event to endpoint loss.

The first primary measure is therefore **the expected fraction, and
frequency-weighted mass, of currently needed corrections losing substantial
response over the next natural learning phase, versus width**. Report ordinary
step ranges for the phases. Then measure how long deficits persist and whether
the corresponding residual errors remain. Exact next-step SGD prediction is
unnecessary for these targets.
