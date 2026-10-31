# Current capability, protected reserve, and tolerable common learning

29 October 2026. This revises the priority of the
[short-horizon probabilistic model](power_law_competition.md). The immediate
problem is to describe **what currently supports a correction, how much of it
is exposed to dominant learning, and what common learning it can survive**.
Predicting a parameter trajectory at a prescribed future step is secondary.

Two networks can have similar current response and very different reserves.
The useful distinction is between support that dominant training can alter and
support that it cannot. This gives an exact state certificate under broad
common-task pressure, then a more informative completion curve under a stronger
coherent-pressure approximation. Neither requires an initialization law.

## 1. Retain the functional object; add a pressure-relative decomposition

Use squared loss, the mean-field shallow ReLU network

\[
f(x)=c+\frac1m\sum_{j=1}^m a_j[w_j^Tx]_+,
\qquad M_{w_j}=M_{a_j}=mI,\quad M_c=1.
\]

The metric is fixed and block diagonal. There is no weight decay or momentum.
On a fixed audit distribution, absorb its square-root weights into the Jacobian:

\[
K=bb^T+\frac1m\sum_j Z_j,\qquad Z_j\succeq0,
\quad C_\tau(K)=K(K+\lambda I)^{-1},\quad\lambda=(2\tau)^{-1}.
\]

Here each \(Z_j\) is the derivative Gram block of a feature and its coefficient,
and \(b\) is the weighted constant-output direction. For a unit functional
contrast \(d_r\), current response is
\(\beta_r=d_r^TC_\tau(K)d_r\). This is a capability probe. Options are
the nonzero spectral components of a declared residual, together with their
responses and residual-weighted values; eigenspaces alone are capability modes.
Zero response is a closed option only when the corresponding demand remains.
Neuron counts and contrast counts are not option counts. A positive contrast
response need not mean the complete contrast lies in the correctable subspace.

Declare a common-context support \(\mathcal H\). Define the protected units and
their kernel, **retaining the original denominator \(m\)**:

\[
\mathcal F_H=\{j:\sup_{x\in\mathcal H} w_j^Tx<0\},\qquad
K_H^{\rm safe}=bb^T+\frac1m\sum_{j\in\mathcal F_H}Z_j.
\tag{1}
\]

This is a support certificate, not a role assignment. A protected unit may
implement many refinements. For box support \(x=h+\xi,\ |\xi_i|\le r_i\),
the condition is simply \(w_j^Th+|w_j|^Tr<0\). Checking a finite sampled panel
alone does not certify the population support.

**Protected-reserve theorem.** From any current state, under arbitrary subsequent
gradient updates supported only on \(\mathcal H\),

\[
K(t)\succeq K_H^{\rm safe}(0),\qquad
C_\tau(K(t))\succeq C_\tau(K_H^{\rm safe}(0)).
\tag{2}
\]

This holds for gradient flow and ordinary discrete gradient descent, including
minibatch randomness and either sign of residual pressure. The protected units
have identically zero incoming and readout gradients. Their parameters therefore
remain fixed. Every remaining kernel block is positive semidefinite. Finally,
\(C_\tau=I-\lambda(K+\lambda I)^{-1}\) is operator monotone, proving (2).
The output bias can change, but its derivative block is constant.

Consequently the two useful state quantities are

\[
\beta_r^{\rm safe}=d_r^TC_\tau(K_H^{\rm safe})d_r,
\qquad \rho_r^{\rm safe}=\beta_r^{\rm safe}/\beta_r\quad(\beta_r>0).
\tag{3}
\]

Current capability answers “can this correction respond now?” Protected reserve
answers “how much response survives *any amount* of common-only training?”
Their difference is nonnegative, but is not an additive allocation of options
to neurons because the spectral filter couples implementations.

The same operator can be projected onto the existing demand object: for a
fixed nonzero test residual \(e\),
\(e^TC_\tau(K_H^{\rm safe})e/e^TC_\tau(K)e\) is its protected share of local
response value (when the denominator is positive). This separates a large
reserve in irrelevant directions from reserve supporting actual demand. The
bound compares the same test demand across states; evolving residuals must not
be silently substituted into opposite sides of that comparison.

For currently available contrasts and declared thresholds \(b_r\), let a random
requested refinement have weights \(\pi_r\), normalized over those contrasts.
For any common-pressure history, even correlated with the requested refinement,

\[
\Pr(\beta_R(t)<b_R\mid\theta,\text{common-only training})
\le\sum_r\pi_r\,\mathbf1\{\beta_r^{\rm safe}<b_r\}.
\tag{4}
\]

No independence between features, refinements, or updates is needed. This is a
bound on the mass of potentially vulnerable demands; it need not be tight.
For a fixed set, the analogous sum without \(\pi_r\) bounds the number of
contrasts below threshold. Initially absent responses must be excluded if the
claim concerns *newly lost* responses.

### What changes under full training?

Rare examples can move protected units. The initial floor is then not an exact
bound. Retain the original selected blocks and call their later kernel
\(\widetilde K_H^{\rm safe}(t)\), whether or not they stay inactive on the head.
The resolvent identity gives the deterministic perturbation bound

\[
\beta_r(t)\ge\beta_r^{\rm safe}(0)-\delta(t)/\lambda,
\qquad
\delta(t)=\|\widetilde K_H^{\rm safe}(t)-K_H^{\rm safe}(0)\|_{\rm op}.
\tag{5}
\]

This becomes a prospective certificate only when there is an independent bound
on \(\delta\). Small rare-task frequency by itself does not supply one:
residuals may be large and gates may be near boundaries. The head margin
\(\gamma_j=-\sup_{x\in\mathcal H}w_j^Tx\), together with a bound on parameter
displacement and input norm, can certify continued head inactivity. Audit-gate
changes must also be controlled when bounding kernel drift. Recomputing (3)
tracks the reserve after other tasks alter it.

The certificate may be vacuous for broad head support. That is a substantive
boundary of this particular protection mechanism, not grounds for assuming the
missing reserve exists.

## 2. A stronger approximation: coherent common learning

A plausible restricted phase is one in which a frequent common rule supplies
most of the parameter update, its gradient directions are concentrated, and
rare/context-specific corrections are weaker perturbations. This is stronger
than just a power-law frequency assumption. It can be tested through head
geometry, residual coherence, gate margins, and the size of other updates.

For analytic control, take the common context to be \(u=e_1\), with positive
common residual. Write \(W_j=w_j^Tu\), \(A_j=a_j\), and let
\(s=\int p_H e_H\,dt\). The existing equations give, while the common gate is on,

\[
W_j(s)=W_j\cosh s+A_j\sinh s,\qquad
A_j(s)=A_j\cosh s+W_j\sinh s,
\tag{6}
\]

with perpendicular weights fixed and \(c(s)=c(0)+s\). Common-inactive units
stay fixed. If \(A_j<-W_j\), the common-active unit reaches \(W_j=0\) and is
absorbed there under the declared zero-at-zero ReLU derivative convention.
The implementation includes this event; it does not continue a negative gate.
Call the resulting full state map \(\Phi_s(\theta)\).

Let \(F(s)=f_{\Phi_s(\theta)}(u)\). Away from gate events,

\[
F'(s)=1+\frac1m\sum_{j:\,W_j(s)>0}(W_j(s)^2+A_j(s)^2)\ge1.
\tag{7}
\]

Thus a specified amount of common prediction gain \(C\ge0\) determines a
unique exposure by a scalar inversion,

\[
F(s_C)-F(0)=C,\qquad 0\le s_C\le C.
\tag{8}
\]

No future-step simulation or fit to failure rates is involved. The state-derived
**completion reserve curve** is

\[
\mathcal R_r(C;\theta)=d_r^TC_\tau(K(\Phi_{s_C}(\theta)))d_r.
\tag{9}
\]

It tells us what this state retains after completing a given amount of common
learning. Its value at \(C=0\) is current capability; its lower floor is (3).
Both layers and the physical rare gates evolve in this calculation. Negative
readouts, gate absorption, and opening or strengthening of rare responses are
allowed. In particular, **finite-strength completion curves need not be
monotone**, even in a regime with monotone eventual structural closure.

A simple qualitative prediction follows: multiplying the frequency of the
common task rescales its training clock, but does not change (9) at fixed common
gain in the common-only model. Frequency changes vulnerability in ordinary
training through the *relative competing updates and available recovery*, not
through an assumed frequency-to-death hazard.

For a declared distribution \(G\) of remaining common gain, the exact restricted
model gives

\[
\Pr(\beta_r^{\rm end}<b_r\mid\theta)
=\int\mathbf1\{\mathcal R_r(C;\theta)<b_r\}\,dG(C).
\tag{10}
\]

This integrates uncertainty in task demand rather than a future parameter path.
If demand is known, use that value of \(C\). We have not fitted or validated a
universal distribution \(G\), and do not treat a deterministic score as a
calibrated probability.

## 3. Critical common progress and joint loss laws

There is a particularly transparent structural special case. Let the rare
prototype pair for refinement \(r\) be

\[
x_{r\pm}=-\cos\vartheta\,u\pm\sin\vartheta\,v_r,
\qquad v_r\perp u,\quad \|v_r\|=1,\quad0<\vartheta<\pi/2.
\]

The \(v_r\) may be correlated. Use a symmetric audit weight on each pair and
its antisymmetric contrast. Assume strictly positive current readouts in this
special case, not in the general reserve theorem or experiments.

For a common-active unit, let
\(T_{jr}=\tan\vartheta\,|w_j^Tv_r|\). Its rare pair loses both gates exactly when
\(W_j(s)\ge T_{jr}\). If already blind its closing exposure is zero; otherwise

\[
s_{jr}=\log\frac{T_{jr}+\sqrt{T_{jr}^2+A_j^2-W_j^2}}{W_j+A_j}.
\tag{11}
\]

A common-inactive unit that currently responds has \(s_{jr}=\infty\).
A unit with \(W_j=0\) and \(w_j^Tv_r=0\) is initially blind, not a protected
route. Strictly positive readouts ensure that an open rare gate supplies a
nonzero derivative response in the contrast. As the kernel is a sum of Gram
blocks, its contrast enters the nullspace precisely when **all** such routes
close. Therefore

\[
s_r^*=\max_j s_{jr},\qquad C_r^*=F(s_r^*)-F(0).
\tag{12}
\]

This is an exact structural last-alternative law in the monotone sector. It is
not a theorem about crossing an arbitrary positive response threshold.

If \(C\) is random, structural losses are nested functions of the same random
common demand. For initially present contrasts, finite \(C\), and continuous
CDF \(G\),

\[
\Pr(r\text{ lost}\mid\theta)=1-G(C_r^*),\qquad
\mathbb E[N_{\rm lost}\mid\theta]=\sum_r[1-G(C_r^*)],
\tag{13}
\]

\[
\Pr(N_{\rm lost}\ge k\mid\theta)=1-G(C_{(k)}^*),
\tag{14}
\]

where \(C_{(k)}^*\) is the kth smallest critical gain. With atoms use
\(\Pr(C\ge C_r^*)=1-G(C_r^*-)\), respecting the gate convention. Infinite
critical gains contribute zero. For a random rare request independent of
\(C\), its loss probability is \(\sum_r\pi_r[1-G(C_r^*)]\); otherwise use
the request-conditional distribution of \(C\). These formulas explicitly
preserve correlation between refinements. Multiplying independent per-factor
survival probabilities would be unjustified.

### What width does, and what it does not prove

At fixed exposure and with old feature states held fixed, extra alternative
routes can only increase the maximum in (12). Protected routes can make it
infinite. Duplicate routes add no new closing tolerance. This is how width can
protect against complete closure through the competition dynamics themselves.

Natural widening also changes the \(1/m\) normalization, functional strength,
and the common progress map \(F\). It need not monotonically improve every
finite-strength reserve. Equation (12) is therefore not a universal monotonicity
theorem at matched training time or gain. The earlier
[independent Gaussian construction](probabilistic_competition.md) gives the
additional restricted law \(\Pr(\text{complete closure at }s)=b(s)^m\), with
\(b\) derived from the same dynamics. We do not assume independence of trained
features to import that law into arbitrary states.

The general state prediction is instead: width helps when it increases
directionally useful protected reserve or the tolerance of remaining exposed
routes. A count of units or a large current response alone does not establish
that condition. Initialization still affects the state; none of these results
claims that training forgets it.

## 4. A deliberately stronger, testable data assumption

The new tests use 24 rare factors with \(q_r\propto r^{-1.15}\), transferring
without refitting to 32. The total rare mass is either .04 or .15. Each factor
has a signed prototype pair as above. Within each of four groups,

\[
v_r=\sqrt{1-\rho}\,e_r+\sqrt\rho\,e_{g(r)},\qquad\rho=.5,
\]

so different refinements share context geometry. The frequent context is \(u\).
The continuous extension adds bounded scale variation, shared nuisance inputs,
and small attribute variation. Its head support is a known box, allowing a
population protection certificate rather than just checking observed samples.

Targets take the globally shared form
\(y=4[u^Tx]_++v^Tx\). This has an exact **three-unit** representation, also
on continuous examples. Widths 4, 8, and 16 therefore all have enough capacity.
This strong shared-refinement assumption isolates competition rather than
requiring one hidden feature per task. It describes a coherent common rule plus
many contexts sharing a linear correction. It is narrower than arbitrary
conditional refinements or the earlier co-occurring-factor task family; this is
not evidence of realism on language datasets or real data.

The audit is uniform over prototypes so that a rare contrast does not appear
weak merely because it receives less audit weight. Natural task probabilities
are reported separately using \(q_r/\sum q_r\). Audit probes remain prototypes
even in the continuous test, so population-wide correction capability is not
claimed.

## 5. What the tests establish

See the [complete evidence record](../evidence/competition_reserve/README.md).
The main cohort has 16 fresh seed banks, two initialization families, both task
variants and rare masses, three widths, and two phases: first half of the
remaining common residual is learned, then half of what remains. The second
phase starts at the actual trained state. No absorption is imposed on training.

The measured event is **losing more than half of current contrast response**,
not structural option extinction or a final-task failure. No rate, threshold,
or coefficient is fitted to the observed losses.

| State score ranking later half-response loss | 24 factors: AUC | 32 factors: AUC |
|---|---:|---:|
| Low current response | .689 | .704 |
| Low screened implementation participation | .777 | .702 |
| Low protected fraction | .907 | .841 |
| Low completion reserve fraction | .964 | .902 |

The completion rule has precision/recall .919/.795 and .908/.768, respectively.
It underestimates total event frequency: 2.89% predicted versus 3.34% observed
in confirmation; 2.91% versus 3.43% in transfer. These are event classifiers and
rankings, not calibrated stochastic forecasts. Pooled AUC is descriptive;
width/stage strata are retained, and completion is not the best score in every
transfer stratum.

Protected reserve is at least half current response on 71.4% of confirmation
probes. Under full training, 0.152% of that subset still loses more than half:
the rare-update qualification to the theorem is empirically necessary.
Transfer gives 59.7% protected probes and 0.177% violations within that subset.

Mean half-response-loss fractions fall from 6.09% to 3.89% to 0.049% across
widths 4, 8, 16 in confirmation; transfer gives 6.62%, 2.39%, 1.29%. A paired
seed-bank bootstrap gives a width-4 minus width-16 difference of 6.04 percentage
points [2.77, 9.62] and 5.32 [1.07, 10.03], respectively. These support a
conditional width benefit for response retention in this class, not its
mediation of final loss. The phases are matched by common functional progress.

Halving the step size changes the event classification on 0.043% of a matched
192-state audit. Mean absolute response difference is 0.0000484, but the worst
individual difference is 0.0276: boundary sensitivity has not vanished.

The main gain is a more explanatory state description and a progress-based
structural law. The remaining substantive gaps are broader common supports,
quantitative rare-update disturbance control, empirically grounded uncertainty
over common demand, and whether retained demanded response improves subsequent
correction loss. The older tests where common pressure creates responses and
closures recover remain evidence against a universal absorbing model.
