# Task-structured option value: delayed modules and Hall reserve

23 October 2026. Second theory iteration. This note starts from task properties
rather than from a generic neural-state object. It supersedes the option Gramian
as the leading model for the *exclusive specialization* version of competition.
The Gramian remains appropriate when neurons can linearly superpose roles.

## 1. Why the previous object was underdetermined

A release-time viable probability \(p_j\) says how many neurons could still serve
role \(j\), but not whether the *same* neurons are the only candidates for several
roles. That overlap is exactly where competition for options lives. Two tasks can
have identical marginal role counts and radically different probabilities that
all roles can be filled.

The option Gramian supplies the missing overlap information only when a neuron
can contribute linearly to several roles at once. If training produces nearly
exclusive specialization, linear superposition is the wrong low-level model.
The task assumptions must decide between these objects.

## 2. High-level task class

The proposed scope is **delayed modular specialization**. It has four substantive
properties.

1. **Modular necessity.** Low risk requires covering several distinguishable
   task modules. A module can be a latent factor, mixture component, subgroup
   correction, class contrast, compositional branch, or teacher feature. It is
   not defined as an individual neuron.
2. **Representation-gated revelation.** Modules become learnable in a partial
   order. A child module has negligible or dominated feature-learning signal
   until prerequisite modules have been represented. This is stronger than
   merely assigning unequal target coefficients.
3. **Overlapping eligibility.** Before commitment, a neuron can be capable of
   learning several modules, and different neurons have different eligibility
   sets because of initialization and data geometry.
4. **Horizon-exclusive specialization.** Once a neuron commits sufficiently to
   one module, it cannot serve another required module before the evaluation
   horizon. Exclusivity can arise from attraction basins, gating, saturation,
   routing, normalization, or simply a switching time longer than the remaining
   training time.

These assumptions are restrictive on purpose. They describe a coherent family
of tasks on which "preserving options" has content. If superposition is cheap,
property 4 fails and a Gramian or low-rank approximation theory is more suitable.
If every module is visible immediately, property 2 fails and there is no release
order for early learning to exploit destructively.

## 3. Where the release order comes from

The release order should be a task/architecture property, not fitted from final
width outcomes. Let \(S\) denote the modules already represented. For module
\(j\), define its accessible residual utility

\[
U_j(S)
=\sup_{h\in\mathcal A_j(S),\ \|h\|\le1}
  -D\mathcal L(f_S)[h],
\tag{1}
\]

where \(\mathcal A_j(S)\) is the set of infinitesimal feature changes toward
module \(j\) that the architecture makes accessible near a representation of
\(S\). A task has representation-gated revelation when there is a directed
acyclic graph \(D\) such that

\[
U_j(S)\le \eta_j^-
\quad\text{if }\operatorname{pa}(j)\nsubseteq S,
\qquad
U_j(S)\ge \eta_j^+
\quad\text{if }\operatorname{pa}(j)\subseteq S,
\tag{2}
\]

with \(\eta_j^+\) separated from both \(\eta_j^-\) and the competing utilities.
Equation (2), not Gaussianity, is the central task assumption.

It is plausible in three important idealized task families:

- hierarchical sparse functions, where low-order or parent features expose new
  latent coordinates and interaction terms;
- common-plus-conditional mixture tasks, where fitting a common component leaves
  subgroup-specific residuals;
- symmetry-breaking specialization tasks, where an unspecialized population is
  stable until a task-dependent order parameter crosses a threshold.

[Merged-staircase](https://proceedings.mlr.press/v178/abbe22a.html) and
[saddle-to-saddle](https://proceedings.mlr.press/v195/abbe23a.html) results
provide existing settings in which feature support is learned sequentially.
[Committee-machine theory](https://proceedings.neurips.cc/paper_files/paper/2018/hash/84f0f20482cde7e5eacaf7364a643d33-Abstract.html)
and finite-width
[student-specialization results](https://proceedings.mlr.press/v119/tian20a.html)
supply settings with genuine specialization transitions. Neither line alone
gives the finite-width option-allocation theory below.

## 4. The task-induced option graph

At a release frontier, split each future module \(j\) into \(r_j\) required slots.
Construct a bipartite graph

\[
\mathcal G_t=(\text{uncommitted neurons},\text{future role slots},E_t),
\]

where neuron \(r\) is connected to module \(j\) if isolated post-release dynamics
from its current state can specialize it to \(j\) before the horizon. The edge
rule must be derived from the low-dimensional order-parameter dynamics of the
chosen task family.

For a set \(A\) of future modules, let \(\Gamma_t(A)\) be the neurons adjacent to
at least one slot in \(A\), and let \(r(A)=\sum_{j\in A}r_j\). Define the
**release-frontier Hall reserve**

\[
\boxed{
H_t=\min_{\varnothing\ne A\subseteq\mathcal R_t}
\left\{|\Gamma_t(A)|-r(A)\right\}.
}
\tag{3}
\]

Hall's theorem gives an exact statement under exclusive specialization:

- \(H_t\ge0\) if and only if all future role quotas can still be assigned;
- \(H_t=0\) means the representation remains possible but has no combinatorial
  slack at its most constrained bundle of roles;
- \(H_t<0\) certifies that at least \(-H_t\) required slots cannot be filled.

The minimizing set \(A_*\) is the **competition bottleneck**. This is already more
predictive than the rarest individual role: several individually well-covered
roles can fail jointly because they depend on the same small neuron pool.

## 5. The objective-valued object: reachable partition risk

Hall reserve detects whether every role can receive a separate specialist, but
it does not say how costly a failure is. Let \(C(B)\ge0\) be the prospectively
derived excess risk when one specialist must serve a block \(B\) of task roles,
with \(C(\varnothing)=C(\{j\})=0\). The task distribution, not the endpoint,
must determine \(C\). Section 9.2 derives it exactly for conditional regimes.

Let \(\mathcal A_t\) be the set of role-to-neuron assignments still reachable
from time \(t\). An assignment may send several roles to one neuron, representing
a forced merge; its block must be compatible with that neuron's remaining
routing basin. Define the **reachable partition risk**

\[
\boxed{
D_t^{\mathrm{part}}
=\min_{a\in\mathcal A_t}
\sum_{r} C\!\left(a^{-1}(r)\right).
}
\tag{4}
\]

When every non-singleton block has positive cost, \(D_t^{\mathrm{part}}=0\) if
and only if the option graph has a complete matching, equivalently \(H_t\ge0\).
When failure consists of missing rather than merging a role, add a dummy failure
expert and let \(C(B)\) be the sum of the missing-slot costs. The earlier
minimum-cost maximum-matching deficit is this additive special case.

Under an exact conditional-risk decomposition, (4) is the terminal profiled loss
floor. Under a moment certificate it is a certified component of that floor.
Adding neurons or preserving reachability edges cannot increase (4). Early
commitment can: it removes assignments from \(\mathcal A_t\). Competition damage
is the increase in (4) caused by actual early specialization relative to a
specified counterfactual allocation policy, not merely a high value of (4) at
the endpoint.

## 6. Width law from the task's eligibility distribution

Let an ordinarily initialized neuron have a random eligibility set
\(S_r\subseteq\mathcal R\), independently across neurons in the reference model.
For every role subset \(A\), define the **union viability mass**

\[
p(A)=\Pr\{S_r\cap A\ne\varnothing\}.
\tag{5}
\]

Then

\[
|\Gamma(A)|\sim\operatorname{Binomial}(m,p(A)).
\]

Consequently the failure probability obeys the prospective bounds

\[
\max_A
\Pr\{\operatorname{Bin}(m,p(A))<r(A)\}
\le
\Pr\{H_t<0\}
\le
\sum_A
\Pr\{\operatorname{Bin}(m,p(A))<r(A)\}.
\tag{6}
\]

The leading width scale is therefore controlled by

\[
\boxed{
m_*\approx
\max_{A\subseteq\mathcal R}
\frac{r(A)}{p(A)},
}
\tag{7}
\]

with concentration and multiple-subset corrections. Marginal viable masses
\(p_j=p(\{j\})\) are insufficient; the complete union-mass set function \(p(A)\)
is the task-level object. It describes whether future roles draw on separate,
overlapping, or identical neuron pools.

A standardized finite-width reserve useful near threshold is

\[
Z_m=min_A
\frac{m p(A)-r(A)}{\sqrt{m p(A)(1-p(A))}}.
\tag{8}
\]

Large positive \(Z_m\) predicts robust coverage; negative \(Z_m\) identifies a
likely bottleneck. Unlike a generic reachability metric, (7)--(8) are determined
by the task's module demands and eligibility geometry.

## 7. Dynamics as online matching

The release DAG turns the static graph into an online allocation problem. Early
roles arrive first, ordinary gradient dynamics commits neurons without knowing
or respecting later scarcity, and those assignments delete options for future
roles. This suggests two distinct quantities:

1. **Initial option shortage:** (4) is already positive before training.
2. **Allocation regret:** (4) is initially zero but becomes positive because
   early commitments consume neurons belonging to a future bottleneck set.

The second is the clean form of the proposed competition phenomenon. Width helps
because positive Hall reserve makes a locally greedy assignment less likely to
destroy every global matching.

It also generates stronger predictions than "rare roles need width":

- an early role can be individually easy yet harmful if it consumes generalists
  that are the only candidates for a later bundle;
- two tasks with identical per-role counts can have different width thresholds
  because their eligibility overlaps differ;
- releasing constrained roles earlier, or reserving high-flexibility neurons,
  can rescue a minimal-width network;
- extra plasticity can paradoxically worsen a greedy narrow network if it makes
  scarce generalists more attractive to early roles, even though extra edges
  always help the offline matching;
- the relevant intervention protects the bottleneck neighborhood, not a fixed
  output sign.

## 8. Exploratory matching experiments

The executable experiment is
[`explore_option_matching.py`](../../scripts/studies/explore_option_matching.py).
It tests the combinatorial deductions, not the neural assumptions.

### 8.1 Marginal viable mass is not enough

Four roles each have marginal viability \(p_j=.25\) in three initialization
families:

1. every neuron is a specialist for exactly one role;
2. viable neurons can serve one of two paired roles, with half the neurons dead;
3. one quarter of neurons can serve any role and the rest are dead.

The worst-subset ratios \(r(A)/p(A)\) are respectively 4, 8 and 16. At width 8,
the simulated probabilities of complete coverage are:

| Eligibility structure | Complete coverage | Mean missing slots |
| --- | ---: | ---: |
| Separate specialists | .623 | .402 |
| Paired flexible pools | .356 | .928 |
| One shared flexible pool | .113 | 2.036 |

Any theory using only the four identical marginal \(p_j\)'s predicts no
difference. Hall bottlenecks give the correct ordering and computable lower and
upper failure bounds.

### 8.2 Early commitment can destroy an available solution

The dynamic toy task has one early role and two late roles. Neurons are early-only,
late-1-only, late-2-only, or flexible. At minimal width 3:

- 50.8% of initial option graphs possess a complete offline matching;
- random allocation to the early role succeeds only 36.0% of the time;
- conditional on an available initial solution, random early commitment destroys
  it in 29.2% of cases;
- assigning the least flexible eligible neuron to the early role achieves the
  full 50.8% offline-feasible rate.

At width 12, random commitment destroys only .14% of initially feasible cases.
This is an exact small-model realization of "width buys option slack": capacity
is present initially, early conversion loses it, and redundant width makes the
same greedy allocation policy safer.

## 9. Candidate natural instantiations

### 9.1 Branching compositional functions

A target such as a parent feature followed by several interactions sharing that
parent has a task-level release DAG. Merged-staircase theory makes sequential
support discovery plausible in Boolean or Gaussian latent-variable models. The
new requirement is a branching staircase: after learning a shared prefix,
particles must divide among several children. The option theory predicts that
width is controlled by the least-covered subtree, not merely hierarchy depth.

### 9.2 Common plus subgroup-specific prediction

Assume examples come from latent subgroups. A common feature explains much of
every group's response; each subgroup also has a necessary conditional feature.
The common feature has early global signal, while subgroup features become
dominant in the residual later. A neuron near several subgroup basins is a
generalist; early capture by the common feature can consume the only options for
rare groups. The bottleneck mass \(p(A)\) is then determined by unions of subgroup
basins and can be related to mixture weights and separation.

This family supplies a stronger task-level theorem. Let \(Z\in\{1,\ldots,K\}\)
be a regime identifiable from routing covariates \(U\), with mass \(\pi_j\). Let
\(S\) be the within-regime content variable and, under the simplifying assumption
that it has a common law across regimes, define the conditional Bayes functions

\[
h_j(s)=\mathbb E[Y\mid S=s,Z=j]\in L_2(P_S).
\]

If a hard or effectively exclusive router forces a block \(B\) of regimes to
share one unrestricted expert, its best predictor is

\[
\bar h_B=\frac{\sum_{j\in B}\pi_jh_j}{\pi(B)},
\qquad \pi(B)=\sum_{j\in B}\pi_j,
\]

and its exact squared excess risk is

\[
\boxed{
C(B)=\frac12\sum_{j\in B}\pi_j\|h_j-\bar h_B\|_2^2
=\frac{1}{4\pi(B)}
\sum_{j,k\in B}\pi_j\pi_k\|h_j-h_k\|_2^2.
}
\tag{9}
\]

Thus the data distribution itself supplies a **regime-incompatibility geometry**
\(d_{jk}=\|h_j-h_k\|_2^2\). If \(d_{jk}=0\), those regimes do not require
separate specialists. If it is large, merging them has an unavoidable, known
cost. For any routing partition \(\mathcal P\), the exact profiled loss floor is
\(\sum_{B\in\mathcal P}C(B)\). Cross-entropy has an analogous conditional
Bregman-information cost rather than an \(L_2\) variance.

This is the desired assumptions/object coupling:

- **task assumption:** identifiable conditional regimes with incompatible Bayes
  rules;
- **task-derived loss object:** the block costs \(C(B)\);
- **architecture assumption:** competitive routing makes specialization
  effectively exclusive;
- **finite-width object:** Hall reserve determines which low-cost routing
  partitions remain reachable.

The remaining dynamic assumption is that regime imbalance creates an ordered
residual drive before the router has settled. That part can be studied directly
in the closed population router ODE below.

### 9.3 Symmetry-breaking committee models

Teacher/student committee machines and mixture-of-experts models naturally have
exchangeable unspecialized phases followed by specialization. Unequal component
strengths give a release order. Order-parameter ODEs can potentially derive the
eligibility sets and commitment times. This is less teacher-free, but probably
the fastest route to a complete low-level theorem.

### 9.4 Low-level screening changed the preferred first model

Two small population experiments tested whether the high-level properties arise
rather than assuming that they do.

**Branching Boolean ReLU target: negative.** A standard two-layer ReLU network
was trained on the population of ten-dimensional Rademacher inputs with target

\[
y=2x_1+x_1x_2+0.6x_1x_3.
\]

Widths 6, 12 and 24 all reached very small loss across ten SGD and ten Adam
starts; width 6 was generally faster under the tested scaling. The hierarchical
coefficients did not create horizon-exclusive specialization. This rejects a
plain branching staircase as the first model: a release DAG alone is
insufficient.

**Imbalanced mixture with competitive routing: promising.** Consider three
separated regimes with locations \(\mu=(-2,0,2)\), masses
\(\pi=(.7,.2,.1)\), and incompatible scalar regression slopes
\(\beta=(1,-1,2)\). A softmax-gated expert model has trainable centers \(c_r\),
expert slopes \(a_r\), and population predictions

\[
g_{jr}
=\frac{\exp\{-(\mu_j-c_r)^2/(2\tau^2)\}}
       {\sum_s\exp\{-(\mu_j-c_s)^2/(2\tau^2)\}},
\qquad
\widehat\beta_j=\sum_r g_{jr}a_r,
\tag{10}
\]

with exact loss

\[
\mathcal L=\frac12\sum_j\pi_j(\widehat\beta_j-\beta_j)^2.
\tag{11}
\]

For ordinary gradient flow, this model has the closed population dynamics

\[
\dot a_r
=\sum_j\pi_j(\beta_j-\widehat\beta_j)g_{jr},
\qquad
\dot c_r
=\frac1{\tau^2}\sum_j
\pi_j(\beta_j-\widehat\beta_j)g_{jr}
(a_r-\widehat\beta_j)(\mu_j-c_r).
\tag{12}
\]

The component-to-expert residual drive
\(\pi_j(\beta_j-\widehat\beta_j)g_{jr}\) is therefore explicit. Mixture mass,
target incompatibility, routing temperature and the current residual jointly
determine release and capture; no unknown teacher-neuron assignment is needed.
The analytical task is to identify invariant regions and basin boundaries of
(12), then compute their initialization measure.

Width 3 can represent all three slopes exactly. In 50 paired Adam starts with
\(\tau=.7\), trainable routing failed above loss threshold \(10^{-3}\) in 20%,
6% and 0% of widths 3, 4 and 6. The release-frontier matching deficit was positive
in exactly the same fractions under the provisional eligibility diagnostic
described below. In a smaller optimizer screen, width-3 failure occurred in 2/30
starts for SGD, 2/30 for momentum and 5/30 for Adam, while widths 4 and 6 solved
all 30 under each tested update rule. These rates are exploratory and the
step/rate choices are not performance-matched.

A decisive paired trajectory starts at width 3 with center locations
\((-2.329,-1.066,1.365)\), zero Hall deficit and router minimum singular value
.511. Freezing this router reaches numerical zero loss. When the centers train,
the dominant component initially has weighted residual .665 versus .208/.203.
By step 30 its residual has fallen to .128 and the delayed components have
.245/.180, but the centers have already moved to
\((-3.151,-1.856,.609)\): Hall deficit is one and the router minimum singular
value is .0076. The run converges to loss .300.

That plateau is task-derived. If the middle and right regimes must share one
expert, their optimal common slope is

\[
\bar\beta
=\frac{.2(-1)+.1(2)}{.2+.1}=0,
\]

and their irreducible merged loss is

\[
\frac12\{.2(-1-0)^2+.1(2-0)^2\}=.300.
\]

More generally, if routing collapse forces a component block \(B\) to share one
scalar expert, its exact excess risk is

\[
L(B)=\frac12\left[
\sum_{j\in B}\pi_j\beta_j^2
-\frac{(\sum_{j\in B}\pi_j\beta_j)^2}{\sum_{j\in B}\pi_j}
\right].
\tag{13}
\]

Thus this model couples all required levels: mixture imbalance creates a release
order, routing creates exclusive specialization, Hall reserve detects option
loss, and (13) predicts the resulting loss rather than merely classifying a run
as failed. The executable probe is
[`explore_mixture_routing.py`](../../scripts/studies/explore_mixture_routing.py).

The strongest theory-directed regime makes dynamic destruction, rather than
initial coverage, dominant. With \(\pi=(.85,.1,.05)\), \(\tau=1\), and router
learning rate three times the expert-rate, a vectorized 500-seed screen gave:

| Width | Initial Hall failure | New Hall collapse | Final Hall/loss failure |
| ---: | ---: | ---: | ---: |
| 3 | 3.2% | 20.2% | 22.2% |
| 4 | 1.0% | 5.6% | 6.0% |
| 6 | 0% | 0% | 0% |

Equalizing the initial component drives to
\(\pi_j|\beta_j|=(.4,.4,.4)\), while keeping the same router dynamics, reduces
width-3 new collapse from 20.2% to 5.6% and final failure to 7.0%. This is a
prospective structural prediction: stronger early imbalance consumes future
routing options, whereas redundant width or balanced release pressure protects
them. It also separates event probability from objective damage: making a
component rarer can make it more likely to be abandoned while reducing the loss
cost in (13).

For this experiment an expert is provisionally connected to component \(j\) when
its router weight is at least .2 times the strongest router weight for that
component. The relative threshold avoids mechanically deleting edges as width
increases, but it is still a diagnostic choice. A complete theory must replace
it with a reachability statement derived from the center ODE. The exact agreement
between this proxy and loss failures is evidence that such a basin description
may exist, not proof that the chosen edge definition is canonical.

## 10. What should be derived next

The screening changes the order of attack. The most promising first target is
now the **imbalanced conditional-mixture model with competitive routing**:

1. characterize all specialized and collapsed stationary partitions of
   (10)--(12) and compute their losses from (13);
2. derive which initialization regions flow to each partition as a function of
   mixture masses, target incompatibility and router temperature;
3. identify the time at which each component becomes relatively dominant and
   prove when its Hall edge is lost before that release;
4. integrate the basin measure to predict failure probability at each width,
   before simulating those widths;
5. test the prediction that balancing mixture masses, increasing router
   temperature, or reserving a generalist until the delayed release removes the
   redundant-width advantage;
6. only after closing this model, return to a branching compositional network
   with an architecture known to produce exclusive child specialization.

## 11. Current judgment

The central high-level insight is that the task should supply a **release order of
necessary conditional modules**, while the architecture supplies **exclusive,
overlapping eligibility sets**. Under those assumptions, the single
objective-valued object is the reachable partition risk (4); the
release-frontier Hall reserve (3) is its zero-loss feasibility certificate and
drives the width law. The mixture-routing model is currently the first setting
in this repository where those assumptions arise together and the partition
deficit predicts an exact nonzero loss.

This theory is narrower than the generic future-option deficit, but much more
predictive. It says which bundle of roles is endangered, gives a width threshold
from the task-derived union masses, distinguishes initial shortage from damage
caused by early training, and specifies interventions that should erase the width
advantage. Its main unresolved step is now concrete: derive the option graph and
release DAG from one genuine neural population dynamics rather than assuming
them.
