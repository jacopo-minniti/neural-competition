# Maintained options model

The model has one functional capability object, one demand object, and an
explicit local dynamics.

The current [reserve model](competition_reserve.md) asks which responses are
protected from dominant learning, and how much common learning exposed responses
can tolerate. Its state certificate needs no initialization distribution or
coherent-pressure assumption. A stronger coherent-pressure model derives
completion reserves and joint structural loss laws indexed by functional
progress. Tests cover 24–32 power-law refinements and bounded continuous context
variation. The earlier [conditional forecasting model](power_law_competition.md)
remains a numerical tool on a broader co-occurring-factor family; its forecast
accuracy alone does not establish an explanatory state theory.

## 1. Capability

At a network state, \(C_\tau\) measures exponentially averaged local response,
while \(B_t\) or \(B_{h,\eta}\) uses a fixed continuous or discrete horizon.
Their eigenspaces describe capability modes. Rank, effective breadth, and
response mass summarize capability independently of current demand.

## 2. Demand

For a declared residual \(e\), an **option** is a nonzero spectral correction
\(q_E=\Pi_Ee\), together with the response available in eigenspace \(E\).
Its value is \(\beta_E\|q_E\|^2/(2n)\), and summing gives the unchanged
local benefit \(V(e)=e^TBe/(2n)\), with \(B\) the chosen benefit operator.
Demand remaining with zero response means a closed option; zero demand means
there is no current option in that eigenspace. Use eigenspaces to avoid choosing
an arbitrary basis within a repeated eigenvalue. See the
[precise definition](spectral_options.md#3-capability-modes-and-demanded-options).

Raw value, normalized directional strength, residual magnitude, and actual loss
must be reported separately. A fixed arbitrary direction is a capability probe;
it represents an option only when loaded by declared demand. Reference residuals
may be held fixed to compare capability across checkpoints. Historical filenames,
result keys such as `option_mass`, and evidence snapshots retain their original
names; `option_mass` measures response mass, not a residual-dependent option count.

## 3. Dynamics

For

\[
f_\theta(x)=m^{-1/2}\sum_j a_j[z_j(x)]_+ + c,
\]

the local model freezes gates only within short intervals and evolves both
layers under the original loss and optimizer:

\[
\widehat f(\delta,x)=f_0(x)+J_0(x)\delta+
m^{-1/2}\sum_j\delta a_jG_j(x)\delta z_j(x).
\]

The last term is the minimal coefficient–feature feedback required to change
responses. Physical ReLU gates are used when measuring forecast capability.

## 4. Competition

Task pressures add in the parameter update, but the same feature blocks can
support several functional directions. Training one component therefore
transports the complete response operator. Harmful competition is a negative,
demand-loaded, finite-budget consequence—not merely a decrease in one spectral
summary.

## 5. Width

Natural widening supplies differently initialized feature blocks. It can improve
initial coverage, preserve alternatives, or enable recruitment. Identical
duplication does not reproduce this. Width protection is conditional on task
structure, optimizer scaling, implementation overlap, and recovery time.

The [probabilistic extension](probabilistic_competition.md) solves one coherent
pressure regime explicitly. The probability of complete loss of a fixed
refinement response is b(s)^m, with b(s) derived from Gaussian initialization
and the feature flow. Joint laws separate initial absence, later closure, and
reopening. These are exact at fixed accumulated pressure; natural training at a
fixed time additionally requires control of the pressure clock and other forces.

## Technical chapters

1. [Spectral derivation](spectral_options.md)
2. [Local dynamics](local_dynamics.md)
3. [Finite-budget criterion](finite_budget_competition.md)
4. [Shared implementations](shared_implementation_competition.md)
5. [Delayed demand and width](delayed_demand_and_width.md)
6. [Observed learning ledger](learning_ledger.md)
7. [Probabilistic competition and width](probabilistic_competition.md)
8. [Conditional probabilities on power-law refinement tasks](power_law_competition.md)
9. [Protected reserve and tolerable common learning](competition_reserve.md)
