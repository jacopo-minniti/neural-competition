# Maintained options model

The model has one functional capability object, one demand object, and an
explicit local dynamics.

## 1. Capability

At a network state, (B_\tau) or (B_{h,\eta}) measures which output-space
corrections gradient descent can make over a declared horizon. Options are its
positive eigenspaces. Rank, effective breadth, and mass are summaries; they are
not literal neuron counts.

## 2. Demand

Residual demand (e) determines which available directions currently matter.
Raw value (e^TBe), normalized directional strength, residual magnitude, and
actual loss must be reported separately.

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

## Technical chapters

1. [Spectral derivation](spectral_options.md)
2. [Local dynamics](local_dynamics.md)
3. [Finite-budget criterion](finite_budget_competition.md)
4. [Shared implementations](shared_implementation_competition.md)
5. [Delayed demand and width](delayed_demand_and_width.md)
6. [Observed learning ledger](learning_ledger.md)
