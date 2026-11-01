# Research objectives

## Question and primary framework

Explain when competition between learned corrections produces a finite-budget
loss penalty, and when and why greater width prevents or repairs that penalty.
The [core theory](report.tex) distinguishes response capability from residual
demand: an option is a nonzero demanded correction together with the response
available to make it. Exact zero response with demand remaining is closure;
weak response and satisfying demand are different events. A fixed contrast is a
probe of capability, not automatically an option or a count of spectral modes.

The primary extension is [protected reserve and tolerable dominant learning](docs/model/competition_reserve.md).
It distinguishes current response from a pressure-relative guarantee and models
response under a specified dominant-learning trajectory. Current results are
conditional on the architecture, optimizer, audit and pressure assumptions in
that chapter. The coherent-pressure completion curve is an approximation under
full training; an exact common-only bound does not automatically survive rare
updates. Neither current capability nor reserve alone establishes loss harm.

The current extension follows a [controlled assumption-relaxation ladder](docs/model/reserve_relaxation.md)
from this successful reserve model. Its [first bridge](docs/evidence/reserve_bridge/README.md)
tests co-occurring factors while retaining the old prediction and changing
premises separately. The [direct parity exploration](archive/docs/evidence/frequency_pressure/README.md)
is diagnostic evidence, not the primary framework. Accurate simulation of future
SGD is outside the current target; probabilistic and approximate option
statistics, width, and consequential loss are the priority.

The older [conditional forecasting model](archive/docs/model/power_law_competition.md)
and [Gaussian pressure law](archive/docs/model/probabilistic_competition.md) are secondary:
they supply useful techniques, special-case laws and counterexamples. Preserve
their evidence and limitations, including recovery and response creation.

## Research goals

1. **Options:** derive or forecast opening, weakening, closure, recovery and
   useful lifetimes, including conditional probabilities and expected numbers.
   Separate structural zero from finite-threshold and relative-change events.
2. **Loss:** relate demand-weighted spectral response and its evolution to
   finite-horizon loss. State when local approximations can be used for more than
   a step, measure their error, and test whether altered response causes a loss
   difference rather than merely accompanying it.
3. **Width:** identify when width changes initial coverage, survival, diversity
   and recovery of useful responses, and quantify their contribution to loss.
   Width-dependent exponents are interesting when their assumptions and error
   are justified; independent alternatives are not presumed under shared updates.

The desired concrete result is a prospective narrow-versus-wide learning plot:
mark predicted and observed weakening or closure of demanded responses, show the
associated loss forecast and observed loss, and test the proposed causal chain.
The [existing selected three-prototype illustration](docs/evidence/competition_reserve/learning_curve_example_20261030/README.md)
is a starting example, not validation of that general claim.

We do not seek scaling laws or universal width exponents at this stage. See the
[precise quantities, baselines and plots](docs/research_agenda.md).

## Assumptions and intended dataset

Strong assumptions may make a derivation possible, but our priority is for the
assumptions that carry a proposed prediction to be measurable and at least
approximately plausible in the [intended frequency-heterogeneous toy dataset](docs/dataset_design.md).
An exact special case is valuable as a benchmark, not evidence that its premises
hold in that dataset. Document which assumptions are exact, measured
approximations, or violated; quantify their errors and retain failure cases.
In particular, do not assume a single dominant direction, stable gates,
protected units, negligible non-dominant updates, independent alternatives, or
frozen local response merely because a derivation uses them.

The dataset should have recurring components with a broad frequency spectrum,
co-occurrence, context variation and compositional interactions. No factor is
inherently common or rare; these are relative descriptions. It should be a controlled idealization of properties
that can appear in language, images and other data—not a claim that tokens,
patches or latent skills are interchangeable. Run designed ablations and compare
natural-frequency and balanced evaluation. Establish representational
sufficiency separately from successful training.

## Evidence standards

Separate mathematical identities and conditional theorems from approximations,
prospective forecasts, descriptive outcomes and causal tests. Predeclare events,
thresholds, budgets, matched comparisons and primary baselines; save forecasts
before outcomes. Preserve null and adverse cases, recovery, uncertainty and
numerical boundary sensitivity. Treat shared factors and repeated checkpoints as
clustered; replicate over independent dataset/seed banks. Compare widths at both
matched steps and matched functional progress, and distinguish their own residuals
from fixed common reference demands. Report when an assumption is vacuous or a
prediction is not applicable. See [reproduction](docs/reproduction.md).
