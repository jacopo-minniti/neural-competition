# Research objective

Develop a simple theory connecting functional correction options, competition
through shared trainable implementations, and the learning benefit of width.

An option is a demanded correction together with the response available to make
it. Capability modes alone do not specify options: their residual loading
determines the demanded corrections and their values. Closure means that response
vanishes while demand remains, rather than that the correction has been fitted.

## Maintained hypothesis

Tasks can contain common structures and less frequent refinements whose relative
residual demand changes during training. Learning the currently dominant
structure changes shared feature implementations. This can strengthen, weaken,
rotate, close, or create corrections that matter later. Width can help by
providing more alternative implementations and more opportunities for timely
recruitment or recovery.

Harmful competition requires all of the following:

1. learning pressure changes a useful response relative to a declared
   counterfactual;
2. residual demand loads onto the affected direction;
3. alternatives or recovery are inadequate within the correction budget;
4. the response deficit causes a measurable loss consequence.

Neither an option decrease nor a width gap alone establishes this chain.

## Current scope

- Primary setting: shallow ReLU networks on controlled heterogeneous-frequency
  tasks with sufficient narrow representational capacity.
- General task family: 24–32 power-law refinements, shared attributes, and
  associated continuous contexts; predict conditional on arbitrary current
  states rather than assuming a particular initialization law.
- Transfer setting: local controlled probes on MNIST; new MNIST research is
  paused.
- Primary target: describe current capability, pressure-relative protected
  reserve, and the amount of dominant learning that useful responses tolerate.
  Derive conditional loss probabilities and cumulative behavior from structural
  assumptions before attempting detailed trajectories. Short-horizon forecasts
  remain a numerical tool; whole-run success prediction remains later work.

## Evidence standards

Separate mathematical identities, descriptive trajectory evidence, causal
interventions, and prospective predictions. Preserve recovery cases, harmful
interventions, numerical counterexamples, and uncertainty. Compare widths on
common functional demands as well as their own residuals. Use explicit budgets
and verify step-size sensitivity when ReLU boundaries matter.
