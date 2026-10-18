# Read-only checks prompted by the literature

The separate [objective-scale audit](effect_magnitude/README.md) analyzes
existing full-cohort tables for practical magnitude and early predictive
value. It performs no optimizer steps and does not replace the saved-state
check described below.

This page describes the saved-state readout/gate check. The separate
[strict-interference follow-up](strict_matched_loss/README.md) replays original
trajectories to recover frequency and harm at matched loss; its provenance
and scope differ from this read-only state evaluation.

These are **new diagnostics of existing checkpoints**, performed on 18 October
2026. No optimizer step, training continuation, dataset change or modification
of the retained archive was performed. They are separate from the original
full-cohort tables and their published uncertainty estimates.

## Inputs and outputs

The inputs are the exact teacher-8/2027 dataset and retained robustness
checkpoints for seeds **0 and 3**. Settings are momentum .1, Adam .001/.01,
and Adam .001 with outgoing initialization std .1/1. States are initialization,
10k, 50k and 100k updates. The core settings include widths 8/10/16/32;
the initialization controls include 8/16/32. This yields **144 states**.

These seeds were purposively retained before this literature search. They are
illustrative cases, not a random sample and not the five-seed cohort. Final
checkpoint errors here are also different estimands from the original report's
mean over evaluations 91k–100k. Do not infer cohort rates from these outputs.

- [saved_state_checks.csv](saved_state_checks.csv) contains every state-level result.
- [saved_state_checks.json](saved_state_checks.json) records input hashes,
  script hash, versions, precision and numerical checks.
- [reanalyze_saved_states.py](reanalyze_saved_states.py) is the complete producer.

Reproduce from the repository root, using the existing environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/theory/checks/reanalyze_saved_states.py
```

This rewrites only the two local output files above. It requires the ignored
archive under `runs/retained/`; the versioned outputs and interpretation remain
readable without it. See [data retention](../../data.md) for the archive scope.

## Computation and meaning

All saved float32 parameters and data are evaluated in float64. Gate coverage
counts use w_r^T x_i+b_r>0 over the full training set. Always-off, always-on,
below-1% and above-99% activity are recorded separately. Kink locations are
−b_r/||w_r||. Output coefficient signs do not by themselves imply active
positive or negative contributions.

For each state we solve min_a ||Phi a−y||² by NumPy SVD least squares, with
Phi=ReLU(XW^T+b)/sqrt(m), rcond=1e-12, and **no output bias**. Incoming weights
and biases are fixed, outgoing coefficients are unrestricted, and the fit
uses training targets only. Test MSE evaluates that readout without using
test labels to choose it. This is an oracle diagnostic, not a trained model
or a claim about what an optimizer will discover.

The producer checks that this optimum does not worsen training MSE and that
the normalized normal-equation residual is below 1e-8 in every state. The
observed maximum is **2.50e-15**. Source hashes make the selected inputs explicit.
The target-correlation columns are E[y Phi_r]; at zero output they equal
minus the outgoing gradient. At nonzero output they are not minus that
gradient, which depends on the current residual.

Separate deterministic algebra checks confirmed the pair-subdivision identity,
the residual-based loss budget, invariance of the local allocation diagnostic
under exact subdivision, and the canceling-pair gradient formula against
autograd. Maximum numerical discrepancy was below 7e-18. These sanity checks
verify the calculations; they do not test the proposed training mechanisms.

## Findings that affected the notes

- At zero-output initialization every target-feature correlation is negative
  in both saved seeds at every retained core width. Both failed and successful
  examples share that property.
- The two final width-8 Adam-.001 failures have all gates active on substantial
  subsets; neither has an always-off or always-on neuron. An unrestricted
  readout refit leaves MSE near .104. Those cases require a different hidden
  representation to attain a good fit.
- A final width-32 Adam-.01 success has 23 always-off neurons. Its nominal
  width and its final dead-neuron count alone do not describe the useful
  features its trajectory acquired.

These constrain hypotheses in [the Adam note](../adam_initialization_and_features.md).
They do not show that a failed state is a local minimum, identify its whole
training history, or establish a width effect in a larger population.
