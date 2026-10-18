# Evidence overview

Our [objective](../OBJECTIVE.md) is to identify a conditional, consequential
competition mechanism during training. The evidence currently supports strong
width effects in some settings and several local interaction effects. It does
not establish a general causal explanation of the width advantage.

Read these three reports in order:

1. [Signed residual growth](reports/residual_growth.md): the main research result.
   Opposing-output, overlapping teacher features produce persistent narrow-model
   failures under ordinary initialization. Width 32 beats width 8 in all 20
   paired starts for each of SGD, momentum and Adam. Narrow MSE is 2–5% of target
   variance, versus 0.0012–0.086% wide. Two simple sign predictors fail.
2. [MNIST](reports/mnist.md): width 16→128 improves test accuracy by about 3–5
   percentage points, while several interference quantities give inconsistent
   width trends. The residual-potential hypothesis has not yet been tested here.
3. [Research history](reports/research_history.md): the useful conclusions from
   older teacher, allocation, temporal-intervention and specialization studies.
   These constrain the current interpretation; failed ideas are not erased.

| Claim | Status |
| --- | --- |
| Extra width can materially improve fitting even above a known teacher width | Supported in the controlled teacher studies |
| Residual correlations govern existing neurons' signed amplitude growth | Derived for the stated gradient-flow assumptions |
| Opposing feature demands can suppress a correct output polarity | Derived locally; strongest training regime is consistent with role loss |
| One tested scalar sign predicts where width helps | Not supported by the current experiments |
| Width always reduces pair cost, batch conflict, or allocation inefficiency | Refuted by retained counterexamples |
| Role suppression explains the entire width gap, including MNIST | Unproven |

The [theory guide](theory/README.md) separates local identities, assumptions,
and unvalidated predictions. [Metrics](metrics.md) defines signs and denominators.
[Tables](tables/README.md) retain cohorts and uncertainty; [figures](figures/README.md)
contain only the two main studies. See [reproduction](reproduction.md) before
starting a long run and [retention](data.md) before deleting or interpreting data.
