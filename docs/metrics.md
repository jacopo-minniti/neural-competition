# Objects, signs, and units

The main object is the [signed residual potential](theory/residual_growth.md),
H_f(u)=E[(y−f(x)) ReLU(uᵀx)]. It is the loss derivative with respect to adding
signed feature mass, with sign reversed. Under the stated gradient-flow
assumptions, sign(c) H_f(u)<0 shrinks a neuron's current signed amplitude.
Contraction is not automatically harmful; teacher knowledge identifies useful
roles in the controlled task. The sign is not a validated global width predictor.

## Loss scales

- Residual-growth/polarity targets have unit training variance. Stored loss is
  **half-MSE**. Multiply by 200 for ordinary MSE as percent of target variance.
  Predicting the training mean gives test MSE 1.00055 in the main opposing task.
- Earlier teacher studies mix ordinary MSE fitting columns with half-MSE
  objective budgets. Consult column definitions and historical source before
  comparing them with current results.
- MNIST squared loss is half the **sum** of squared errors across ten logits,
  averaged over examples. It is not averaged over classes or computed after
  softmax. Cross-entropy is mean negative log probability. Accuracy is separate.

A 96% reduction of an error equal to 2.33% of target variance recovers about
2.24 percentage points of variance, not 96% of the original prediction problem.
The baseline, absolute difference, and remaining-error ratio answer different
questions. Squaring also matters: MSE of .0233 corresponds to RMSE of .153
in target-standard-deviation units.

## Historical diagnostics still used in the MNIST report

| Quantity | Meaning and sign | Limitation |
| --- | --- | --- |
| Batch transfer | Gradient dotted with a proposed descent direction; positive helps to first order | Cross-data conflict is not automatically neuron-to-neuron interference |
| Standalone change d_r | Loss after updating only neuron r minus original loss | Negative helps; depends on step size and evaluation data |
| G / H | Sum of −d_r for helpful neurons / d_r for harmful neurons | Larger terms are not causal contributions to a final width gap |
| Pair interaction J_rs | Joint loss change minus both standalone changes | Positive is an unfavorable accounting cost; can be benign diminishing returns |
| Strict event | Both updates help alone; one becomes harmful after the other | Frequency per eligible pair differs from total count and total harm |
| C_plus / C_minus | Accumulated positive pair costs / favorable credits | Wider networks can fit better while accumulating more C_plus |
| Allocation benefit | Extra held-out gain over the best tested uniform shrink | Per-helpful-work units are not percentages of total loss or future rescue |

For squared loss and additive neuron outputs, the complete finite-update budget
is ΔL=H−G+ΣJ. For cross-entropy, a sum of pair terms need not account for all
higher-order interactions; the total remainder is measured separately.
An identity describing a trajectory is not an intervention holding the other
terms fixed. Momentum and Adam's proposed directions depend on optimizer state.

## Uncertainty and controls

Use paired training seeds as the replication unit, not neurons, pairs, batches,
or repeated checkpoints. The 20-start residual-growth cohort includes the first
five starts; 100k runs extend those same starts. Missing matched-loss states
remain missing. Intervals are exploratory and unadjusted for multiple testing.
Control for training phase, step size, initialization, output scaling and
representational capacity before assigning a width effect to competition.
