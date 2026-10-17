# Findings from the eight-neuron teacher

## What we tested

All students can represent the same noiseless teacher. Both layers train in every
run. We first compared K, K+1, K+2, 2K, 8K and 16K students for 20,000 updates
with two SGD rates and two momentum rates (five seeds). We then extended SGD at
learning rate 1 to 100,000 updates and added initializations at the smaller widths.
The expansion was stopped at the user's request: widths 8, 9 and 10 have 20
complete runs, width 16 has eight, and widths 64 and 128 have five each.

All losses in this document are **ordinary test MSE**, twice the half-MSE in the
raw training logs. Results use one fixed teacher and one fixed training/test draw.

## Completed 100,000-update results

"Good fit" means test MSE below 1e-6. An unsuccessful run is not automatically a
proven local minimum.

| Student width | Completed seeds | Mean test MSE | Median test MSE | Good fits |
|---|---:|---:|---:|---:|
| 8 | 20 | 2.379e-03 | 2.850e-03 | 6/20 |
| 9 | 20 | 1.522e-03 | 6.576e-05 | 8/20 |
| 10 | 20 | 2.444e-04 | 1.309e-08 | 18/20 |
| 16 | 8 | 2.535e-08 | 2.844e-08 | 8/8 |
| 64 | 5 | 1.086e-07 | 7.731e-08 | 5/5 |
| 128 | 5 | 2.853e-06 | 1.529e-06 | 2/5 |

Two extra neurons raise the observed good-fit fraction from 6/20 to 18/20.
One extra neuron helps less reliably. Widths 16 and 64 did well on their smaller
seed sets. Width 128 was still refining its solution after 100,000 updates.
The mean at width 10 is much larger than its median because two runs fail.
Extra width can improve reliability while very large widths can learn slowly
under the unchanged learning rates and width normalization.

## Why the training budget matters

To avoid mixing different seed sets, this table uses the same original seeds
0–4 at every width. Entries are median test MSE after the stated update count.

| Width | 2,000 updates | 20,000 updates | 100,000 updates |
|---|---:|---:|---:|
| 8 | 5.489e-03 | 4.870e-03 | 4.894e-03 |
| 9 | 5.426e-03 | 3.382e-03 | 2.876e-03 |
| 10 | 5.097e-03 | 1.201e-04 | 2.621e-08 |
| 16 | 1.702e-03 | 8.266e-06 | 2.771e-08 |
| 64 | 2.859e-03 | 3.919e-05 | 7.731e-08 |
| 128 | 4.953e-03 | 8.353e-05 | 1.529e-06 |

At 20,000 updates, the width benefit is clearest for SGD at learning rate 1 and
momentum SGD at rate 0.1 with beta 0.9. The lower rates are less advanced. The
small historical table in `teacher8_history.csv` retains the full rate comparison.

## Cases retained for mechanism analysis

- Seed 0: width 8 finishes at 5.295e-3; width 10 reaches 1.013e-8. Seed 1 is
  another narrow failure / wider success.
- Seed 3: width 8 itself reaches 3.595e-13. This is a successful narrow control.
- Seed 5: width 8 reaches 5.796e-7, while width 10 finishes at 2.627e-3.
  This paired counterexample prevents treating extra neurons as a guarantee.
- Width 9 exposes mild overparameterization; width 16 is a stronger positive
  case; width 128 illustrates slow refinement at a large width ratio.

The retained seeds were selected for contrasting behavior. **Do not estimate
success probabilities from this curated subset.** Use all completed runs in
`teacher8_history.csv` for the descriptive rates above. Width 64 remains in that
small historical record but its redundant raw checkpoints were removed.

## What remains unresolved

These observations demonstrate an optimization benefit for this fixed task.
They do not establish that update interference causes the failed fits or that the
same pattern is typical across teachers. Gradient magnitudes and conflicts also
change with residual loss, so comparisons should include comparable loss levels.
Neuron direction matches provide a feature-recovery diagnostic, not a certificate
that the learned functions or representations are identical.

Earlier teacher-4, teacher-64 and teacher-512 scans motivated the longer-budget
check. Their raw outputs and old notebooks were removed to focus this repository
on the present teacher-8 experiment. The interrupted Martinelli work is excluded.
