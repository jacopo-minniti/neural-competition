# Experimental protocol and study ledger

## Shared teacher–student setting

The scalar student is f(x)=sum_r a_r ReLU(w_r dot x+b_r)/sqrt(m), with one
hidden layer, no output bias, and neuron blocks (w_r,b_r,a_r). Inputs are
20-dimensional standard Gaussian. There are 8,192 training and 4,096 independent
test examples; input seed 2026. Teacher targets are noiseless, scaled to unit
sample training standard deviation **without centering their mean**.

The original teacher has K=8 neurons, parameter seed 2027. Later robustness
experiments add K=16 and parameter seed 2037. Both student layers and hidden
biases train throughout the documented mechanism studies. Incoming student
weights start N(0,1/20); hidden biases are zero. Outgoing coefficients are
initially zero except in explicit initialization controls. With zero outgoing
weights, the first hidden-layer gradient vanishes, then generally becomes
nonzero. Random outgoing initialization changes both initial predictions and
the immediate hidden-feature gradient.

Every width at least K can represent the teacher by embedding its neurons,
rescaling outgoing coefficients for width normalization, and setting unused
coefficients to zero. Comparisons above K therefore test optimization under
overparameterization relative to teacher representation. They do not isolate
all possible effects of width: the common learning rate and normalization do
not fix hidden-feature learning strength or update norms across widths.

Training minimizes half-MSE. Fitting tables report ordinary MSE unless stated
otherwise; raw logs and exact interaction costs use half-MSE. Minibatches of
64 are sampled with replacement. Initialization seed s also selects minibatch
seed 12026+s, shared across student widths and profiles. Seed variation combines
initialization and minibatch randomness. Equal seeds do not make optimizers'
trajectories identical.

Training uses CPU float32 and four threads per worker. Exact scalar diagnostics
generally use float64. The batched every-update implementation is tested against
PyTorch gradients and optimizer updates; changing operation order can change
which basin a trajectory reaches. Those batched studies are new trajectories,
not bitwise continuations of the earlier sequential experiments.

## Study ledger

| Study | Widths / teachers | Seeds and budget | Main controls |
| --- | --- | --- | --- |
| Original longer-budget study | K=8; students 8/9/10/16/64/128 | Five seeds for initial 20k rate grid; 100k SGD-lr-1 cohort has 20 seeds at 8/9/10, eight at 16, five at 64/128 | SGD .1/1, momentum .01/.1; 198 completed historical entries across separate budgets |
| Main distributions | K=8; students 8/9/10/16/32 | Ten seeds × three optimizers × five widths; 100k; 150 runs | SGD 1, momentum .1, Adam .001 |
| Early distributions | Same K/widths | Forty seeds per profile/width; snapshots near MSE .2; 600 starts, 599 loss-eligible | 128 probes, then 512 fresh probes for widths 8/16 |
| Rate/Muon pilot | K=8; students 8/10/16 | Five seeds; 20k; 45 runs | SGD .1, Adam .01, Muon/Adam hybrid |
| Initial Adam initialization control | K=8; students 8/16 | Ten seeds; 100k; 20 runs | Outgoing N(0,.1²), paired against zero initialization |
| Stability pilot | K=8; students 8/16 | Five seeds; 20k; 40 starts, including three divergent starts | SGD 2/4 and momentum .2/.4 |
| Saved-state continuation | K=8; student 16 | Five identical 20k high-rate starting states; another 10k at each of four rates | SGD 1/2/4/5, common future minibatches |
| Neuron and pair diagnostics | K=8; students 8/16 | Reuse original states, forty early paired seeds; eight SGD/seven momentum/five low-rate later pairs | Both gradient and actual optimizer directions; full and 1% step; fresh sources |
| Every-update trajectory | K=8; students 8/16 | Ten seeds per optimizer; 200k; 40 runs, eight million updates | SGD 1 and momentum .1; source plus fixed 512-point reference |
| Robustness main/control sweep | K=8/16; students K/K+2/2K/4K | Five seeds; 100k; 175 planned runs | Momentum .1, Adam .001/.01; momentum .03; second teacher realization |
| Robustness initialization follow-up | K=8; students 8/16/32 | Five seeds; 100k; 30 additional runs | Outgoing std .1/1 with Adam .001; post-hoc motivated control |
| Quick Adam cross-batch | K=8; students 8/16/32 | Five seeds × two rates × six state views = 180 states | 100/1k/10k/50k/100k plus early matched loss; 128 fresh batches |

The distribution/pilot collections together contain 875 training starts: 872
completed and three divergent. Many are short early runs; counts across studies
should not be treated as independent replications of one setting. Diagnostic
reuses of existing snapshots are not new training starts. The robustness sweep
has 205 completed runs and 20.5 million actual updates.

## Optimizers and labels

Classical momentum has beta .9, no dampening/Nesterov or weight decay. Adam uses
betas (.9,.999), epsilon 1e-8, no weight decay/AMSGrad and constant learning rate.
The small Muon pilot uses PyTorch Muon on the incoming weight matrix, lr .02,
momentum .95 and Nesterov with original aspect-ratio adjustment; Adam .001
updates biases and outgoing weights. It is a hybrid pilot, not an optimizer
ranking or an exhaustive search.

Labels are historical and **not globally interchangeable**. In the original
distribution collections `adam_0001` means lr .001 and pilot `adam_001` means
lr .01. In the robustness and latest Adam check, `adam_001` means lr .001 and
`adam_01` means lr .01. `sgd_large` is plain SGD lr 1. Always use the saved
optimizer specification and collection, rather than joining on labels alone.

## Checkpoint and evaluation conventions

Original sequential snapshots at step t store parameters/gradients before update
t+1, after exactly t completed updates. `parameters_after` and `updates_completed`
include that next update. Batched trajectory checkpoint `completed=t` instead
stores the state after exactly t updates. Adam bias correction for its next
candidate therefore uses t+1. These conventions matter when restoring moments.

Original probe batches are sampled from the training data, independently with
replacement; they need not be disjoint. Cross-batch raw scores can use unordered
pairs, but optimizer transfer uses all ordered off-diagonal pairs because it is
asymmetric. Independent test examples are used for collateral prediction harm.
Same-batch pair experiments evaluate each candidate on its own source batch.

The 200k study measures every actual update on the source batch and a fixed
512-training-example reference. Robustness uses a fixed 256-example reference.
Both use draw seed 20261020. These are distinct evaluation sets; their raw
penalty magnitudes must not be interpreted as extensions of the same curve.
Full train/test MSE is evaluated every 1,000 updates. Robustness's main fitting
summary averages the last ten evaluations, updates 91k–100k, within each seed.
Latest cross-batch checkpoint tables use the actual checkpoint loss instead.

## Matching, uncertainty and numerical checks

Early comparisons target training MSE .2. Later comparisons near .01 use
explicit eligibility windows and retain attained loss/seed counts. The latest
Adam replay uses a small grid and a ±10% eligibility window. Equal-step curves
are not equal-loss comparisons. Matched reference-loss summaries require common
bin coverage and record missingness; success-conditioned late samples do not
represent every initialization.

Seed bootstrap intervals are exploratory, pointwise, paired and conditional on
the selected teacher and dataset. Most scalar tables use 4,000 resamples with
seed 741; some trajectory curves use fewer. Batch pairs, neurons and examples
are not treated as independent uncertainty samples. No multiple-comparison
correction or broad population generalization is claimed.

Audits check autograd gradients, optimizer moments/updates, exact finite
prediction changes, direct four-loss contrasts, fixed-reference telescoping,
source hashes, eligibility and sign-tolerance sensitivity. Full-cohort audit
results were saved before pruning. [The retention manifest](data.md) separates
those historical validations from checks that can be rerun on the compact archive.
