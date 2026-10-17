# Same-batch interference between neuron updates

The strict pairwise test detects fewer harmful combinations per eligible pair
at width 16, most clearly early in momentum training. Their aggregate marginal
harm is also smaller. However, individual events are not clearly milder relative
to the affected neuron's standalone benefit, and the absolute number of events
per batch is not clearly lower. The aggregate harm is small and strongly depends
on step length. These observations do not establish the cause of the fitting
advantage.

## Protocol

We reuse the existing fixed eight-neuron teacher, Gaussian inputs in 20
dimensions, 8,192 training examples and noiseless targets. Both student layers
and hidden biases train, with 1/sqrt(width) output normalization and zero initial
outgoing coefficients. Widths are 8 and 16. Optimization uses half-MSE; matching
uses ordinary full-training MSE.

Main settings are plain SGD lr 1 and classical momentum lr 0.1, beta 0.9.
An existing SGD-lr-0.1 control supplies five later paired seeds. Momentum candidate
directions retain the saved PRE-update buffer: delta_B=-lr*(beta*v_previous+g_B).
We hold this history and all proposed block updates fixed while combining neurons.
No gradients are recomputed after applying another neuron's update.

No training was repeated. There are 352 unique checkpoints across 360 comparison
views:

- Early matched training MSE approximately 0.2: 40 paired seeds per main optimizer.
- Later matched MSE approximately 0.01: eight paired SGD seeds, seven momentum
  seeds, and five SGD-lr-0.1 control seeds. The matching window is +/-25%; attained
  mean MSE is approximately 0.009005/0.008974 for SGD widths 8/16 and
  0.009317/0.008938 for momentum. Eligibility is conditional on having a suitable
  saved checkpoint.
- Equal-step checks: steps 100, 700, 1,999 and 19,999, ten paired seeds per main
  optimizer. Losses can differ substantially between widths at equal steps.

For EACH state we use all 128 saved independent minibatch draws of size 64, and
evaluate effects on the SAME source minibatch that generated the update. We
enumerate all 28 unordered neuron pairs at width 8 and all 120 at width 16,
checking both orders: 56 and 240 directed tests per batch. The two step scales
are full length and 1%. Across comparison views and step scales, the original
probe analysis evaluates 13,639,680 directed cases. These are correlated cases,
not independent uncertainty samples.

An early validation repeats the experiment on 32 fresh source minibatches using
the previously recomputed gradients (draw seed 20261018). Both main profiles
retain all 40 paired seeds. No held-out batch C is involved in this experiment.

## Exact criterion and definitions

For current residual e=f-y and each neuron's exact finite prediction change d_r,

\[
A_r=\mathbb E_B[e d_r+\tfrac12 d_r^2],\qquad
I_{rs}=\mathbb E_B[d_rd_s].
\]

Here A_r is the half-MSE change from updating r alone. The joint loss change is
A_r+A_s+I_rs, so the marginal loss change of adding s after r is A_s+I_rs.
These identities are exact for this additive architecture and squared loss,
including ReLU gate crossings and simultaneous changes of both layers. We compute
them in float64 using saved parameters and candidate updates, without full
network forwards for every pair.

An eligible ordered pair r->s has A_r<0 and A_s<0. It is a strong interference
event when A_s+I_rs>0: both updates help alone, but adding s after r increases
the batch loss. We use tolerance 1e-10 times that source batch's current
half-MSE for all sign tests. Checks at tolerance zero and 1e-8 leave the main
early conclusions intact.

We report three distinct quantities to avoid conflating frequency and severity:

1. **Frequency F:** number of strong events / number of eligible ordered pairs.
2. **Aggregate severity S:** sum of positive marginal harm over strong events /
   sum of the second neuron's standalone benefit (-A_s) over ALL eligible cases.
   This weights contributions by size and includes zero harm for eligible
   non-events. It combines how often interference occurs and how costly it is.
3. **Event severity S_event:** the same harm numerator, with the benefit
   denominator restricted to interfering cases. This describes how harmful an
   event is relative to the benefit that its affected update originally had.
   It is undefined if a seed has no events; valid counts are retained.

Batches are pooled within each training seed before forming these ratios; the
reported result averages the seed-level ratios. Intervals resample paired
training seeds 4,000 times, seed 741, conditional on this teacher/data. They are
exploratory pointwise 95% intervals without multiplicity correction. Averages of
ratios need not equal ratios of the counts averaged across seeds.

## Early results at matched MSE approximately 0.2

Full candidate steps, 40 paired seeds:

| Optimizer | Width | Frequency F | Aggregate severity S | Event severity S_event |
| --- | ---: | ---: | ---: | ---: |
| SGD lr 1 | 8 | 0.0276% | 0.0000840% | 0.296x |
| SGD lr 1 | 16 | 0.0098% | 0.00000488% | 0.390x |
| Momentum lr 0.1 | 8 | 1.2448% | 0.04492% | 0.975x |
| Momentum lr 0.1 | 16 | 0.2532% | 0.002155% | 1.039x |

For momentum, frequency falls by **0.992 percentage points**, paired interval
[0.759, 1.237] for the reduction. This is about an 80% decrease per eligible
ordered pair. Aggregate severity decreases by 0.04276 percentage points,
interval [0.02870, 0.05904], about 95% of its already small initial value.

The condition that both neurons help alone holds for **73.0% -> 82.6%** of
ordered momentum cases. The reduction is therefore not obtained by excluding
more pairs at width 16. Nearly all SGD cases are eligible.

Event severity does NOT clearly improve with width. The early momentum estimate
is about one times the affected update's standalone benefit at both widths;
its paired width difference is +0.063 [-0.100, 0.235], using 39 common seeds
with events. Width 8 has events in 39/40 seeds and width 16 in 40/40. For SGD,
events are present in 17/40 and 19/40 seeds; its conditional severity comparison
has only ten common event-positive seeds and is also uncertain.

### Pair count and magnitude matter

| Early momentum diagnostic | Width 8 | Width 16 |
| --- | ---: | ---: |
| Harmful ordered cases per source batch | 0.481 | 0.474 |
| Summed pairwise marginal harm per batch, half-MSE units | 1.535e-5 | 1.523e-6 |
| Pairwise harm sum / full-step source improvement | 0.394% | 0.0346% |

The total event count per batch is essentially unchanged: paired difference
-0.007 [-0.121, 0.109]. A wider model provides many more eligible pairs, so the
lower per-pair frequency should not be described as fewer total events.
Nevertheless, the actual sum of marginal harm is about ten times smaller;
that reduction is not simply division by a larger pair count.

The event-positive updates have very small standalone NET benefits: they account
for about 0.0463% of eligible standalone benefit at width 8 and 0.00217% at
width 16. This explains how an event can be severe relative to its affected
update's benefit while contributing very little aggregate harm. It does not
imply the neuron's parameter or prediction movement itself is tiny.

Summed pairwise harm is a diagnostic over alternative two-neuron combinations,
NOT an additive decomposition of the loss cost of the actual full update.
Pairs can share neurons. We cannot read these percentages as a causal fraction
of training loss wasted through interference.

![Pair frequency and severity](../runs/pier_study/pairs/pair_interference.png)

## Fresh-batch validation

At full step length, the fresh 32-batch check gives:

| Optimizer | Frequency: width 8 -> 16 | Aggregate severity: width 8 -> 16 |
| --- | ---: | ---: |
| SGD | 0.0293% -> 0.0075% | 0.0000440% -> 0.00000648% |
| Momentum | 1.1344% -> 0.2767% | 0.03692% -> 0.002138% |

Momentum's frequency and aggregate-severity reductions remain clear across
seeds. SGD's frequency reduction repeats; its extremely small aggregate-severity
difference has an interval including zero in this fresh check.

## Step length

At 1% step length, SGD has no observed strong interference in either matched-loss
comparison. Early momentum frequency becomes **0.00793% -> 0.00146%** in the
original probe draw, with aggregate severity only **0.00000163% -> 0.0000000663%**.
The fresh tiny-step draw instead gives frequency 0.0043% -> 0.0040%, with no
clear width difference. Events are too rare and weak here to establish a
reproducible small-step advantage.

Thus this strict same-batch pair metric mainly detects finite-step interactions.
This differs from the earlier momentum-transfer result, where a width advantage
survived shortening steps when evaluating OTHER batches. The experiments measure
different effects; one should not be used as a substitute for the other.

![Step-length control](../runs/pier_study/pairs/pair_step_sensitivity.png)

## Later stages

At matched training MSE approximately 0.01, full steps:

| Optimizer | Seeds | Frequency: width 8 -> 16 | Aggregate severity: width 8 -> 16 |
| --- | ---: | ---: | ---: |
| SGD lr 1 | 8 | 0.0977% -> 0.0110% | 0.0001780% -> 0.00000372% |
| Momentum lr 0.1 | 7 | 0.1567% -> 0.1228% | 0.001054% -> 0.000979% |
| SGD lr 0.1 | 5 | no events observed | no events observed |

The later SGD reductions are clear within this sample, although absolute harm
remains very small. Momentum's later differences are uncertain; its early
advantage does not clearly persist at matched loss.

Equal-step checks are saved in the notebook. The full-step momentum per-pair
frequency is lower at width 16 at steps 100, 700, 1,999 and 19,999, with varying
uncertainty. Same-step losses differ, particularly by 20k updates. For SGD, the
per-pair frequency advantage is not uniform; at 20k the wider point estimate is
higher, while aggregate severity is still lower.

## Interpretation

This is direct evidence about interactions between neuron updates on ONE batch.
It supports reduced pairwise marginal harm at greater width in the tested early
momentum regime, and a lower probability per eligible pair. It does not show
fewer harmful interactions per batch or milder events relative to their own
standalone benefit. Their overall magnitude is small.

The strict definition excludes ordinary diminishing returns, events involving an
already harmful individual update, and interactions that require three or more
neurons. It therefore does not measure every form of redundancy or competition.
Width also changes individual step sizes and how useful corrections are spread
among neurons. Matching loss and normalizing severity do not establish learned
coordination or a causal explanation for the final fitting advantage.

## Reproduce and inspect

Open [06_pair_interference.ipynb](../analyses/06_pair_interference.ipynb).

```sh
uv run python scripts/analyze_pairs.py
uv run python scripts/audit_pairs.py
uv run pytest
uv run ruff check src scripts tests analyses
```

All generated data are ignored under `runs/pier_study/pairs/`: per-source-batch
counts, directed neuron-pair aggregates, exact standalone/interaction arrays,
tolerance checks, fresh-batch validation, hashes, figures and the executed
notebook. A compact matched-loss summary is versioned beside this report.

The audit verifies 355 immutable inputs, checks 2,960 pair evaluations against
independent full-model forwards across ten states, including seven real positive
events, and recomputes twenty complete raw-array count/harm summaries with NumPy.
Maximum loss discrepancy is about 3e-17. Constructing early full updates in
float32 leaves event counts unchanged; normalized severity changes by at most
7.4e-9 per seed. Synthetic tests distinguish overshooting
from harmless diminishing returns, asymmetric pair orders, undefined eligibility,
and exact network losses with gate crossings.
