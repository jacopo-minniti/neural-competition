# Unequal-frequency parity dataset

This is an initial, fully enumerable dataset for studying whether pressures from
features at different frequencies alter shared neural implementations. It is a
controlled idealization, not an empirical model of natural data and not evidence
that competition occurs. For eight independent bits, sample `X_i ~ Bernoulli(q_i)`
and set `Y = sum(X_i) mod 2`. Every bit follows the same rule; multiple bits may
be active together. Bit presence is the frequent or rare event: all coordinates
are observed in every example. There is no assigned neuron role or common task.

## Usage

```python
import numpy as np
from neural_competition.parity import frequencies, panels, sample

q = frequencies(n_features=8, alpha=1.0, maximum=0.4)
train = sample(q, count=4096, rng=np.random.default_rng(1))
validation = sample(q, count=4096, rng=np.random.default_rng(2))
population, audit = panels(q)
# train/validation: x [N,8], y [N], float labels 0 or 1 for squared loss
# population/audit: x [256,8], y [256], p [256]
```

Use independent random generators for train and validation. `population` enumerates
all inputs with their exact product-distribution probabilities. `audit` uses the
same ordered support with fixed uniform weights, independent of `q`. It is an
exhaustive diagnostic panel, not an independent held-out dataset. A weighted loss
on `population` is the exact population loss; an equally weighted audit reports a
different risk. Sampling with replacement allows input overlap between train and
validation; the held-out data measure sampling variability, not novel input support.
The `panels` limit of 20 features prevents accidental unbounded enumeration.

`alpha=0` gives equal probabilities; larger values produce a smooth rank spectrum.
`maximum` fixes the most frequent probability, not the expected number of active
bits. Arbitrary strictly interior probability vectors can be passed directly.
Document the vector and encoding in every experiment. Do not standardize features
without accounting for the resulting change in update magnitude and optimizer geometry.

## What to measure

Compare widths using the existing shallow ReLU and squared-loss setup, at matched
steps and matched progress. Record realized exposure and probability per factor,
residual demand, fixed-audit response, gate changes, opening and reopening, and
population loss. A fixed contrast is a capability probe, not automatically a
spectral option. Finite-threshold weakening is not structural closure. A label-
relevant factor need not correspond to one neuron, one learned feature, or one
independent correction. Compare same-checkpoint pressure interventions and later
correction under matched conditions before claiming that response loss caused harm.

A bit-specific incoming-weight update includes `q_i` times a conditional residual
and gate expectation, but frequency is not pressure by itself. Co-occurring bits
prevent summing conditional example gradients as if they were disjoint. For an
exact additive decomposition, project the *current residual* into orthogonal
product-distribution components, including interactions; their velocities add.
The decomposition itself depends on `q`, so compare capability on the fixed audit.
Test changes in response under actual shared updates; do not presume independent
routes, protected neurons, coherent pressure, or permanent closure.

Changing `q` also changes label balance and signal geometry. In particular,
`P(Y=1) = (1 - product_i(1 - 2*q_i))/2`. Record this, total expected active
bits, initial gradients, and interaction strengths; use appropriate matched or
balanced controls. In the uniform `q_i=0.5` case full parity has no lower-order
population correlations. The same target has a simple width-eight shallow ReLU
witness with biases: compute the active-bit count and interpolate its alternating
values with piecewise-linear hinges. This establishes representational sufficiency,
not successful gradient training. No training results or width predictions are
claimed by this dataset implementation.

## Subsequent training and model extension

The initial raw parity exploration was diagnostic. As identified in
[reserve relaxation](model/reserve_relaxation.md), raw $\{0, 1\}$ binary parity
removed two essential preconditions for option competition:
1. Raw inputs with an intercept gave strictly non-negative dot products ($x^T x' \ge 1$),
   eliminating opposing geometry.
2. The empty input had target $0$ and began with near-zero residual ($e(\mathbf{0}) \approx 0$),
   exerting $< 1\%$ of the initial gradient norm despite high frequency.

### Standardized Bipolar Parity Resolution

Standardizing the setup resolves both limitations without synthetic features:
- **Bipolar labels:** $y = (-1)^{\sum X_i} \in \{-1, +1\}$. The empty pattern
  has $y(\mathbf{0}) = +1$ and initial residual $e(\mathbf{0}) \approx 1.0$.
  With $p(\mathbf{0}) \approx 0.46$, it exerts massive initial gradient pressure ($p \cdot e \approx 0.46$).
- **Standardized features:** $x_i = (X_i - q_i) / \sqrt{q_i(1 - q_i)}$. Whenever
  $\sum_i \frac{q_i}{1 - q_i} < 1$, every active state has a strictly negative
  dot product with the empty pattern direction ($u^T x < 0$), restoring genuine
  opposing geometry.

Under this formulation, learning becomes naturally and sequentially episodic:
dominant error pressure shifts from the empty pattern (Episode 1, $p \cdot |e| = 0.46$)
to frequent 1-bit patterns (Episode 2, $p \cdot |e| = 0.12$), 2-bit interactions,
and finally higher-order rare parity interactions. The analytic completion-reserve
flow $\mathcal{R}_r(s; \theta)$ predicts the resulting response weakening on rare
directions, and network width functions as the buffer protecting reserve capacity
($0.0\%$ rare probes weakened at width 16 vs $4.7\%$ at width 4, cutting final loss
from $0.082$ to $0.038$).

The [natural parity evidence record](evidence/natural_parity/README.md) details
comprehensive confirmations across 384 fresh seed banks, higher input dimensions ($D=6$),
and negative controls.

The earlier [frequency-pressure extension](../archive/docs/model/frequency_pressure.md) and
[executed study](../archive/docs/evidence/frequency_pressure/README.md) remain preserved as diagnostic
baselines.

## Literature and context

- [Cornacchia and Mossel, A Mathematical Model for Curriculum Learning for Parities (2023)](https://proceedings.mlr.press/v202/cornacchia23a.html): product distributions and training order can materially affect parity learning; their curriculum result is not a guarantee for this fixed distribution.
- [Abbe et al., Learning High-Degree Parities: The Crucial Role of the Initialization (2024/2025)](https://arxiv.org/abs/2412.04910): initialization affects learnability in the paper's high-degree parity setting; its results do not automatically transfer to this eight-bit setup.
- [Michaud et al., The Quantization Model of Neural Scaling (2023)](https://arxiv.org/abs/2303.13506): frequency-heterogeneous multitask sparse parity is a related benchmark, but its task-indexed subtasks and quantization assumptions differ from this single co-occurring-factor task.
