# Experimental protocol

## Model and data

For width $m$, the student is

$$
f_\theta(x)=\frac1{\sqrt m}\sum_{r=1}^m a_r\,\operatorname{ReLU}(w_r^\top x+b_r).
$$

Inputs have 20 independent standard Gaussian coordinates. The fixed teacher has
eight neurons and the same functional form, with random incoming and outgoing
weights, zero hidden biases, and no output bias. Its output coefficients are
rescaled so that the sample standard deviation of its training outputs equals
one; the mean is not subtracted. Targets contain no added noise.

- Data seed: 2026; teacher parameter seed: 2027.
- Training examples: 8,192; test examples: 4,096, shared by every run.
- Student incoming weights: independent N(0, 1/20); hidden biases and outgoing
  weights start at zero. Incoming weights are different across neurons.
- Both layers and hidden biases train. Hidden gradients vanish on the first
  update because outgoing weights start at zero; they generally become nonzero.
- No output bias. Extra-width students can represent the teacher by embedding
  its neurons, adjusting outgoing coefficients for the width factor, and setting
  unused outgoing coefficients to zero.

## Optimization

- Plain SGD, constant learning rate 1, no momentum or weight decay.
- Loss: half-MSE, $L=\frac1{2n}\sum_i(f_\theta(x_i)-y_i)^2$.
- Batch size 64, sampled with replacement, for 100,000 updates.
- Within a seed, every width receives the same batch sequence. The sequence is
  generated with seed `data_seed + 10000 + student_seed`. Across seeds, both
  initialization and minibatch draws change, so seed variability combines them.
- CPU float32 training; four CPU threads. Curvature uses CPU float64.

The legacy optimizer label `sgd_large` means SGD at learning rate 1. The config
schema's `momentum` default is used only when optimizer name is `momentum`;
the saved Torch optimizer state confirms momentum is zero for these runs.

## Focused collections

| Collection | Widths | Seeds | Runs |
|---|---|---|---:|
| `runs/teacher8` | 8, 9, 10, 16, 128 | 0, 1, 3 | 15 |
| `runs/teacher8_counterexample` | 8, 10 | 5 | 2 |

All 17 cases already had complete diagnostics. Their original files were copied
without modification, verified against source checksums and numerically audited.
No training was repeated during cleanup. The seed selection is for mechanism
inspection; the broader reliability record is preserved separately.

## Checkpoints and probe batches

Saved pre-update step indices are
`0, 1, 20, 100, 700, 1999, 4999, 9999, 19999, 39999, 59999, 79999, 99999`.
At index t, gradients and `parameters_before` refer to theta_t, after exactly t
updates. The saved actual update produces theta_(t+1). Post-update metrics at
index 99999 therefore measure the result after 100,000 updates.

The dataset contains 64 fixed probe minibatches, each with 64 training indices
sampled with replacement, shared across widths/seeds/checkpoints. Four of these
batches supply individual curvature estimates. The raw files contain every probe
gradient, not just averaged conflict statistics. Parameters and data permit
additional offline diagnostics without rerunning optimization.

## Reproduction

```sh
uv sync --python 3.12 --extra analysis --extra dev
uv run python scripts/audit_results.py
```

To reproduce optimization in a separate output directory:

```sh
uv run python scripts/run_experiments.py --config configs/teacher8.json --device cpu --output runs/reproduction
uv run python scripts/run_experiments.py --config configs/teacher8_counterexample.json --device cpu --output runs/reproduction_counterexample
```

`--resume` skips completed/diverged runs and restarts an incomplete run from its
initialization. It does not continue from a saved optimizer state. The saved
configuration must match. The existing curated outputs need no rerun.
The generic code still supports other configurations and optimizers, but only
the focused eight-neuron teacher results are retained here.
