# Neural competition

Small PyTorch experiments asking whether extra width makes useful updates cause
less damage to other predictions, even when a narrower network can represent the
target. The first sweep is a synthetic teacher–student regression experiment.

## Setup

With [uv](https://docs.astral.sh/uv/):

```sh
uv sync --python 3.12 --extra analysis --extra dev
```

Or create a Python environment and install this package with
`pip install -e '.[analysis,dev]'`. No external dataset or account is required.

## First experiments

```sh
uv run python scripts/run_experiments.py --device cpu --output runs/first
```

The default configuration is `configs/first.json`:

- Inputs: 20 independent standard Gaussian coordinates.
- Teacher: 4 hidden ReLU neurons, with outputs rescaled to unit training standard
  deviation. No target noise.
- Students: widths 8, 64, and 512; one hidden layer; both layers trained. Each can
  represent the teacher exactly by embedding its neurons and zeroing the rest.
- Data: 8,192 training examples and 4,096 test examples, shared across all runs.
- Loss: **half-MSE**, `0.5 * mean((prediction - target)**2)`.
- Model: `sum(a_r * relu(w_r @ x + b_r)) / sqrt(width)`. Incoming weights have
  variance `1/input_dim`; biases and outgoing weights initially equal zero.
  Predictions and initial loss are therefore identical across all widths and
  seeds. Incoming weights are random, so neurons are not identical. Hidden-weight
  gradients are zero on the first update and generally nonzero afterward.
- Three initialization seeds, batch size 64, and 2,000 updates per run.
- SGD at learning rates 0.1 and 1.0; momentum SGD at 0.01 and 0.1 with beta 0.9.
  These pair the nominal low-frequency rate `lr / (1-beta)` with the SGD rate;
  they do **not** make the full trajectories or initial transients equivalent.
  Labels `small` and `large` describe relative rates; they do not certify EoSS.
- Every width and optimizer gets the same sampled training minibatches within a
  seed. Diagnostic probe batches are sampled independently with replacement from
  the training set and then fixed across checkpoints and runs.

This is 36 runs. Run a tiny end-to-end check first if changing the implementation:

```sh
uv run python scripts/run_experiments.py --smoke --device cpu --output runs/smoke
uv run pytest
```

Use `--device mps` for Apple GPU, `--device cuda` for NVIDIA GPU, or `--device auto`
to choose an available GPU before falling back to CPU. For the initial small
models, a CPU benchmark is worthwhile. `cpu_threads` controls CPU parallelism.
Curvature calculations always use CPU float64 and Hessian–vector products.

The entry point runs in the foreground and prints progress. To stop it, use
Ctrl-C (or SIGTERM). To continue the sweep:

```sh
uv run python scripts/run_experiments.py --device cpu --output runs/first --resume
```

Resume skips completed or diverged runs and restarts an incomplete run **from its
initialization**. It is not an optimizer-checkpoint continuation. The saved
configuration must match. Only one process may write to an output directory;
after a forced kill, verify the PID in `.lock` is no longer alive before removing
that stale lock.

## What gets saved

All generated data, logs, and results belong under `runs/`, which is gitignored.

```text
runs/first/
  config.json                 # Complete protocol
  environment.json            # Python, Torch, device and precision
  manifest.json               # Sweep progress and per-run status
  data.pt                     # Inputs, targets, teacher and fixed probe indices
  w0008_seed0_sgd_small/
    state.json                # Progress, status and final losses
    training.csv              # Sampled minibatch loss and step norm
    metrics.csv               # Checkpoint summaries
    checkpoints/step_000000.pt # Raw gradients, parameters and actual update
    final.pt                  # Final parameters and optimizer state
```

At each checkpoint, all gradients refer to the same **pre-update** state theta_t.
The checkpoint then records one real training update, theta_(t+1)-theta_t.
Step 0 is initialization; step 1999 precedes the 2,000th update. The last update
is always measured. `train_loss` and `test_loss` are pre-update; the corresponding
`*_after` fields measure the post-update state.

Each raw checkpoint contains:

- Probe minibatch gradients, shape `[probe_batch, neuron, input_dim + 2]`, with
  columns ordered as incoming weights, incoming bias, and outgoing weight.
- The exact full-training gradient, the actual training minibatch gradient, and
  individual probe losses before and after the update.
- Parameters before and after the update, the actual parameter change, and the
  previous actual parameter change.
- Momentum buffers and Adam's first/second moments before and after the update.
  Buffers absent from an optimizer are saved as zero arrays.
- Individual batch-sharpness samples, plus pre-update test predictions.

The raw gradients allow recomputing pair distributions and neuron attributions
without repeating training. Parameter snapshots and saved data also allow later
inspection of learned neuron responses. Checkpoints contain tensors and basic
Python data; load with `torch.load(path, map_location='cpu', weights_only=True)`.

## Measurements and interpretation

For batches B and C at the same state, `g_B dot g_C < 0` predicts that a small
SGD update for B increases C's loss. For each neuron r, the block score
`q_r = g_B,r dot g_C,r` attributes that conflict to competing demands on the same
neuron. The sum of block scores equals the global score. Negative contributions
from some neurons can be hidden by positive contributions from others.

For the actual update `delta = theta_after - theta_before`,
`g_C dot delta > 0` predicts harm. This sign convention also works for momentum
and Adam. Observed cross-batch loss changes measure actual finite-step effects.
The observed-minus-linear remainder includes higher-order terms and ReLU boundary
crossings; it is not a pure second-order measurement.

Batch sharpness is the **mean of per-batch Rayleigh quotients**
`mean(g_B.T @ H_B @ g_B / ||g_B||^2)`, not a ratio of their expectations and not
the full-data Hessian's top eigenvalue. Zero-gradient samples are recorded as NaN.
The current four-batch estimate is deliberately cheap and noisy. It is a pilot
diagnostic, not proof of operating at the edge of stochastic stability. The
plain-SGD `2/lr` reference is not applied to momentum or Adam.

Pairwise summaries exclude self-pairs. Histogram pairs are correlated; uncertainty
should come from independent runs/seeds, not treating all pairs as independent.
Inspect raw inner products alongside cosine similarities, gradient norms, actual
loss changes, and comparable-loss checkpoints. Euclidean gradient geometry and
width effects depend on parameterization. No monotonic benefit of width is assumed.

Add Adam by appending, for example, this optimizer configuration in a **new**
config/output directory:

```json
{"name": "adam", "lr": 0.001, "label": "adam_small"}
```

## Analysis

Open `analyses/first_experiments.ipynb` and select the `.venv` Python interpreter.
It loads partial results while training runs, plots learning curves, raw alignment
distributions, neuron attribution, actual cross-batch loss effects, curvature,
step reversals, and nearest checkpoints at a chosen loss. Rerun the loading cell
to refresh. The default experiment is `runs/first`; change `EXPERIMENT` in the
notebook to inspect another sweep.

The source is intentionally small: `model.py` defines the model/data,
`diagnostics.py` defines the measurements, `experiment.py` runs and saves the
sweep, and `config.py` defines the protocol. Tests check gradient attribution,
first-order loss effects, curvature against a tiny explicit Hessian, capacity,
and saved checkpoint consistency for SGD, momentum and Adam.
