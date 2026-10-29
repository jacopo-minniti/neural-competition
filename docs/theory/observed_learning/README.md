# Observed-learning evidence and reproduction

Interpretation and derivations:
[Observed learning, sampling, and finite updates](../observed_learning_and_sampling.md).

| Record | Scope |
|---|---|
| `mnist/` | Four fresh pairs, seeds 283001–283004; every-step ledger and panel draws |
| `mnist_confirmation/` | Eight further pairs, 283101–283108; unchanged training/measurement design |
| `mnist_original_replay/` | Instrumented replay of existing 281801–281808 runs; not new independent evidence |
| `toy/` | 64 fresh nested starts, bank 283010; three tasks, widths 4/8 |
| `toy_half_step/` | Same starts at step .025 instead of .05; numerical sensitivity |
| `batch_probe/` | Fixed-checkpoint minibatch-size interventions on the eight confirmation pairs |
| `boundary_audit/` | Four step sizes and three quadratures on the same toy starts, to time 20 |
| `analysis.json` | Recomputed seed-level statistics and historical checks |
| `verification.json` | Uninstrumented fresh-run replay, historical loss checks, runtime versions |
| `mnist_ledger.png`, `toy_ledger.png`, `batch_intervention.png` | Regenerated figures |

Producers refuse to overwrite existing protocols. Protocols and source snapshots
precede their associated results. The boundary audit followed the half-step
discrepancy; the batch intervention followed the initial ledger finding. Neither
is an independent preregistration of the program. Confirmation uses the unchanged
initial design. No regression or prospective endpoint predictor was fitted.

Saved hashes identify executed source versions. MNIST records include the
prepared-data SHA-256; the local `data/mnist/prepared.pt` is unchanged from the
earlier studies. No dataset was downloaded.

## Reproduce into fresh directories

Run from the repository root with the existing environment and prepared data:

```sh
.venv/bin/python scripts/studies/trace_observed_learning.py mnist --output /private/tmp/ledger-mnist --seed 283001 --count 4
.venv/bin/python scripts/studies/trace_observed_learning.py mnist --output /private/tmp/ledger-confirmation --seed 283101 --count 8
.venv/bin/python scripts/studies/trace_observed_learning.py mnist --output /private/tmp/ledger-original-replay --seed 281801 --count 8
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/trace_observed_learning.py toy --output /private/tmp/ledger-toy --seed 283010 --count 64
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/trace_observed_learning.py toy --output /private/tmp/ledger-half-step --seed 283010 --count 64 --dt .025
.venv/bin/python scripts/studies/probe_ledger_noise.py --input docs/theory/observed_learning/mnist_confirmation --output /private/tmp/ledger-batch-probe
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/audit_ledger_boundaries.py --output /private/tmp/ledger-boundaries
```

The batch probe input must be a completed run directory inside the repository
(its protocol uses relative checkpoint references). Outputs can be elsewhere.
Use the retained inputs above, or reproduce confirmation into a fresh
repository-relative directory first.

Recompute retained analyses and verification:

```sh
MPLCONFIGDIR=/private/tmp/neural-competition-mpl .venv/bin/python scripts/studies/analyze_observed_learning.py
.venv/bin/python scripts/studies/verify_observed_learning.py
.venv/bin/pytest -q tests/test_learning_ledger.py tests/test_option_recovery.py
```

The analyzer targets the retained layout above and reads historical records
without changing them. The verifier independently reruns the 24 fresh float32
SGD trajectories without instrumentation and compares final parameters, then
re-evaluates 16 historical saved endpoints. Toy finite-step and component
identities are checked directly from retained arrays.

## Raw schemas

- MNIST `ledger_SEED_WIDTH.npz`: named `columns` of `values`, one row per SGD
  update. `signal` and `sampling` include step size. Positive `cost` is
  **subtracted**; `rounding` is signed. `closure` independently tests the
  output-space identity. Every intermediate parameter state is not retained.
- `checkpoints.jsonl`: full-training/validation loss, full signal and mean
  per-example squared gradient norm. `model_*.pt` contains actual float32 states.
- `panels.jsonl`: uniform samples without replacement, nested across sizes,
  shared across widths **and seeds**. `own` is self-response; `corrected`
  subtracts finite-population variance; `cross` uses the full gradient, an
  expensive benchmark. Seed SEs are conditional on the shared panel bank.
- Toy `trajectories.npz`: columns are in the protocol. First four are current
  losses; remaining columns sum **every** optimizer step. Detail self/cross
  terms concern a prespecified task projection, not new spectral options.
  `dense_loss` uses separate nine-node evaluation. Final states are retained.
- Batch `outcomes.jsonl`: actual full-training gains, exact ledger terms, and
  local GGN cost. Batches sample with replacement. Uncertainty is over eight
  initialization pairs after averaging the 32 draws at each checkpoint.
- Boundary `outcomes.jsonl`: one record per start, width, quadrature and step.
  `smooth` plus `gate` equals nonlinear work. These reuse the same starts;
  four finite steps do not prove a limit theorem or boundary sliding.

Tests enumerate all small fixed-size subsets for the sampling identity, compare
analytic gradients and Jacobian products with autograd, and verify the Bregman
ledger through ReLU crossings and logit-gauge changes.
