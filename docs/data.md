# Retained data and cleanup

## What remains

There are 17 complete teacher-8 cases, 13 checkpoints each: **221 checkpoints**.
The core collection contains widths 8, 9, 10, 16 and 128 at seeds 0, 1 and 3;
the paired counterexample contains widths 8 and 10 at seed 5.

`teacher8_history.csv` preserves scalar outcomes from all 198 completed teacher-8
runs: 120 at 20k updates across four optimizer profiles and 78 at 100k with SGD
at learning rate 1. It also retains losses at 2k/20k/100k when available, so the
longer-budget conclusion can be checked after redundant raw files are deleted.
It excludes the interrupted case. Rows from different budgets/profiles must not
be pooled as independent trials of one setting.

`data_retention.json` records source configurations, original paths, retained file
checksums and the cleanup inventory. Original run states and checkpoint tensors
are unchanged. The metadata documents why the selected seeds are illustrative.

## Raw layout

```text
runs/teacher8/                  # 15 core runs
  config.json                  # Focused configuration
  environment.json             # Original execution environment
  data.pt                      # Inputs, targets, teacher and probe indices
  curation.json                # Reuse/provenance metadata
  manifest.json
  audit.json                   # Numerical validation report
  w0010_seed0_sgd_large/
    state.json
    training.csv
    metrics.csv
    final.pt
    checkpoints/step_*.pt
runs/teacher8_counterexample/   # Same structure, two seed-5 runs
runs/analysis/                 # Regenerated figures and executed notebooks
```

All generated outputs stay gitignored. The historical scalar record and notes
in `docs/` are versioned. Raw checkpoints use tensors and basic Python types;
load with `torch.load(path, map_location="cpu", weights_only=True)`.

## Saved checkpoint contents

| Field | Meaning |
|---|---|
| `parameters_before`, `parameters_after` | [width, 22] blocks: 20 incoming weights, hidden bias, outgoing weight |
| `probe_gradients` | [64, width, 22], half-MSE minibatch gradients at the same pre-update state |
| `mean_gradient` | Exact full-training gradient |
| `sampled_gradient` | Gradient of the minibatch actually used for this update |
| `actual_step`, `previous_step` | Consecutive realized parameter updates |
| `probe_losses_before`, `probe_losses_after` | Half-MSE on each fixed probe batch |
| `train_loss_*`, `test_loss_*` | Full-dataset half-MSE before/after the step |
| `test_predictions_before` | Pre-update test predictions |
| `batch_sharpness_samples` | Four individual curvature quotients |
| `optimizer_before`, `optimizer_after` | Momentum/Adam buffers; zero when absent |

Saved parameters plus the dataset also support offline per-example gradients,
more curvature probes, and prediction-space diagnostics without repeating training.
Those additional arrays are not claimed to have been stored originally.

## Verification and removal

Before removing old folders, every retained file was checked against its source
checksum. `scripts/audit_results.py` validates all checkpoint inventories/shapes,
the actual parameter differences and neuron-block dot decomposition. At steps
0, 1999, 19999 and 99999 it independently recomputes full-training gradients,
one probe gradient/loss, test predictions and post-step probe/test losses.

The audit passed for all 221 checkpoints. Existing diagnostic data was sufficient,
so **no optimization was rerun**. The notebooks perform further offline analyses.

Removed raw material includes other teachers, redundant teacher-8 runs, interrupted
runs, obsolete plots/logs, Martinelli trials and Julia runtimes/caches. Old notebooks
and configs pointing to deleted results were replaced by the focused notebooks
and configurations. The retention manifest records storage before and after.
