# Data retention

The 18 October 2026 curation keeps the current experiments fully inspectable,
compact historical evidence, and selected old teacher states. It removes retired
raw diagnostics and regenerable caches. Raw data are ignored by Git; publishing
this repository does not publish local checkpoints or datasets.

## What remains

| Location | Retained evidence |
| --- | --- |
| `docs/tables/feature_profile/` | All new intervention curves, controls, forecasts, extension results, executed sources and audits |
| `runs/feature_profile/` | All new raw curves, correction summaries, parameter/optimizer checkpoints and manifests; deterministic datasets rebuilt from seeds |
| `runs/competition_pilot/` | Small exploratory screening and rejected profile-drift traces, with executed source copies; not confirmatory evidence |
| `docs/tables/residual_growth/` | All non-pilot curves, endpoints, roles, feature probes, contrasts, geometry and executed-source provenance |
| `runs/residual_growth/`, `runs/polarity/` | Every reported non-pilot final parameter/optimizer state, generated dataset, curve, probe and manifest |
| `docs/tables/mnist/` | Complete main/control results, diagnostic summaries, tuning, split indices, audits and executed source snapshots |
| `runs/mnist/main/` | All five recorded checkpoints for all 120 main starts, including optimizer state; scalar diagnostics and completion records |
| `runs/mnist/matched/` | All 236 accepted main matched-loss states and their diagnostics |
| `runs/mnist/lower_lr/`, `width_tuned/` | All 80 final control checkpoints, scalar diagnostics and completion records |
| Other MNIST folders | Tuning, control matched-state scalar diagnostics, dense replay curves and audit summaries |
| `data/mnist/raw/` | Four original checksummed compressed IDX archives; the prepared tensor cache is regenerable |
| `docs/tables/history/` | Complete retained historical tables, including unfavorable results and seed uncertainty |
| `runs/retained/` | Older representative datasets, checkpoints and execution provenance; not a full raw-trajectory archive |

[local_retention.json](provenance/local_retention.json) is the authoritative
path/size/SHA-256 inventory at curation time. Verify it with:

```sh
uv run python scripts/check_repository.py --local-data
```

This checks integrity, not the scientific interpretation. New runs and recreated
caches are intentionally not required by the fixed inventory. The later feature-
profile study has its own versioned provenance hashes and reconstruction audit;
rerun its analyzer to validate the new raw checkpoints. Representative
old states must not be used to estimate cohort success frequencies.

## What was removed

Retired specialization, allocation and temporal-branch raw runs; old gradient,
pair and every-update diagnostic arrays; duplicate historical aggregate caches;
superseded independent-probe MNIST runs; intermediate and matched-state control
checkpoints; dense per-update MNIST arrays; logs; Python caches; and the prepared
MNIST tensor cache. Main MNIST checkpoints and all current residual-growth /
polarity result states were kept.

[cleanup.json](provenance/cleanup.json) records removal categories, byte counts,
and table relocations. Deleting tracked files reduces the working tree, not Git
history. No history rewrite or environment removal was performed.

## Recovery and reproducibility limits

Commit `b03c53f` preserves the complete pre-curation tracked research tree,
including retired producers, configurations, reports and formerly versioned
arrays. Use it in a separate checkout for historical reproduction. Raw ignored
files deleted during cleanup are **not** in that commit. Recreating them requires
rerunning training or diagnostics; summary tables cannot recover parameter paths.

Current residual-growth analysis can reconstruct every saved final loss. MNIST
analysis can regenerate all published tables and plots from retained scalar
outputs without dense arrays or prepared data. New MNIST probes will recreate
the prepared cache automatically. Repeating a removed dense audit requires
recomputation, rather than simply reading its old completion marker.

Historical provenance strings can refer to files no longer present or to the
original machine's absolute paths. They are immutable execution records, not
current path indexes. Source hashes are resolved against exact snapshots rather
than silently updated to the refactored implementation. See
[reproduction](reproduction.md) for the maintained commands.
