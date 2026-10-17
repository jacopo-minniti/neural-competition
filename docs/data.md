# Retained data and cleanup

The final archive preserves the reported results of every study while reducing
raw data to representative cases. Original run data occupied **8.27 GB**
across 10,412 files. The curated payload is approximately
**556 MB**, about **93.3% smaller**.
Figures and compact result tables are also versioned separately under `docs/`.
Sizes are decimal bytes; filesystem allocation may differ.

The archive is ignored at `runs/retained/`. The complete
[retention manifest](provenance/retention_manifest.json) records every retained
file's hash, size, source and category, each selected case, raw-array axes and
the original source-tree inventory. The [original curation record](provenance/initial_curation.json)
documents the earlier teacher-8 cleanup and is historical provenance, not the
current inventory. The [archive audit](provenance/archive_audit.json) records
the final numerical and checksum checks.

## What was preserved before deletion

- **All cohort-level computed tables:** 105 compressed CSVs,
  including original/full-seed summaries, paired effects, dense cumulative curves,
  loss bins, coverage, per-batch and per-neuron diagnostics, robustness controls
  and the Adam cross-batch results. Derived analysis caches duplicate these data
  and are not retained separately.
- All scalar training/diagnostic logs and run states from the nine legacy
  collections, including divergent starts. The older 198-entry historical
  teacher-8 record remains [versioned](tables/teacher8_history.csv).
- Every group's original configuration, audited execution sources and historical
  audit output needed to interpret the claims. Immutable pre-cleanup source
  copies distinguish original execution hashes from the relocated live scripts.
- Four exact teacher/dataset realizations, original and fresh probe indices,
  and both the 256- and 512-example reference draws.
- Published figures and compact tables in [figures](figures/README.md) and
  [tables](tables/README.md), so reports do not depend on deleted run folders.

The broad pair-budget summaries were explicitly recovered from all original
exact pair arrays before pruning; the early momentum 14.4%→8.9% result and its
later-stage controls are therefore full-cohort results. Published uncertainty
and success rates were never recomputed from the retained raw subset.

## Representative raw evidence

| Retained evidence | Scope and reason |
| --- | --- |
| 55 legacy cases, 193 gradient checkpoints | Original width-8/9/10/16/128 contrasts; seed-0 narrow failure/wider success; seed-3 narrow success; seed-5 width-10 counterexample; SGD/momentum/Adam, smaller rate, Muon, initialization and high-rate continuation controls |
| 56 compact optimizer-state files for 15 groups | Initial/final 200k SGD/momentum states and initial/10k/50k/100k robustness states; seeds 0/3, all widths in each retained setting; both teacher sizes/realizations, learning-rate and initialization controls |
| Seven compressed every-update archives | Seed 0: full 200k widths 8/16 for SGD and momentum; full 100k K/K+2/2K/4K for the three teacher-8 main profiles and teacher-16 momentum/Adam-.001. Retains 2.8 million model updates exactly |
| 36 Adam cross-batch snapshots | Seeds 0/3, widths 8/16/32, both rates, matched early loss and 10k/100k; parameters, moments, gradients, directions and dot matrices |
| 12 fresh 512-batch snapshots | Widths 8/16, seeds 0/3, SGD/momentum/Adam, for direct distribution checks |
| 106 exact pair-array files | Seeds 0/3, original/fresh source draws, full/tiny steps and relevant phases, for recalculating strict and broad interaction metrics |

The manifest lists the exact model identities and checkpoint steps. These are
**purposive examples**, not a representative probability sample. Do not use
them to estimate success frequencies or population uncertainty. Every setting in the later 205-run robustness sweep remains represented by
checkpoints, even where its full per-update array was pruned.

## Layout and loading

```text
runs/retained/
  manifest.json
  datasets/                 # exact data, teachers, source/reference probe draws
  aggregates/               # original computed CSV.gz tables and all scalar logs
  checkpoints/
    legacy/                 # original gradient snapshots and final states
    trajectory/             # compact initial/final SGD and momentum states
    robustness/             # compact states for all 13 robustness groups
  raw_updates/              # representative exact scalar trajectories, NPZ
  cross_batch/              # Adam gradients, moments and candidate directions
  tail_checks/              # fresh 512-batch gradient snapshots
  pair_interactions/        # exact standalone and pair effects
  provenance/               # original configs, sources, audits and source inventory
```

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

archive = Path("runs/retained")
manifest = json.loads((archive / "manifest.json").read_text())
meta = manifest["update_archives"][0]
with np.load(archive / meta["path"]) as saved:
    records = saved["records"]
print(records.shape, meta["models"], meta["views"], meta["fields"])
table = pd.read_csv(archive / "aggregates/robustness/summary.csv.gz")
checkpoint = torch.load(
    archive / "checkpoints/trajectory/momentum/final.pt", weights_only=True
)
```

Per-update axes are `[updates, selected_models, source_or_reference, fields]`.
Values are losslessly compressed float64. No time thinning or dtype conversion
was applied to retained raw trajectories. Field names, width/seed identities,
original column indices and views are in the manifest.

Compact checkpoints preserve exact selected tensors. They include `models`,
`original_model_indices` and `rng_seed_order` so optimizer moments and random
streams can be mapped to the selected seeds. In these batched files, `velocity` stores the SGD momentum buffer or Adam first moment; `second` stores Adam’s second moment. Actual widths may be smaller than
the padded tensor dimension. They are inspection/continuation evidence, not
drop-in original-layout resume directories. The 200k runs originally kept only
initial/final optimizer states; intermediate ones cannot be recovered here.

Historical absolute paths in original tables/snapshots are provenance strings,
not current file locations. Use the manifest's `path` and `source` fields to
map retained evidence. All Torch data use tensors/basic types and can be loaded
with `weights_only=True` on CPU.

## What was removed and what can still be reproduced

After reports, all numerical tables and the archive were verified, redundant
original raw cohorts, unselected checkpoints, full unselected every-update arrays,
duplicate analysis caches, PDFs, logs, PID markers and every source/executed
notebook were removed. `analyses/` was removed entirely. Python cache directories
were cleared. The environment, Git history, engine, tests, original study code
and configurations remain.

The preserved full-cohort tables support checking the reported findings and
replotting results. Representative raw states support new local gradient/pair
checks; seven raw arrays support exact per-step reanalysis of selected cases.
New arbitrary diagnostics for every original seed require regenerating that
cohort. [Reproduction instructions](reproduction.md) make this distinction explicit.

Run `uv run python scripts/audit_archive.py` to verify checksums, exact accounting,
initial/final checkpoint agreement, retained Adam directions and independent
autograd gradients. Historical full-cohort audits were recorded before deletion;
the current archive audit validates the curated subset rather than claiming to
replay every deleted input.

Final compact-archive audit: 861 file hashes verified, 2.8 million retained model updates checked, maximum accounting error 3.49e-14, and zero discrepancy between retained checkpoint reference losses and raw records. All 36 retained Adam direction snapshots agree exactly with reconstruction; independent autograd error is below 4.2e-17.
