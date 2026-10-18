# Historical evidence retained after consolidation

These tables support [research history](../../reports/research_history.md).
They preserve cohort outcomes, seed-level uncertainty and important negative
results; they are not new experiments or a selection of successful starts.

| Folder | Evidence |
| --- | --- |
| `teacher/` | Original fitting, gradient/step transfer, pair interactions, robustness and saved-state checks |
| `cumulative_budget/` | Exact cumulative endpoint budgets by seed and setting |
| `effect_magnitude/` | Objective-scale comparisons and early prediction checks |
| `strict_matched_loss/` | Matched-loss coverage, seed contrasts, and loss curves |
| `temporal_branches/` | Paired intervention summaries, seed values, coverage, width contrasts, and recipient summaries; `fixed_fraction/` is the alternative dose control |
| `allocation/` | Allocation/width summaries, seed values, readout refits, and original provenance |
| `specialization/` | Endpoint/paired results, population curvature, and original provenance |

Historical provenance contains paths to retired code and arrays. Those paths
record the original execution, not a promise that files still exist locally.
Git commit `b03c53f` preserves the original source, reports and all formerly
versioned tables. Use that revision in a separate checkout for historical reruns.
New producer versions cannot be substituted silently for their recorded hashes.

Episode-level allocation/temporal arrays, old raw diagnostic tensors, and
specialization checkpoints were removed after retaining these summaries.
They cannot be reconstructed from summary tables alone. The current local
[retention manifest](../../provenance/local_retention.json) lists what is available.
