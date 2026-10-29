# Finite-budget implementation-slack evidence

Main note: [Implementation slack](../finite_budget_implementation_slack.md).
All losses are half MSE. `analysis.json` contains means, paired seed SEs, sign
counts, and the underlying paired differences, not just favorable summaries.

| Directory | Seeds | Purpose |
|---|---|---|
| `pilot` | 282801–282804 | Initial exploration |
| `confirmation` | 282811–282822 | Unchanged fresh-seed confirmation |
| `extra_refinement` | 282831–282836 | Additional refinement cohort; originally launched with the affine flag accidentally omitted; preserved |
| `affine` | 282831–282836 | Correctly executed no-refinement control; paired seed IDs with preceding row |
| `active_initialization` | 282851–282862 | Every initial breakpoint inside the input domain; post-pilot protocol on fresh seeds |
| `recovery` | Confirmation seeds | Fixed-feature, perturbation, recycling, and long-training follow-ups |
| `recovery_active` | Initially-active seeds | Same follow-ups |

Each producer writes its protocol and source snapshot before outcomes and refuses
to overwrite a protocol. The snapshots preserve executed code before subsequent
formatting/CLI additions. Recovery snapshots import the common numerical
functions from `explore_recruitable_workspace.py`; those formulas are unchanged
between the saved main-producer snapshots and the final source. The final
`verification.json` records current library versions, checked artifact hashes,
and numerical consistency checks.

`forecasts.jsonl` records checkpoint gradient power and scalar rate matching
before the corresponding correction continuation. The long-horizon research
predictions are qualitative statements in the protocols. These files should
not be presented as a new quantitative prospective width law. Ordinary-learning
outcomes were generated before the correction assays, so any relationship
between those two is retrospective.

The main producer saves training/held-out data, common demands, and initial,
paused, exposed, and all-data checkpoint parameters in `states_SEED.npz`.
Ordinary and branch losses are in separate JSONL files. The follow-up saves
loss trajectories and audit results; its late parameter states are not retained.

## Reproduction

From the repository root, choose new output directories for reruns. For example:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/explore_recruitable_workspace.py \
  --output docs/theory/recruitable_workspace/confirmation_rerun \
  --seeds 282811 282812 282813 282814 282815 282816 282817 282818 282819 282820 282821 282822

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/explore_recruitable_workspace.py \
  --output docs/theory/recruitable_workspace/active_rerun --active-init \
  --seeds 282851 282852 282853 282854 282855 282856 282857 282858 282859 282860 282861 282862

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/explore_recruitable_workspace.py \
  --output docs/theory/recruitable_workspace/affine_rerun --affine \
  --seeds 282831 282832 282833 282834 282835 282836

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/probe_workspace_recovery.py \
  --source docs/theory/recruitable_workspace/confirmation_rerun \
  --output docs/theory/recruitable_workspace/recovery_rerun

MPLCONFIGDIR=/private/tmp/neural-competition-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/studies/analyze_recruitable_workspace.py
MPLCONFIGDIR=/private/tmp/neural-competition-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/studies/audit_restart_evidence.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/verify_workspace_evidence.py
```

The analyzer reads the **saved original cohort directories** listed in the table;
producing a rerun under another name does not make the analyzer use it. Change
its explicit cohort mapping if analyzing a new collection. The original scripts
and older research artifacts outside this directory have not been rewritten.
