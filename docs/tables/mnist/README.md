# MNIST study tables

These are the complete computed cohorts for the [MNIST report](../../reports/mnist.md),
not a selected subset of example seeds. Five paired main seeds are used throughout.
Run `scripts/studies/analyze_mnist.py` to regenerate from the ignored raw study.
Gzip CSV files are ordinary pandas-readable tables. For exact decimal round trips,
use `pd.read_csv(path, float_precision="round_trip")`.

- `final_per_seed`, `final_summary`: fitting, cumulative budgets and their ratios.
  `accounting=every_update` identifies exact primary G/H/C_all sums; controls use
  sampled updates. C_plus remains time-sampled in every collection.
- `final_paired_per_seed`, `final_paired_summary`: width differences versus 16.
  The summary's `statistic` names the physical measurement; `metric` distinguishes
  absolute differences from relative changes. Intervals resample paired seeds.
- `epochs`, `transfers`, `step_controls`: original epoch evaluations, same-epoch
  batch probes, full/1% fresh-source controls. These are different from the
  matched-loss follow-up tables.
- `diagnostics`, `budgets`: original sampled diagnostic rows and cumulative curves.
  Primary G/H/C_all curve columns are replaced by exact sums; retained sampling
  SE columns refer to the original estimators, not error in the exact sums.
- `exact_cumulative_curves`, `sampling_audit`, `dense_full.json`: complete-cohort
  every-update accounting, the sampled-versus-exact comparison and replay audits.
  `dense_audit.json` records the initial twelve-case sampling check.
- `matched_coverage`, `replay_audits`: accepted and excluded target states, plus
  bitwise original-checkpoint comparisons.
- `matched_raw_interactions`: every source-level pair and isolated-effect aggregate;
  exact within the fixed evaluation set, with inverse-inclusion pair weights for CE.
- `matched_interactions_per_seed`, `matched_interactions_summary` and paired
  counterparts: sum the 16 sources within each seed before forming ratios.
- `matched_transfers_per_seed`, `matched_transfers_summary` and paired counterparts:
  64 source/target probe batches at each accepted matched state. The 4,032 ordered
  pairs share gradients and must not be treated as independent uncertainty units.
- `full_objective_probes`, `full_objective_per_seed`, `full_objective_summary` and
  paired counterparts: first eight matched source draws evaluated against all
  55,000 training examples, with logit-work/curvature accounting.
- `audits`, `audit.json`, `provenance.json`: identities, coverage, dataset and source hashes.
- `cohort_registry`: all 296 tuning/main/control starts, configurations and completion hashes.
  `tuning.csv` retains the complete independent rate grid. `data_indices.npz`
  preserves the exact train/validation split and reference indices into it.
  `sources/` preserves the numerical and analysis producers with their tests.

Collections `main`, `lower_lr` and `width_tuned` are newly trained runs.
`width_tuned_complete` combines twenty individually tuned replacements with
forty unchanged main cases; it is not another sixty independent runs.
`reference` means the fixed 256-image training probe. Older output field
`heldout` in same-epoch controls means the 256-image validation probe; matched
tables name it `validation`. That split also selected rates and is not an
untouched test set. Official test accuracy is evaluated only after full runs.

The first matched diagnostic pass used independent probes across widths. It was
superseded by common probes, which reduce comparison noise; it is not used in
the reported estimates. Its raw cache and source snapshot remain under
`runs/mnist/matched_initial_independent_probes` for execution provenance.
