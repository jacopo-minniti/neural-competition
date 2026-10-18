# Versioned result tables

The [residual-growth tables](residual_growth/README.md) retain the ordinary
random-initialization, signed-output and polarity-control studies, with
population geometry, all seed curves, negative predictions, and source audits.

The [specialization tables](specialization/README.md) contain the new
mechanism-derived training experiments, including full seed curves, paired
contrasts, source hashes, and long-budget catch-up results.

The [allocation tables](allocation/README.md) contain the new residual-work,
cross-fitted neuron-step allocation, calibration-size, matched-target and
full-cohort readout-refit diagnostics, including negative results and audits.

The [MNIST tables](mnist/README.md) contain the complete real-data width study,
including exact main G/H budgets, sampled pair costs, matched-loss diagnostics,
learning-rate controls and full-training-objective probes.

These preserve the reported cohorts before raw-data pruning. Rows have different units (seeds, comparisons or summary metrics); use each report and the saved count columns. Historical optimizer labels differ across collections: see [the protocol](../protocol.md).

Dense per-seed/per-batch tables and curves are separately preserved under `runs/retained/aggregates/`. The selected raw checkpoints are not the cohort used for these summaries.

| File | Rows |
| --- | ---: |
| [adam_cross_batch_effects.csv](adam_cross_batch_effects.csv) | 384 |
| [adam_cross_batch_per_seed.csv](adam_cross_batch_per_seed.csv) | 180 |
| [adam_cross_batch_summary.csv](adam_cross_batch_summary.csv) | 36 |
| [cumulative_interference_effects.csv](cumulative_interference_effects.csv) | 52 |
| [cumulative_interference_fit.csv](cumulative_interference_fit.csv) | 4 |
| [cumulative_interference_seeds.csv](cumulative_interference_seeds.csv) | 80 |
| [momentum_actual_step_effects.csv](momentum_actual_step_effects.csv) | 300 |
| [momentum_actual_step_summary.csv](momentum_actual_step_summary.csv) | 20 |
| [momentum_gradient_effects.csv](momentum_gradient_effects.csv) | 275 |
| [momentum_gradient_summary.csv](momentum_gradient_summary.csv) | 10 |
| [momentum_neuron_summary.csv](momentum_neuron_summary.csv) | 56 |
| [neuron_gradient_effects.csv](neuron_gradient_effects.csv) | 126 |
| [neuron_gradient_summary.csv](neuron_gradient_summary.csv) | 6 |
| [neuron_prediction_summary.csv](neuron_prediction_summary.csv) | 24 |
| [pair_budget_effects.csv](pair_budget_effects.csv) | 300 |
| [pair_budget_summary.csv](pair_budget_summary.csv) | 60 |
| [pair_interference_effects.csv](pair_interference_effects.csv) | 70 |
| [pair_interference_summary.csv](pair_interference_summary.csv) | 28 |
| [pier_agreement_summary.csv](pier_agreement_summary.csv) | 18 |
| [pier_continuation_summary.csv](pier_continuation_summary.csv) | 4 |
| [pier_early_effects.csv](pier_early_effects.csv) | 144 |
| [pier_outcomes_summary.csv](pier_outcomes_summary.csv) | 24 |
| [pier_tail_effects.csv](pier_tail_effects.csv) | 18 |
| [pier_tail_summary.csv](pier_tail_summary.csv) | 12 |
| [robustness_effects.csv](robustness_effects.csv) | 704 |
| [robustness_neuron_structure.csv](robustness_neuron_structure.csv) | 205 |
| [robustness_per_seed_results.csv](robustness_per_seed_results.csv) | 410 |
| [robustness_summary.csv](robustness_summary.csv) | 82 |
| [teacher8_history.csv](teacher8_history.csv) | 198 |

[index.json](index.json) records schemas, sources and SHA-256 hashes.

The [objective-scale magnitude and forecasting outputs](../theory/checks/effect_magnitude/README.md)
are separately versioned with their producer and input hashes. They include
per-seed cumulative budgets, matched-loss retention, direct objective probes
and early-feature/held-out-outcome tables.

The later [matched-loss strict-interference tables](../theory/checks/strict_matched_loss/README.md)
are versioned separately with their replay audits. They recover additional
harm statistics for original teacher-8 runs and do not alter the historical
tables or this original index.
