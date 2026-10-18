# Figures from the completed studies

These PNG figures are versioned with the reports and no longer depend on deleted run folders. Intervals are exploratory and pointwise; see the associated reports.

## Distributions

- [agreement sgd 1 w8 w16](distributions/agreement_sgd_1_w8_w16.png)
- [evolution adam 0001](distributions/evolution_adam_0001.png)
- [evolution momentum 01](distributions/evolution_momentum_01.png)
- [evolution sgd 1](distributions/evolution_sgd_1.png)
- [finite step intervention](distributions/finite_step_intervention.png)
- [gradient distributions 0.01](distributions/gradient_distributions_0.01.png)
- [gradient distributions 0.2](distributions/gradient_distributions_0.2.png)
- [gradient moment step distributions](distributions/gradient_moment_step_distributions.png)
- [gradient tails 0.01](distributions/gradient_tails_0.01.png)
- [gradient tails 0.2](distributions/gradient_tails_0.2.png)
- [stability continuation](distributions/stability_continuation.png)
- [stability pilot](distributions/stability_pilot.png)
- [training](distributions/training.png)
- [width statistics 0.01](distributions/width_statistics_0.01.png)
- [width statistics 0.2](distributions/width_statistics_0.2.png)

## Momentum

- [momentum agreement](momentum/momentum_agreement.png)
- [momentum gradients](momentum/momentum_gradients.png)
- [momentum neurons](momentum/momentum_neurons.png)
- [momentum step sensitivity](momentum/momentum_step_sensitivity.png)

## Neurons

- [neuron PN](neurons/neuron_PN.png)
- [neuron prediction interactions](neurons/neuron_prediction_interactions.png)
- [neuron step sensitivity](neurons/neuron_step_sensitivity.png)

## Pairs

- [pair interference](pairs/pair_interference.png)
- [pair step sensitivity](pairs/pair_step_sensitivity.png)

## Trajectory

- [comparison summary](trajectory/comparison_summary.png)
- [momentum cumulative](trajectory/momentum_cumulative.png)
- [momentum equal gains](trajectory/momentum_equal_gains.png)
- [momentum local windows](trajectory/momentum_local_windows.png)
- [momentum matched loss](trajectory/momentum_matched_loss.png)
- [momentum source cumulative](trajectory/momentum_source_cumulative.png)
- [sgd cumulative](trajectory/sgd_cumulative.png)
- [sgd equal gains](trajectory/sgd_equal_gains.png)
- [sgd local windows](trajectory/sgd_local_windows.png)
- [sgd matched loss](trajectory/sgd_matched_loss.png)
- [sgd source cumulative](trajectory/sgd_source_cumulative.png)

## Robustness

- [adam initialization control](robustness/adam_initialization_control.png)
- [k16 teacher2027 adam 001 trajectory](robustness/k16_teacher2027_adam_001_trajectory.png)
- [k16 teacher2027 adam 01 trajectory](robustness/k16_teacher2027_adam_01_trajectory.png)
- [k16 teacher2027 momentum 01 trajectory](robustness/k16_teacher2027_momentum_01_trajectory.png)
- [k16 teacher2037 adam 001 trajectory](robustness/k16_teacher2037_adam_001_trajectory.png)
- [k16 teacher2037 momentum 01 trajectory](robustness/k16_teacher2037_momentum_01_trajectory.png)
- [k8 teacher2027 adam 001 trajectory](robustness/k8_teacher2027_adam_001_trajectory.png)
- [k8 teacher2027 adam 01 trajectory](robustness/k8_teacher2027_adam_01_trajectory.png)
- [k8 teacher2027 init01 adam 001 trajectory](robustness/k8_teacher2027_init01_adam_001_trajectory.png)
- [k8 teacher2027 init1 adam 001 trajectory](robustness/k8_teacher2027_init1_adam_001_trajectory.png)
- [k8 teacher2027 momentum 003 trajectory](robustness/k8_teacher2027_momentum_003_trajectory.png)
- [k8 teacher2027 momentum 01 trajectory](robustness/k8_teacher2027_momentum_01_trajectory.png)
- [k8 teacher2037 adam 001 trajectory](robustness/k8_teacher2037_adam_001_trajectory.png)
- [k8 teacher2037 momentum 01 trajectory](robustness/k8_teacher2037_momentum_01_trajectory.png)
- [robustness controls effects reference](robustness/robustness_controls_effects_reference.png)
- [robustness controls effects source](robustness/robustness_controls_effects_source.png)
- [robustness core effects reference](robustness/robustness_core_effects_reference.png)
- [robustness core effects reference compact](robustness/robustness_core_effects_reference_compact.png)
- [robustness core effects source](robustness/robustness_core_effects_source.png)
- [robustness fitting](robustness/robustness_fitting.png)
- [robustness loss coverage](robustness/robustness_loss_coverage.png)
- [robustness normalized controls](robustness/robustness_normalized_controls.png)
- [robustness raw penalty](robustness/robustness_raw_penalty.png)
- [robustness strict pair frequency](robustness/robustness_strict_pair_frequency.png)
- [robustness unfavorable pair frequency](robustness/robustness_unfavorable_pair_frequency.png)

## Adam Cross Batch

- [adam cross batch matched loss](adam_cross_batch/adam_cross_batch_matched_loss.png)
- [adam cross batch trajectory](adam_cross_batch/adam_cross_batch_trajectory.png)

- [Broad pair cost](pairs/broad_pair_cost.png) — broad unfavorable cost and frequency, distinct from strict events.

## Objective-scale effect magnitude

See the [magnitude and forecasting report](../reports/effect_magnitude.md).
The prediction-error percentages are not percentages of loss explained.
Vector PDFs are saved beside the PNGs.

- [SGD/momentum objective efficiency](effect_magnitude/trajectory_matched_efficiency.png)
- [Adam and control objective efficiency](effect_magnitude/robustness_matched_efficiency.png)
- [Early metrics: incremental forecasting value](effect_magnitude/predictive_increment.png)

## Running objective budgets

Exact every-update prefix sums in raw half-MSE units, plotted against training
updates without loss bins or division by G. See the
[report](../reports/cumulative_budget.md); vector PDFs accompany each PNG.

- [Adam, both learning rates](cumulative_budget/adam_cumulative_budget.png)
- [SGD and momentum, 200k updates](cumulative_budget/sgd_momentum_cumulative_budget.png)
- [Momentum, 100k updates including width 32](cumulative_budget/momentum_100k_cumulative_budget.png)
- [Adam outgoing-initialization controls](cumulative_budget/adam_initialization_cumulative_budget.png)

## Strict interference at matched loss

Verified replays of the original five-seed teacher-8 cohorts; see the
[report](../reports/strict_matched_loss.md) for matching, severity denominators,
raw event counts and uncertainty. Vector PDFs are saved beside these PNGs.

- [Default Adam, reference evaluation](strict_matched_loss/adam_default_matched_loss_reference.png)
- [Default Adam, source evaluation](strict_matched_loss/adam_default_matched_loss_source.png)
- [Adam including initialization controls, reference evaluation](strict_matched_loss/adam_matched_loss_reference.png)
- [Adam including initialization controls, source evaluation](strict_matched_loss/adam_matched_loss_source.png)

## Temporal peer-update interventions

The [report](../reports/temporal_branches.md) defines the causal contrasts and
normalizations. Positive peer benefit means the ordinary sender update reduced
later recipient harm relative to damping that update. Widths receive matched
prediction-space pulses; intervals resample original seeds. PDFs accompany PNGs.

- [Held-out effects, all sampled checkpoints](temporal_branches/peer_harm_heldout_all_sampled.png)
- [Held-out effects relative to initial pulse impact](temporal_branches/peer_harm_heldout_all_sampled_pulse.png)
- [Held-out effects, first sampled checkpoint in the shared loss band](temporal_branches/peer_harm_heldout_first_shared_loss.png)
- [Training-reference effects](temporal_branches/peer_harm_reference_all_sampled.png)
- [Lag dependence on the same five checkpoints](temporal_branches/peer_harm_lags_heldout.png)
- [Fixed-20% pulse sensitivity](temporal_branches/fixed_fraction/peer_harm_heldout_all_sampled_pulse.png)
