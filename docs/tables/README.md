# Numerical evidence

- [Coupled-mode forecasts](coupled_modes/README.md): predicted and observed MNIST
  branch loss contrasts, failed pilots, horizon limits and source audits.

- [Sign survival](sign_survival/README.md): targeted sign-crossing interventions,
  paired robustness, population error certificates and crossing attribution.

- [Feature profiling](feature_profile/README.md): paired interventions, full
  controls, failed forecasts, long continuations, source provenance and audits.
- [Residual growth](residual_growth/README.md): complete current cohorts, seed
  curves, feature probes, role counts, uncertainty, source versions and audits.
- [MNIST](mnist/README.md): full main/control cohorts, matched-state diagnostics,
  exact budget summaries, dataset indices, tuning and provenance.
- [History](history/README.md): compact evidence supporting the consolidated
  history note. Original detailed arrays and older plots are no longer active.

CSV and gzip CSV files are directly readable with pandas. Use
`float_precision="round_trip"` when exact decimal recovery matters. Tables
include unsuccessful settings and seeds; they are not only successful examples.
Raw local checkpoints are described separately in [data retention](../data.md).
