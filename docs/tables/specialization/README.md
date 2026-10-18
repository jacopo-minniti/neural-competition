# Specialization study evidence

See the [mechanism and derivation](../../theory/specialization_geometry.md),
[prospective protocol](../../theory/specialization_protocol.md), and
[training report and reproduction commands](../../reports/specialization.md).

| File | Unit and interpretation |
| --- | --- |
| `curves.csv.gz` | One model at one recorded update; all non-pilot collections |
| `endpoints.csv` | One model at its collection's final update; inspect `budget`/`completed` |
| `summary.csv` | Equal-seed means by collection, angle, optimizer and arm |
| `paired_contrasts.csv` | Treatment minus control with pointwise 95% paired-seed bootstrap intervals; negative test-loss differences favor treatment |
| `population.csv` | Derived one-pair population quadratic coefficients and innovation energy; not estimated from training |
| `audit.csv` | Per-file source-hash checks, initial perturbation size and final clean/copy prediction discrepancy |
| `provenance.json` | Every executed argument list, model list, producer/protocol hashes and raw curve-file checksum |
| `analysis.json` | Analysis source hash, bootstrap specification, repeat audit, and configuration counts |

`test_loss` and `train_loss` are half-MSE. Targets have unit training variance.
`fraction_remaining` divides test loss by the training-fitted shared-feature
baseline's test loss; it is undefined at zero angle. `success` means this
fraction is below .01, not that the entire target error is below 1% or that
interpolation is exact. `resolved` is a secondary geometric matching count:
distinct positive-output student directions within half of θ of each teacher
ray, using maximum bipartite matching. It is not the proposed theoretical object.

Collection names identify `main`, `replication20`, `bias`, `teacher2`,
`lower_rate`, `small_jitter`, `large_jitter`, `unstable_control` (75°), and
`longer` (100k updates). The 20-start replication repeats the main sweep's
first five starts. Perturbation controls repeat arms unaffected by jitter.
The longer collection extends existing starts and supplies no new independent
seeds. Never pool these rows as independent replicates. Intervals condition on
the given teacher/data and are not adjusted for multiple comparisons.

The default controlled perturbation has zero within-center mean, so a singleton
does not move at initialization. Exact unperturbed functions agree across
controlled arms; finite perturbations generally change predictions slightly.
The audit reports their largest RMS change from the common initial function.

Raw data and final parameters remain locally under `runs/specialization/`,
which is git-ignored. Versioned curves and metadata are sufficient to inspect
the reported results; reproducing final-parameter audits requires raw files
or rerunning the producer. No pilot evidence is included in the tables.
