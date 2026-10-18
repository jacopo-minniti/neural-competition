# Residual potential and polarity evidence

See the [derivation](../../theory/residual_growth.md) and
[full report](../../reports/residual_growth.md). The two prospective records are
[conditional growth](../../theory/residual_growth_protocol.md) and
[polarity follow-up](../../theory/polarity_followup_protocol.md).

| File | Unit and meaning |
| --- | --- |
| `curves.csv.gz` | One model at one recorded checkpoint; all non-pilot settings |
| `endpoints.csv` | One model at its executed final budget; inspect `budget` |
| `summary.csv` | Equal-seed mean, median and maximum held-out half-MSE |
| `paired_effects.csv` | Treatment minus control at the final budget; negative favors treatment; pointwise 95% paired-seed intervals |
| `sign_width_interaction.csv` | Extra width benefit in (1,−3) versus (1,3), paired within seed; positive means stronger benefit with opposing outputs |
| `geometry.csv` | Exact population raw, centered and conditional signed feature evidence for every task; no trajectory fitting |
| `feature_probes.csv.gz` | Teacher-feature residual potentials at recorded states; exact Gaussian population values for bias-free models and empirical held-out values |
| `roles.csv` | Initial/final output-sign counts and required teacher-sign counts, with final loss |
| `audit.csv` | Source-version checks and maximum discrepancy from reconstructing final test losses from saved parameters |
| `provenance.json` | Executed arguments, source hashes and checksums of raw curves/final parameter files |
| `analysis.json` | Analysis code hash and bootstrap specification |
| `source_snapshots/` | Exact executed producer/module content retained across the balanced-sign change and code refactor |

Targets have unit training variance. Loss columns are half-MSE; multiply by
200 to express ordinary MSE as a percentage of that variance. Population
feature evidence is signed by the teacher coefficient, so a negative value
suppresses the correct output polarity at the specified feature/state.
It is not a validated scalar predictor of final width benefit.

`residual_growth/main` is the original five-seed geometry/output-sign sweep.
`replication20` includes its first five seeds again. `longer` extends those
same first five starts. `sign_scan`, `data2`, `hidden_bias`, `low_rate`, and
`larger_readout` are exploratory follow-ups. `polarity/main` contains learned
output-intercept controls; `polarity/balanced_signs` starts with exactly half
the outgoing weights of each sign. These share initialization/data streams
where applicable and must not be pooled as independent replicates.

The no-output-bias polarity arms repeat the matching residual-growth arms,
up to floating-point differences from the padded constant-feature encoding.
The balanced-sign producer is a later code version; the audit checks the
earlier version against the immutable source snapshot when its current source
hash differs. It does not silently waive source checks.

`roles.csv` excludes the scalar output intercept. Four positive and four
negative hidden units are necessary for exact population representation of
the opposing-output targets with eight distinct hyperplanes. Enough units
of both signs is not sufficient for fitting, and the condition is not an
approximation-error bound on a finite dataset.

The `peer_reversed_fraction` diagnostic compares a neuron's current signed
amplitude growth with growth holding its own contribution fixed but removing
other predictions. Both inequalities use tolerance 1e−5. This counterfactual
removal is only a read-only calculation, not a training intervention. It can
count appropriate suppression near a good fit and is not the primary object.

Analysis: `scripts/studies/analyze_residual_growth.py`; producers:
`scripts/studies/run_residual_growth.py` and `scripts/studies/run_polarity.py`.
Raw datasets and final `.pt` states are local under git-ignored `runs/`.
All reported tables and figures are versioned. Reproduction commands are in
the report. Bootstrap intervals resample paired training seeds, not neuron
pairs or test examples, and are exploratory/unadjusted for multiple tests.
