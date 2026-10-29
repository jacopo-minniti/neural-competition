# Feature-profile evidence

Read the [report](../../reports/feature_profile.md) and
[protocol](../../theory/feature_profile_protocol.md) before interpreting tables.

| File | Contents |
| --- | --- |
| `curves.csv.gz`, `endpoints.csv` | Every recorded model/step and 10k endpoint, including divergent starts |
| `summary.csv`, `paired_effects.csv` | Cohort means, medians, divergence counts, paired bootstrap intervals and fractions of mean width gap closed |
| `causal_controls.csv` | Paired early correction versus initial-only and moment-reset controls |
| `forecasts.csv`, `forecast_summary.csv` | Failed leave-setting-out final-loss forecast using ordinary step-1000 features |
| `extension.csv` | First five main starts continued to 100k, all interventions stopped at 10k |
| `audit.csv`, `repeat_audit.csv` | Loss reconstruction, source integrity, untreated-branch and repeated-cohort checks |
| `provenance.json`, `analysis.json`, `sources/` | Dataset/checkpoint/curve hashes, executed source snapshots, analysis hash and bootstrap definition |

Losses are **half-MSE**, normalized by training target variance. Multiply by
200 for MSE as a percentage of that variance. Lower loss is better. A paired
difference is intervention minus comparator, so negative favors the intervention.
`profile_minus_X` compares full correction with X. Gap closed is a ratio of mean
loss differences, not an estimate of unique causal mediation. Uncorrected 95%
bootstrap intervals use 5000 paired resamples, seed 845. Infinite failed losses
remain in tables; finite bootstrap intervals are not assigned to those contrasts.

`main`: seeds 100–119, original data 9027, 30°/(1,−3), no intercept.
`controls`: six settings, seeds 100–109, same data seed.
`initialization_controls` and `history_controls`: same main starts, two extra
arms each. Their repeated original arms match exactly and are **not additional
independent observations**. `data2`: data seed 9037, seeds 100–109. Extension
starts 100–104 overlap main. Forecasts use only main and controls, not repeated
arms or dataset follow-ups. Collection `validation` is excluded from analysis.

33 audited bundles contain 2970 executed 10k model slots plus 105 extended
slots, including intentional repeats. These counts are not independent sample
sizes. The scientific main comparison has 20 paired starts per optimizer.
The analyzer checks every final prediction against saved parameters and verifies
ordinary8/late identity through step 9000. Source snapshots are hash-addressed;
current code edits do not rewrite execution history. Raw checkpoints are local
and Git-ignored, as described in [data retention](../../data.md).
