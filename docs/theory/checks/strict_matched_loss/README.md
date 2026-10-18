# Strict interference at matched loss: recovered cohort diagnostics

This follow-up replays **90 original teacher-8/2027 runs**, each for 100,000
updates, to recover strict-harm statistics that the original summary tables
did not retain. It introduces no new initialization, minibatch sequence,
optimizer setting or independent replicate. The retained archive is read-only.
The [report](../../../reports/strict_matched_loss.md) interprets the results.

## Reproduction and provenance

From the repository root, with the existing analysis environment:

```sh
MPLCONFIGDIR=/tmp/neural-competition-matplotlib .venv/bin/python scripts/studies/analyze_strict_matched_loss.py
```

The [producer](../../../../scripts/studies/analyze_strict_matched_loss.py)
skips an existing condition cache. Use `--refresh` to replay it again, or
`--summarize-only` to rebuild comparisons and figures from the saved caches.
The compact caches are versioned here; original training inputs remain under
the ignored `runs/retained/` directory.

Conditions are Adam .001/.01 with zero outgoing initialization, Adam .001
with outgoing std .1/1, and momentum .1 with zero outgoing initialization.
Each contains the same five original seeds. Core conditions include widths
8/10/16/32; initialization controls include 8/16/32. All layers and hidden
biases train, using the original float32 optimizer steps. Diagnostics evaluate
the actual rounded parameter changes in float64.

Each condition's `.json` records input hashes, versions and verification:

- Regenerated initial parameters match every retained initial state bitwise.
- Parameters and optimizer moments match retained seeds 0/3 at updates
  10k, 50k and 100k bitwise. Minibatch generator states also match.
- Where seed-0 raw updates survive, all 14 original fields are compared on
  **every update**, without a numerical tolerance.
- Bin visit counts and strict frequencies match the original five-seed
  archived tables, including the full-budget frequencies.

These checks establish replay fidelity; replaying a run does not increase
the statistical sample size. `manifest.json` records the analysis convention
and producer/engine hashes. Tests independently check marginal versus joint
harm, benign overlap, sign-tolerance eligibility, and undefined denominators.

## Definitions

Let A_s be the isolated half-MSE change of neuron s, and J_rs the empirical
inner product of exact prediction changes. Updates are fixed throughout
the comparison; no gradient is recomputed after moving another neuron.

For each evaluation set, define ordered eligible pairs and strict events:

\[
E=\{(r,s):r\ne s,\ A_r<-\tau,\ A_s<-\tau\},\qquad
Q=\{(r,s)\in E:A_s+J_{rs}>\tau\},
\quad \tau=10^{-10}L_V.
\]

For a run/bin, sum across all its update visits:

\[
N=\sum_t |Q_t|,\quad D=\sum_t |E_t|,\quad
H_Q=\sum_t\sum_{(r,s)\in Q_t}(A_s+J_{rs}),
\]

\[
B_E=\sum_t\sum_{(r,s)\in E_t}(-A_s),\qquad
B_Q=\sum_t\sum_{(r,s)\in Q_t}(-A_s).
\]

- Strict frequency F=N/D counts events per eligible ordered pair.
- Aggregate severity S=H_Q/B_E measures marginal harm relative to all
  eligible standalone benefit; it includes both rarity and magnitude.
- Conditional event severity S_event=H_Q/B_Q compares harm with the affected
  updates' standalone benefit, restricted to strict events.
- H_Q/N is mean half-MSE harm per strict event; H_Q/visits and N/visits
  expose the total pair-harm burden and number of events per update.

If an eligible pair exists but no event occurs, F and S are zero. With no
events, conditional severity and harm/event are undefined, not zero. With
no eligible pairs, F and S are undefined. The code computes B_E and B_Q
directly with the eligibility masks; it does not approximate the denominator
using all negative A values, some of which can lie below the sign tolerance.

Pair harms overlap in neurons. H_Q is a diagnostic sum over alternative
two-neuron comparisons, **not an additive term in the full-network loss
budget**. In particular, an event need not make the combined two-neuron
update worse than the initial model.

## Matching, aggregation and files

Assignment uses the **pre-update MSE on the same fixed 256-example training
reference set** as the original study. It is not full-training MSE or test
MSE. Source-view outcomes evaluate the update on its 64-example source batch,
but are still grouped by reference MSE. There is no matching on source MSE.

The original primary bins are 12 equal logarithmic intervals from .02 to .2.
Five upper bins, from approximately .0766 to .2, are visited by every seed
at every width in all five conditions. The primary scalar comparison uses
these **same five bins**, taking ratios of sums within each seed/bin, then
giving bins equal weight, then seeds equal weight. This avoids weighting
the comparison toward narrow runs' longer plateau residence. It also avoids
selecting only narrow runs that reach lower losses.

Conditional event metrics are not averaged after dropping zero-event bins.
Their numerator and denominator are separately pooled over the shared range
in `shared_pooled` rows. This weights visits differently from the primary
equal-bin comparison and is labeled separately.

- `k8_*.csv.gz`: sufficient statistics for every seed/width/view, including
  original overview/primary/early bins, full budget and phase bands.
- `comparison_per_seed.csv`: common-bin means, pooled common-range ratios,
  and original full-budget ratios.
- `comparison_effects.csv`: paired width contrasts and 4,000 seed-bootstrap
  intervals (seed 741); differences use the saved fractional units.
- `loss_curves.csv`: bin-level means and pointwise intervals; `n_defined`
  distinguishes absent conditional-event values from observed zeros.
- `coverage.csv`: number of represented seeds and minimum/maximum visit
  counts. Main curves show a bin only where all five seeds contribute.
- `k8_*.json` and `manifest.json`: fidelity checks and provenance.

Uncertainty resamples **five seeds**, not millions of dependent pairs or
updates. Intervals are exploratory and pointwise. The matched range is
chosen from coverage of existing runs, not preregistered. Matching a scalar
loss does not match residual structure, representations, optimizer age or
moments, and therefore does not identify a causal mediation mechanism.
