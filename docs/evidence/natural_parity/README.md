# Standardized bipolar parity

The parity experiment tests dominant-direction estimation and completion
reserve on a target with interacting binary factors. Inputs have unequal
occurrence probabilities and are standardized; targets are bipolar parity.
The full squared-loss objective trains both layers and the output intercept.
The primary four-factor condition uses 128 independent initializations and
widths 4, 8, 16 and 32. Response measurements use fixed one-factor contrasts
with equal configuration weights, not population-weighted loss.

At the initial checkpoint, the leading incoming-gradient direction is compared
with the known empty-state direction. Mean absolute alignment ranges from 0.738
to 0.935 across widths and mean coherence exceeds 0.71. The first phase ends
when half the initial empty-state error is corrected. The SVD-based completion
forecast predicts one half-response loss at width 4 while eight of 512 contrasts
are observed; the known-direction forecast also predicts one. No half-response
losses occur at the wider widths in this sample. These zero counts are not a
protection guarantee, and coherent updates do not certify the completion model.

Mean final population loss falls from 0.0826 at width 4 to 0.0520 at width 32.
The comparison does not establish that response weakening caused the loss gap.
The terminal checkpoint is not exactly step 2,500: the follow-up budget is
adjusted using mean phase length. See [analysis.json](analysis.json) for all
conditions, uncertainty and outcomes, and [the report](../../../report.tex) for
the interpretation.

Run the producer and analyzer from the repository root:

```sh
.venv/bin/python scripts/studies/run_natural_parity.py --output runs/natural_parity
MPLCONFIGDIR=/tmp/natural_parity_mpl .venv/bin/python scripts/studies/analyze_natural_parity.py --input runs/natural_parity/results.json --output docs/evidence/natural_parity
```

The producer writes to its output location; preserve previous results before
rerunning. Local raw outcomes remain in the ignored `runs/natural_parity/`
directory on this machine; the analysis and figure are retained here.
