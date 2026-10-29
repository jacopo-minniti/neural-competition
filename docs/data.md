# Data retention

The maintained repository stores compact analysis records, figures, protocols,
and source snapshots under `docs/evidence/`. Large model checkpoints and raw
trajectories are not required for ordinary inspection or tests.

Official compressed MNIST archives remain under `data/mnist/raw/`. The prepared
tensor cache is regenerable and intentionally absent. Ignored legacy `runs/`,
scratch output, and temporary build directories were permanently removed during
the 29 October curation.

All tracked pre-curation files—including later implementation-distribution and
probabilistic-transition work—are recoverable from commit `92b7904`. Large
ignored runs were never part of Git and must be regenerated from their archived
protocols if needed.
