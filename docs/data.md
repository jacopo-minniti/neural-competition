# Data retention

The current evidence directories retain analyses, protocols and, where needed,
forecasts, results and historical source snapshots. The selected learning-curve
example includes its initial state and observed outcomes. The ignored
`runs/natural_parity/` directory contains the local parity run used for its
analysis; it is regenerable and is not part of Git.

During the protection-model cleanup, redundant local bridge and older parity
runs were removed. Older power-law and probabilistic study arrays were removed
from the working tree, while their compact records were moved to the
[archive](../archive/README.md). Previously tracked arrays remain in Git history;
ignored local runs were never tracked. The local MNIST download cache under
`data/` is unrelated to the active protection studies and remains ignored.
