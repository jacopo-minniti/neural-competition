# Shared package

The maintained package contains the finite-update learning-ledger utilities used
by the options experiments. Study-specific dynamics live in `scripts/studies/`.

`parity.py` provides the [unequal-frequency parity dataset](../../docs/parity_dataset.md).

`frequency_pressure.py` implements orthogonal residual decompositions, signed
pressure diagnostics, exact finite gate/loss ledgers, and the conditional
Gaussian pressure forecast. See the [model](../../archive/docs/model/frequency_pressure.md)
and [ordinary-SGD validation](../../archive/docs/evidence/frequency_pressure/README.md).
