# Validation

Run `uv run pytest -q` from the repository root. Tests use small synthetic inputs
and do not require ignored raw runs or a MNIST download.

- `test_training.py`: shared scalar analytic gradients, padded models, both
  layers, bias freezing, optimizer history and chunked continuation versus PyTorch.
- `test_residual_growth.py`: Gaussian kernel/conditional geometry, exact signed
  amplitude growth and scalar-intercept update convention.
- `test_mnist.py`: vector gradients/optimizers versus PyTorch and finite-update
  interaction identities, including cross-entropy.
- `test_measurements.py`: generic training/checkpoint integration and probes.

Tests for retired pipelines are recoverable with those pipelines at `b03c53f`.
Their removal does not mean the old scientific claims have been revalidated by
the smaller active suite. Source-hash and local-file audits are separate commands
in [reproduction](../docs/reproduction.md).
