# Numerical package

| Module | Responsibility |
| --- | --- |
| `residual_growth.py` | Gaussian ReLU kernel, conditional Gram matrix, residual potential |
| `tasks.py` | Deterministic correlated-pair Gaussian teacher construction and normalization |
| `training.py` | Shared batched scalar ReLU prediction, analytic gradient and joint-layer SGD/momentum/Adam loop |
| `mnist.py` | MNIST loading/splits, vector-output gradients/optimizers, and diagnostic calculations |
| `model.py`, `config.py` | Generic scalar PyTorch model and experiment configuration |
| `experiment.py`, `diagnostics.py`, `optimizers.py` | Generic training/checkpoint engine, probes and optional Muon optimizer |

The scalar batched engine and MNIST engine deliberately use distinct layouts:
scalar studies batch independent models with padded neurons; MNIST uses a
single model with ten outgoing coefficients per neuron. Do not force these
into a generic optimizer abstraction that hides normalization or state semantics.
The generic PyTorch model is also an independent reference for numerical checks.

Study orchestration belongs in `scripts/studies/`, not in this package.
Retired allocation, splitting, distribution and temporal-analysis modules were
removed; exact historical code remains in Git commit `b03c53f`. In particular,
the active training loop no longer lives in the retired specialization module.
