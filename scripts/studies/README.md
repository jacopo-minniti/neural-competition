# Maintained studies

| Study | Training | Analysis / validation |
| --- | --- | --- |
| Signed residual growth | `run_residual_growth.py` | `analyze_residual_growth.py` reconstructs all final losses and checks executed source versions |
| Output intercept / balanced signs | `run_polarity.py` | Included in the residual-growth analyzer |
| MNIST | `run_mnist.py` | `analyze_mnist.py` publishes seed results and plots |
| MNIST controls | `check_mnist.py` (rate controls and matched states) | `audit_mnist.py` (dense replay); `probe_mnist_objective.py` (full objective) |

See [reproduction](../../docs/reproduction.md) for order and arguments. Defaults
can launch long experiments. Existing completion markers cause producers to
skip work; use fresh collections/output directories for changed protocols.
Do not treat running a completed producer as rerunning or validating training.

Historical executed sources are retained separately in the study evidence.
Refactoring does not rewrite old manifest hashes. The current source can be
used for new runs, while audits resolve older hashes to exact snapshots.
