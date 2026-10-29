# Maintained studies

| Study | Training | Analysis / validation |
| --- | --- | --- |
| Probabilistic option transitions | `fit_probabilistic_options.py`; `run_probability_validation.py` | `probabilistic_options.py` provides events and hazard models; `analyze_probability_validation.py` replays frozen probabilities and audits calibration/seed-cluster scores; [reproduction](../../docs/theory/probabilistic_options/README.md) |
| Functional implementation state | `run_refinement_hierarchy.py`; `probe_option_matched_implementations.py`; `probe_moment_boundary_closure.py` | `implementation_moments.py` provides functional moments and particles; `analyze_implementation_state.py` audits transitions and interventions; [reproduction](../../docs/theory/implementation_state/README.md) |
| Heterogeneous toy resources | `run_toy_resource_competition.py`; `recycle_toy_resources.py`; `probe_initial_option_coverage.py` | `audit_toy_option_events.py`; `audit_recycled_frozen_response.py`; `analyze_toy_resource_competition.py`; shared geometry in `toy_option_resources.py`; [reproduction](../../docs/theory/toy_resource_competition/README.md) |
| Finite-budget causal response | `train_response_checkpoints.py`; `probe_causal_response.py`; `probe_late_preservation.py` | `analyze_causal_response.py` audits fresh predictions, timing controls and failed targeted replication; `audit_correction_closure.py` separates curvature from response renewal; [reproduction](../../docs/theory/causal_response/README.md) |
| Local response model | `probe_response_feedback.py`; `forecast_response_feedback.py`; `probe_peer_response.py` | `analyze_response_feedback.py` verifies saved forecasts, source hashes, seed-level scores and model ablations; [protocols and reproduction](../../docs/theory/response_feedback/README.md) |
| Sign survival / spectral capacity | `run_sign_survival.py` | `analyze_sign_survival.py` audits endpoints and certificates; `trace_sign_crossings.py` attributes early crossing signals |
| Coupled readout / feature formation | `run_feature_profile.py`; `extend_feature_profile.py` | `analyze_feature_profile.py` reconstructs endpoints, audits repeated branches, and publishes all controls |
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
