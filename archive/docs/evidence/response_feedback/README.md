# Local response-model evidence

`analysis.json` retains the locked pilot and validation scores for full-gradient
and independent-minibatch intervention contrasts, physical option changes, and
negative scalar/linear ablations. `summary.png` and `model_comparison.png` are the
maintained figures. Design files preserve the cohort and forecast choices.

The bilinear feature–coefficient model predicts short local contrasts, while
physical ReLU evaluation is necessary for option changes. Raw checkpoints and
branch outputs are archived at commit `92b7904` and are reproducible with
`forecast_response_feedback.py`, `probe_response_feedback.py`, and
`analyze_response_feedback.py`.

Theory: [local dynamics](../../model/local_dynamics.md).
