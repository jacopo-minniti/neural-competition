# Coupled-mode loss forecasts: prospective local test

23 October 2026. Recorded before executing this probe. This tests a quantitative
candidate definition, not a declaration that it explains the previous width gap.

Use the existing MNIST squared-loss SGD checkpoints at update 429, widths 16 and
128, seeds 0–4. Pick a fixed 256-example training subset and independent 1024-example
validation subset using seed 231023. No tuning on the validation split.

For current hidden features Phi, D=diag(Phi^T Phi/n), Z=Phi D^-1/2, and C=Z^T Z/n.
Discard exactly dead feature columns. Normalize the readout by B=D^1/2 A. On the
current feature span, fit B* and decompose initial residual energy into C's modes.
With eta=.5/max(1,lambda_max(C)), predict the loss after k steps of readout
gradient descent in B using (1-eta lambda)^(2k). A joint-curvature comparator
uses C^dagger and has contraction (1-eta)^(2k) on the identifiable span.

The prediction for coupling damage is L_diagonal(k)-L_joint-curvature(k), on
the SAME actual objective and initial state. Report positive slow-mode cost and
negative fast-mode benefit separately. Eigenvalues alone do not predict damage:
residual energy in each mode is required. The comparator has extra computation
and changes the optimizer metric. This is a precisely specified optimization
contrast, not unique causal attribution or a compute-matched training comparison.

Forecast at normalized times .1,.5,2 (rounded upward to k integer steps).
Verify fixed-feature forecasts with iterative matrix-gradient updates. Then run
two live-feature branches from the same state, using the same frozen readout
preconditioners and ordinary hidden-layer gradient descent with step eta. Compare
their observed loss contrast to the frozen forecast; include a zero-damage
prediction baseline. All parameters are float64. These continuations reset
optimizer state and use full gradients on the fixed subset; they are not a replay
of the original SGD/minibatch optimizer. Save predictions before continuations.

This is a small real-data mechanism test. It does not measure whole-dataset MNIST
training efficiency, establish transfer to Adam/cross-entropy, or give a full-run
generalization forecast. Validation contrast tests whether a training-subset
benefit persists outside that subset. A failed forecast must remain visible.

## Follow-ups after the initial probe

The 256-example probe had an unstable live-feature start and worse validation
loss after joint-curvature updates. Repeat on 4096 examples, forecasting validation
loss from initial validation features separately. Keep the pilot, including its
unstable start. This change was chosen after seeing the pilot.

The normalized comparison alters the readout optimizer. Add an ordinary-metric
comparison: retain ordinary Euclidean gradient descent as the baseline. In
normalized coordinates define B=D and A=C^(1/2) B C^(1/2). Compare actual
loss-coordinate dynamics q'=-Aq with q'=-Bq. The latter removes C while keeping
the block-speed matrix B; their traces match because diag(C)=1. Implement its
corresponding positive-semidefinite readout preconditioner. Use readout and
hidden rate 1, or reduce both if the frozen readout matrices require it for
stability. Predict after 1,10,100 steps, before running branches. This is still
a full-gradient subset continuation, not original minibatch training. Evaluate
both training and validation contrasts, and retain forecasts with wrong signs.

The ordinary-metric epoch-one results showed that readout-only forecasts can
get even short-horizon signs wrong once hidden gradients are included. Add a
full frozen-Jacobian forecast including incoming weights and biases, using the
same actual and counterfactual optimizers. Test it prospectively at checkpoint
2145 (five epochs) on the same five starts and input subsets, at 1,10,100 steps.
This is a new training phase, not new independent initializations. Compute and
save both forecasts before either nonlinear continuation. No fitted coefficients
or outcome-based horizon selection. The previous failed forecasts remain retained.
