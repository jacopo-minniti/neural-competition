# Follow-up: signed feature evidence and polarity capture

Exploratory follow-up, recorded after inspecting the residual-growth main sweep
and before executing this experiment. The conditional-Gram sign failed as a
predictor of strong width benefit: the largest ordinary-training failures were
in the opposite-output setting, where that conditional sign was positive.

The revised candidate is the signed initial evidence for a correct teacher
feature: sign(beta_j) E[y phi_j]. Opposing coactive target components can make
this negative, even with the feature direction already correct. A trainable
output intercept that absorbs the target mean replaces the raw Gram by its
centered covariance C=G-mu mu^T in the quasi-equilibrated-bias calculation.
This is a statement about competing target signals; direct harmful peer-update
competition is a stronger claim that it does not establish.

For four independent Gaussian feature-pair planes, the centered cross-pair
covariances vanish. Within a unit ReLU pair at full angle 2theta, v=1/2-1/(2pi)
and c=k(2theta)-1/(2pi). With target coefficients (1,-r), the weaker positive
feature's evidence is proportional to v-r*c. At theta=30 degrees and r=3 this
is negative; at theta=60 degrees it is positive. Giving an output bias the
initial training mean should remove global mean interference at 60 degrees,
but need not remove local covariance interference at 30 degrees.

Test ordinary random initialization, no copied or inserted hidden neurons.
Use the previous data/model seeds, both layers trained, widths 8/16/32 at
unchanged rates, and all three optimizers. Compare an ordinary learned output
bias initialized at zero versus initialized at the training target mean.
The bias is a scalar parameter, with its own ordinary optimizer update, not a
fixed subtraction that changes representability. The actual target remains
unchanged. Compare against the preceding no-output-bias runs. Retain output
polarity counts and exact raw/centered population feature evidence.

The predicted geometric difference may fail because ordinary trajectories
never form the assumed features, or because full-network adaptation overrides
the local sign. A successful width comparison alone is not evidence of this
mechanism. Check additional seeds, data, budgets and rates if the pattern is
strong; preserve null and reverse outcomes.
