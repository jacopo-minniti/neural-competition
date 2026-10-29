# Follow-up protocol: coupled readout learning and feature formation

Recorded after the exploratory screening on original seeds 0–4, before the
fresh-seed experiments below (18 October 2026). This extends the residual-growth
study; it does not replace its negative sign/bias results.

The screening found that full readout least-squares updates often rescue narrow
opposing-output models, whereas mean/affine profiling and rate changes are mixed.
It also found that profiled feature loss generally decreases on failing ordinary
trajectories: a simple "feature loss increases" criterion is not supported.
The main SGD pilot used a float32 normalization constant with float64 parameters;
all subsequent screening and the following study use the original float64 norm.
The pilot is exploratory, not part of fresh-seed estimates.

## Object and candidate mechanism

For current hidden features Phi(W), define R(W)=min_a ||y−Phi(W)a||²/(2n).
Then L(W,a)=R(W)+E(W,a), where E=||Phi(W)(a−a*)||²/(2n).
The conditional signal for coefficient r removes the span of every other
*current* feature, rather than an assumed teacher-center span. Its sign and
conditional feature variance determine a*_r whenever that variance is nonzero.

The exact full-gradient feature drift is −∇_W L; profiled feature loss changes
at rate −<∇R,∇_W L>. That identity does not imply that this rate explains eventual
width benefit. Competition here means that overlapping features' coupled
coefficient learning changes the features subsequently acquired. A stronger
harmful-peer attribution remains to be established.

Prediction: removing coupled readout lag during feature formation improves
narrow final held-out loss beyond a terminal readout refit. A diagonal readout
step and faster readout rate test whether a generic correction/rate change is
enough. Equal-duration early/late interventions test formation-time dependence.
Failures of these predictions will be reported, not used to redefine the test.

## Fresh comparisons, fixed before execution

Same tasks, 4096/4096 samples, normal initialization, 1/sqrt(width), rates and
minibatch streams as the residual-growth study; seeds 100–119 in the main
30-degree/(1,−3)/no-intercept task. Ten fresh seeds 100–109 in controls:
30/(1,−1), 60/(1,−3), 30/(1,3), mean-initialized output intercept at 30 and 60,
and the new 45/(1,−2) geometry. Three optimizers; 10,000 ordinary update steps.
No neuron copying, teacher directions in training, or assigned neuron groups.
Control seeds overlap across settings and must not be pooled as independent.

Arms: ordinary width 8; ordinary width 32; width 8 with full-training readout
least-squares correction every 10 steps; the same cadence with one diagonal
preconditioned step and exact line search; full correction only in steps 0–999;
full correction only in steps 9000–9999; and ordinary width 8 with 10x readout
rate. Both layers otherwise train. Readout optimizer moments reset after *both*
full and diagonal corrections; incoming moments remain. Optimizer update counts
are unchanged, so Adam bias correction follows elapsed training steps.
Corrections cost extra full-data work: this is a mechanism test, not a claim of
compute-matched training efficiency. Report losses before any terminal refit,
plus a common terminal-refit diagnostic for all arms.

For the intercept controls, retain the original constant-feature encoding and
readout rate/epsilon convention. Profiling optimizes that intercept jointly.
The intervention's first correction changes the prediction: report its effect
and do not describe it as function-preserving. The late arm shares the ordinary
trajectory until step 9000; the early arm gets no corrections after step 1000.

Primary outcomes: final held-out half-MSE, paired difference from ordinary
width 8, fraction of the ordinary narrow–wide mean gap closed, comparisons with
diagonal/fast-rate controls, and terminal profiled loss. Do not interpret a
fraction closed as a uniquely identified fraction causally mediated by competition.
Use paired seed bootstrap intervals; retain all starts and unsuccessful arms.

Record loss/profiled loss, readout condition, conditional sign availability,
and actual-gradient/profile-gradient alignment at steps 0,10,100,1000,3000,
9000,10000. For prospective usefulness, compare early ordinary loss alone with
adding profiled loss/gap in leave-setting-out prediction of log final loss;
this is exploratory with few distinct settings, not a universal sign theorem.

## Follow-up registered after seeing the fresh main comparison

The first 20-seed results showed a large benefit from early correction and a
much weaker benefit from the equal-duration late arm. Before interpreting that
as ongoing coupling, add `initial_only` (one full refit at step zero) and
`reset_only` (readout moment resets every ten steps during the first 1000, no
coefficient correction). Repeat the main setting on the SAME 20 seeds with
`--initial-controls --collection initialization_controls`. Repeated original
arms are an implementation consistency check, not new independent evidence.
These controls were chosen after the main result; their status is exploratory.

The reset-only control substantially improved Adam, unlike SGD/momentum. Add
`profile_keep` and `early_keep`, identical coefficient corrections but retaining
all optimizer moments, on the same 20 starts (`--history-controls`, collection
`history_controls`). This follow-up separates readout changes from explicit
moment erasure. It was specified after the initialization/reset control results;
no new independent starts are implied by repeating the original arms.

Final robustness follow-ups, chosen after the controls: repeat the main geometry
on data seed 9037 with ten starts 100–109; extend the preselected first five
main starts (100–104) to 100k updates. In the extension all interventions stop
at 10k, including the faster readout rate; every arm then uses the ordinary
optimizer/rates with its existing state. These are extensions of the same
starts, not independent replications or a preplanned confirmatory cohort.
