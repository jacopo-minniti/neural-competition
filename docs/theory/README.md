# Literature-guided explanations and predictions

Research notes started 18 October 2026. These interpret the completed
[experiments](../README.md); they do not replace the reports or introduce new
training results. Paper results, deductions for our model, and hypotheses to
test are distinguished explicitly.

The questions guiding the search are specific:

- Why can wider Adam students fit better while paying more positive pair cost?
- Why does changing outgoing initialization rescue narrow Adam so strongly?
- Why does momentum's early transfer advantage survive shortening a step,
  whereas strict same-batch interference almost disappears?
- Why can two extra neurons improve reliability while a much larger expansion
  slows final refinement?

The working direction is to separate access to useful features, the optimizer's
allocation of motion, and finite-step costs. A measure should distinguish those
mechanisms and survive simple counterexamples; agreement across optimizers is a
prediction to test, not a criterion for choosing a definition after seeing data.

## What the search changes

The strongest revision is to separate **access to useful corrections** from
**the cost of combining the proposed corrections**. The existing C_plus is
exact accounting, but cannot be a necessary condition for better fitting:
even subdividing the same useful prediction change among identical neuron
copies increases it. Adam's observed reversal therefore need not contradict
a feature-access mechanism, although it does contradict a universal claim
through decreasing C_plus.

Recent work yields useful hypotheses and also assumption checks. In particular,
a sign-preservation argument from balanced small initialization does not apply
to our zero-output initialization. A gate-death explanation suggested by a
2026 preprint fails in two inspected narrow Adam cases. Those same cases
cannot be rescued just by optimally refitting the readout: their current
hidden features impose a high error floor.

| Question from our reports | Useful interpretation or prediction | Detailed note |
| --- | --- | --- |
| Why can Adam improve with more pair cost? | Shared useful corrections incur positive pair cost; inspect standalone harm, residual correction and avoidable allocation cost | [Pair cost and competition](pair_cost_and_competition.md) |
| Why does outgoing initialization matter so much? | Mean-driven early polarity and joint feature development; track feature floors and sign crossings | [Adam, initialization and features](adam_initialization_and_features.md) |
| Why do small expansions help, while large width can refine slowly? | Escape directions, gate changes and learning strength are different questions; a live split need not supply local descent here | [Width, escape and scaling](width_escape_and_scaling.md) |
| What could become a defensible alternative measure? | Readout floors, residual-weighted tangent rates, teacher-feature demand and gain from reweighting neuron steps | [Residual access and next tests](residual_access_and_tests.md) |

[Source selection and assumption audit](sources.md) records the precise
theorems/sections used, including why the generic Cooper/Nguyen guarantees
do not directly cover this regime. Six papers are used substantively;
other classics are retained only where their scope changes an interpretation.
The sources span 2018 through a September 2026 preprint. This is not a claim
that every relevant paper has been found or that the proposed metrics are novel.

## Recommended next decision

First compare failed and successful saved states using the readout floor,
residual access and actual optimizer response. Then check whether changing
relative neuron step weights helps beyond the best uniform step shrink,
using independent evaluation and exact forwards. These diagnostics distinguish
feature shortage, optimizer history and finite-step overshoot before choosing
a larger causal experiment.

Function-preserving expansions and early readout/bias rate interventions are
the most direct continuations if those checks support the corresponding
mechanisms. No proposed metric is guaranteed to improve with every optimizer;
the [test matrix](residual_access_and_tests.md#what-to-predict-before-committing-to-a-larger-experiment)
states outcomes that would favor different explanations.

The [strict-interference follow-up](../reports/strict_matched_loss.md) now
compares frequency and harm at matched reference loss. It uses verified
deterministic replays of the original five-seed teacher-8 cohorts, including
both Adam rates and output-initialization controls. It distinguishes per-pair
rates from total event counts and makes no new independent-replicate claim.

The [temporal peer-update experiment](../reports/temporal_branches.md) now tests
one downstream causal link directly: whether changing an earlier sender step
changes later standalone harm at other neurons. It finds phase-dependent
influence, especially beneficial responses in narrow teacher-8 momentum, but
no universal width ordering. This pulse experiment is separate from optimizing
neuron-step weights against the best uniform shrink.

## Small checks of existing saved states

The [objective-scale audit](../reports/effect_magnitude.md) evaluates practical
size and early predictive value using the existing full cohorts. It finds
material local SGD/momentum budget differences but no strong general early
pair-score predictor. It also derives subdivision-invariant residual-work
accounting and bounds a larger anti-corrective-motion phenomenon in Adam.

[The saved-state check](checks/README.md) inspects 144 retained teacher-8
states without training. It measures gate coverage and the least-squares
readout attainable with hidden features held fixed. It includes its producer,
CSV and input hashes. The retained seeds are purposively selected examples,
not a new cohort estimate.
