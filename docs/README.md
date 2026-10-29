# Evidence overview

For the maintained options framework, three current test tasks, and research
priorities, start with [the research context](research_context.md).

**Quantitative definition, 23 October:** [Coupling-induced optimization damage](theory/competition_definition.md)
specifies a counterfactual and a loss forecast. [MNIST probes](reports/coupled_modes.md)
support short-horizon predictions with the full Jacobian, while retaining failures
at longer horizons and counterexamples to a monotone width explanation.

Our [objective](../OBJECTIVE.md) is to identify a conditional, consequential
competition mechanism during training. The evidence currently supports strong
width effects in some settings and several local interaction effects. It does
not establish a general causal explanation of the width advantage.

**21 October follow-up:** [Preventing sign conversion](reports/sign_survival.md)
rescues narrow feature learning across all three optimizers. A new
[Gaussian spectral bound](theory/signed_capacity.md) quantifies the error forced
by missing signed roles. Crossing-level accounting favors competing target
demands over direct peer-induced sign reversal; release timing remains unresolved.

For a concise discussion document, see the [four-page mentor update](mentor_report/README.md).

Read these reports in order:

1. [Readout adaptation and feature formation](reports/feature_profile.md): the
   latest follow-up. Early corrections rescue width-8 feature learning on fresh
   starts, even without resetting moments. Intercept controls, Adam alternatives,
   and failed cross-setting forecasts limit the causal interpretation.
2. [Signed residual growth](reports/residual_growth.md): the starting research result.
   Opposing-output, overlapping teacher features produce persistent narrow-model
   failures under ordinary initialization. Width 32 beats width 8 in all 20
   paired starts for each of SGD, momentum and Adam. Narrow MSE is 2–5% of target
   variance, versus 0.0012–0.086% wide. Two simple sign predictors fail.
3. [MNIST](reports/mnist.md): width 16→128 improves test accuracy by about 3–5
   percentage points, while several interference quantities give inconsistent
   width trends. The residual-potential hypothesis has not yet been tested here.
4. [Research history](reports/research_history.md): the useful conclusions from
   older teacher, allocation, temporal-intervention and specialization studies.
   These constrain the current interpretation; failed ideas are not erased.

| Claim | Status |
| --- | --- |
| Extra width can materially improve fitting even above a known teacher width | Supported in the controlled teacher studies |
| Residual correlations govern existing neurons' signed amplitude growth | Derived for the stated gradient-flow assumptions |
| Opposing feature demands can suppress a correct output polarity | Derived locally; strongest training regime is consistent with role loss |
| Early coefficient adaptation can materially change final feature quality | Supported by fixed-width interventions; not unique mediation of width |
| One tested scalar sign predicts where width helps | Not supported by the current experiments |
| Width always reduces pair cost, batch conflict, or allocation inefficiency | Refuted by retained counterexamples |
| Role suppression explains the entire width gap, including MNIST | Unproven |

The [theory guide](theory/README.md) separates local identities, assumptions,
and unvalidated predictions. [Metrics](metrics.md) defines signs and denominators.
[Tables](tables/README.md) retain cohorts and uncertainty; [figures](figures/README.md)
contain only maintained study figures. See [reproduction](reproduction.md) before
starting a long run and [retention](data.md) before deleting or interpreting data.

**23 October synthesis:** [General competition formalizations](theory/general_competition.md)
separates useful learning opportunities, interaction operators and causal tests,
and extends the signed spectral certificate to smooth non-Gaussian densities.
The [input-law probe](reports/general_competition.md) tests transfer beyond Gaussian data.
