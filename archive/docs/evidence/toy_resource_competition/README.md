# Heterogeneous toy competition evidence

This directory retains the aggregate analysis, figures, and executed analyzer
sources for six controlled task families. The main results are:

- all fresh discrete runs start structurally covered; 15/64 narrow paths close,
  three recover, and no wide path closes;
- common pressure causes some local support losses but is beneficial overall;
- same-width directional recycling improves eligible discrete and separate-
  attribute states while holding predictions and kernel trace fixed;
- static strength, breadth, and frozen finite-budget response do not explain the
  full long-horizon intervention effect.

`summary.json` contains cohort statistics, uncertainty, thresholds, and negative
results. `natural_transitions.png` and `interventions.png` are the maintained
figures. Raw trajectories are archived at commit `92b7904` and can be regenerated
with the maintained toy producer and audit scripts.

Theory: [shared implementation competition](../../model/shared_implementation_competition.md).
