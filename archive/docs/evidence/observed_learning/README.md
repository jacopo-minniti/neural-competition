# Observed-learning evidence

`analysis.json` retains the exact finite-update ledgers, MNIST panel-bias study,
batch-size intervention, and toy boundary audit. The figures summarize cumulative
accounting and the fixed-checkpoint batch intervention.

The ledger explains observed loss changes but is not an independent competition
mechanism. Large-batch interventions reduce convex finite-step cost locally;
small-panel self-response is strongly biased. Raw panels and trajectories are
archived at commit `92b7904` and can be regenerated with the maintained ledger
scripts.

Theory: [learning ledger](../../model/learning_ledger.md).
