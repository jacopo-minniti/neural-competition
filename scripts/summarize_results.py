#!/usr/bin/env python3
"""Print the final full-cohort summaries without requiring original raw runs."""

from pathlib import Path

import pandas as pd


def main():
    tables = Path(__file__).resolve().parents[1] / "docs/tables"
    adam = pd.read_csv(tables / "adam_cross_batch_summary.csv")
    early = adam.query('comparison == "mse_0.2"').copy()
    early["harm_percent"] = 100 * early.step_p_negative
    print("Adam cross-batch transfer near training MSE 0.2:")
    print(
        early[["profile", "width", "n_seeds", "harm_percent", "step_P", "step_N"]].to_string(
            index=False
        )
    )
    pairs = pd.read_csv(tables / "pair_budget_summary.csv")
    early = pairs.query(
        'comparison == "mse_0.2" and probe_set == "original128" and scale == 1'
    ).copy()
    early["cost_per_100_gain"] = 100 * early.S_plus
    print("\nBroad same-batch pair cost near training MSE 0.2:")
    print(early[["optimizer", "width", "n_seeds", "cost_per_100_gain"]].to_string(index=False))
    fitting = pd.read_csv(tables / "robustness_per_seed_results.csv")
    print("\nRobustness table:", len(fitting), "full-cohort result rows retained (205 runs × two evaluation views).")
    print("Detailed results and interpretations: docs/README.md")


if __name__ == "__main__":
    main()
