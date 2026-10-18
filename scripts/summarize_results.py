#!/usr/bin/env python3
"""Print current full-cohort findings using only versioned tables."""

from pathlib import Path

import pandas as pd


def main():
    tables = Path(__file__).resolve().parents[1] / "docs/tables"
    growth = pd.read_csv(tables / "residual_growth/summary.csv")
    selected = growth.query(
        'study == "residual_growth" and collection == "replication20" '
        'and degrees == 30 and ratio == -3 and arm in ["w8_fixed", "w32_fixed"]'
    ).copy()
    selected["MSE_percent_target_variance"] = 200 * selected.mean_loss
    print("Opposing-output teacher, 10k steps, 20 paired starts; lower is better:")
    print(selected[["optimizer", "arm", "seeds", "MSE_percent_target_variance"]]
          .to_string(index=False))
    mnist = pd.read_csv(tables / "mnist/final_summary.csv.gz")
    selected = mnist.query(
        'collection == "main" and metric == "test_accuracy" and width in [16, 128]'
    ).copy()
    selected["test_accuracy_percent"] = 100 * selected["mean"]
    print("\nMNIST test accuracy, five paired starts; higher is better:")
    print(selected[["loss", "optimizer", "width", "test_accuracy_percent"]]
          .to_string(index=False))
    print("\nWidth helps here; a general competition mechanism remains unproven.")
    print("Interpretation and failed predictions: docs/README.md")


if __name__ == "__main__":
    main()
