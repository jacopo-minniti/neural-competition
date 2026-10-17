#!/usr/bin/env python3
"""Validate tail shapes with 512 fresh probe batches at saved early states."""

from pathlib import Path

import pandas as pd
import torch

from neural_competition.analysis import load_checkpoint
from neural_competition.diagnostics import gradient_blocks
from neural_competition.distributions import (
    analyze_states,
    distribution_arrays,
    matched_states,
    read_study,
    summarize_arrays,
)
from neural_competition.experiment import save_torch
from neural_competition.model import ShallowReLU


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "runs/pier_study/tail_checks"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    data = torch.load(root / "runs/pier_early/data.pt", weights_only=True)
    indices = torch.randint(
        len(data["x_train"]), (512, 64), generator=torch.Generator().manual_seed(20261017)
    )
    save_torch(output / "probe_indices.pt", indices)
    x, y = data["x_train"][indices], data["y_train"][indices]
    metrics, _ = read_study(root, names=("pier_early", "pier_early_extra"))
    chosen = matched_states(metrics, target=0.2, tolerance=0.1)
    chosen = chosen[chosen.loss_eligible & chosen.width.isin([8, 16])]
    baseline, _ = analyze_states(chosen)
    baseline.to_csv(output / "original_128.csv", index=False)
    rows = []
    for _, row in chosen.iterrows():
        path = output / f"w{int(row.width):04d}_seed{int(row.seed)}_{row.optimizer}.pt"
        if path.exists():
            snapshot = torch.load(path, weights_only=True)
        else:
            snapshot = load_checkpoint(row.run_path, int(row.step))
            model = ShallowReLU(20, int(row.width))
            parameters = snapshot["parameters_before"]
            with torch.no_grad():
                model.weight.copy_(parameters[:, :20])
                model.bias.copy_(parameters[:, 20])
                model.output.copy_(parameters[:, -1])
            gradients = [gradient_blocks(model, xb, yb)[1] for xb, yb in zip(x, y)]
            snapshot = {
                key: snapshot[key]
                for key in [
                    "optimizer",
                    "optimizer_after",
                    "actual_step",
                    "mean_gradient",
                    "parameters_before",
                ]
            }
            snapshot.update(
                probe_gradients=torch.stack(gradients),
                source_checkpoint=str(
                    Path(row.run_path) / f"checkpoints/step_{int(row.step):06d}.pt"
                ),
            )
            save_torch(path, snapshot)
        arrays = distribution_arrays(snapshot)
        record = {key: row[key] for key in ["width", "seed", "optimizer", "step", "train_mse"]}
        record.update(summarize_arrays(arrays))
        values = torch.as_tensor(arrays["scaled_dot"])
        z = (values - values.mean()) / values.std(unbiased=False)
        record["left_3sd"] = (z < -3).double().mean().item()
        record["right_3sd"] = (z > 3).double().mean().item()
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_512.csv", index=False)
    print(
        frame.groupby(["optimizer", "width"])[
            ["scaled_dot_skewness", "scaled_dot_excess_kurtosis", "left_3sd", "right_3sd"]
        ].mean()
    )


if __name__ == "__main__":
    main()
