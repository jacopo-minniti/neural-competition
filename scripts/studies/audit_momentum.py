#!/usr/bin/env python3
"""Verify momentum reconstruction and repeat the early analysis on fresh batches."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from analyze_neurons import effects
from audit_neurons import assign
from compare_momentum import MOVEMENT_METRICS, record_movement, statistics

from neural_competition.analysis import load_checkpoint
from neural_competition.diagnostics import gradient_blocks
from neural_competition.model import ShallowReLU, loss
from neural_competition.neurons import (
    candidate_descent,
    movement_statistics,
    prediction_moves,
    source_improvement,
)


def verify(root, data, output, selected):
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    # Independently replay PyTorch's optimizer and isolate each neuron's block.
    samples = selected[(selected.seed == 0) & (selected.comparison_kind == "matched_loss")]
    largest_error = 0.0
    for row in samples.itertuples():
        snapshot = load_checkpoint(row.run_path, row.step)
        before = snapshot["parameters_before"].double()
        model = ShallowReLU(20, row.width).double()
        assign(model, before)
        indices = data["probe_indices"][0]
        x, y = data["x_train"][indices].double(), data["y_train"][indices].double()
        _, reference = gradient_blocks(model, x, y)
        torch.testing.assert_close(
            reference, snapshot["probe_gradients"][0].double(), atol=2e-7, rtol=3e-5
        )
        momentum = snapshot["optimizer"]["momentum"] if row.optimizer == "momentum_01" else 0.0
        optimizer = torch.optim.SGD(
            model.parameters(), lr=snapshot["optimizer"]["lr"], momentum=momentum
        )
        previous = snapshot["optimizer_before"]["momentum_buffer"].double()
        gradients = snapshot["probe_gradients"][0].double()
        for p, g, v in zip(
            model.parameters(),
            (gradients[:, :20], gradients[:, 20], gradients[:, -1]),
            (previous[:, :20], previous[:, 20], previous[:, -1]),
        ):
            p.grad = g.clone()
            if momentum:
                optimizer.state[p]["momentum_buffer"] = v.clone()
        direction = candidate_descent(snapshot, gradients)
        optimizer.step()
        torch.testing.assert_close(
            model.parameter_blocks(),
            before - snapshot["optimizer"]["lr"] * direction,
            atol=1e-14,
            rtol=1e-13,
        )
        # The recorded training step differs only by float32 arithmetic/rounding.
        replay = before - snapshot["optimizer"]["lr"] * candidate_descent(
            snapshot, snapshot["sampled_gradient"]
        )
        torch.testing.assert_close(
            replay, snapshot["parameters_after"].double(), atol=2e-7, rtol=1e-6
        )
        delta = -snapshot["optimizer"]["lr"] * direction
        x_test, y_test = data["x_test"].double(), data["y_test"].double()
        assign(model, before)
        old_source = loss(model, x, y)
        old_test = loss(model, x_test, y_test)
        prediction, moves, _ = prediction_moves(before, delta[None], x_test)
        for r in range(row.width):
            isolated = before.clone()
            isolated[r] += delta[r]
            assign(model, isolated)
            discrepancy = (model(x_test) - prediction - moves[0, :, r]).abs().max().item()
            largest_error = max(largest_error, discrepancy)
            assert discrepancy < 1e-12
        assign(model, before + delta)
        stats, _ = movement_statistics(prediction - y_test, moves)
        torch.testing.assert_close(
            loss(model, x_test, y_test) - old_test, stats["loss_change"][0], atol=1e-12, rtol=1e-10
        )
        source = source_improvement(before, delta[None], x[None], y[None])
        torch.testing.assert_close(
            old_source - loss(model, x, y), source[0], atol=1e-12, rtol=1e-10
        )
    audit = {
        "immutable_sources": len(manifest["source_sha256"]),
        "independent_optimizer_states": len(samples),
        "max_isolated_prediction_error": largest_error,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(audit, flush=True)


def fresh_validation(root, data, output, selected):
    indices = torch.randint(
        len(data["x_train"]), (32, 64), generator=torch.Generator().manual_seed(20261018)
    )
    torch.save(indices, output / "fresh_source_indices.pt")
    x_source, y_source = data["x_train"][indices], data["y_train"][indices]
    x_test, y_test = data["x_test"].double(), data["y_test"].double()
    early = selected[selected.comparison == "mse_0.2"]
    records, saved_gradients = [], {}
    for (optimizer, seed), pair in early.groupby(["optimizer", "seed"]):
        states = {}
        for row in pair.itertuples():
            snapshot = load_checkpoint(row.run_path, row.step)
            parameters = snapshot["parameters_before"]
            model = ShallowReLU(20, row.width)
            assign(model, parameters)
            g = torch.stack([gradient_blocks(model, x, y)[1] for x, y in zip(x_source, y_source)])
            saved_gradients[f"{optimizer}_w{row.width}_seed{seed}"] = g
            pred, _, _ = prediction_moves(parameters, torch.zeros_like(parameters)[None], x_test)
            states[row.width] = (row, snapshot, g, pred - y_test)
        mask = (states[8][-1].abs() <= 0.05) & (states[16][-1].abs() <= 0.05)
        for width, (row, snapshot, g, residual) in states.items():
            modes = ["optimizer", "gradient_only"] if optimizer == "momentum_01" else ["optimizer"]
            for direction in modes:
                descent = candidate_descent(snapshot, g, include_history=direction == "optimizer")
                for scale in [0.01, 1.0]:
                    delta = -snapshot["optimizer"]["lr"] * scale * descent
                    _, moves, _ = prediction_moves(snapshot["parameters_before"], delta, x_test)
                    gain = source_improvement(
                        snapshot["parameters_before"], delta, x_source, y_source
                    )
                    for subset, subset_mask in [("all", None), ("well_fit_0.05", mask)]:
                        stats, _ = movement_statistics(residual, moves, subset_mask)
                        records.append(
                            {
                                "optimizer": optimizer,
                                "comparison": "mse_0.2",
                                "direction": direction,
                                "scale": scale,
                                "subset": subset,
                                "width": width,
                                "seed": seed,
                                "train_mse": row.train_mse,
                                "n_examples": len(y_test)
                                if subset_mask is None
                                else mask.sum().item(),
                                **record_movement(stats, gain, row.train_mse),
                            }
                        )
        if seed % 10 == 0:
            print(f"Fresh directions: {optimizer}, seed {seed}", flush=True)
    torch.save(saved_gradients, output / "fresh_source_gradients.pt")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "fresh_movement_seeds.csv", index=False)
    groups = ["optimizer", "comparison", "direction", "scale", "subset"]
    statistics(frame, groups, MOVEMENT_METRICS).to_csv(
        output / "fresh_movement_summary.csv", index=False
    )
    effects(frame, groups, MOVEMENT_METRICS).to_csv(
        output / "fresh_movement_effects.csv", index=False
    )


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "runs/pier_study/momentum"
    torch.set_num_threads(4)
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    selected = pd.read_csv(output / "selected_states.csv")
    verify(root, data, output, selected)
    fresh_validation(root, data, output, selected)
    print("Momentum verification and fresh-batch replication complete.", flush=True)


if __name__ == "__main__":
    main()
