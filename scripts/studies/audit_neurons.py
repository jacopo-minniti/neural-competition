#!/usr/bin/env python3
"""Independently verify neuron effects and repeat early checks on fresh batches."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from analyze_neurons import effects, estimate

from neural_competition.analysis import load_checkpoint
from neural_competition.diagnostics import gradient_blocks
from neural_competition.model import ShallowReLU, loss
from neural_competition.neurons import movement_statistics, prediction_moves, source_improvement


def assign(model, parameters):
    with torch.no_grad():
        model.weight.copy_(parameters[:, : model.input_dim])
        model.bias.copy_(parameters[:, model.input_dim])
        model.output.copy_(parameters[:, -1])


def verify(root, data, output):
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    selected = pd.read_csv(output / "selected_states.csv")
    samples = selected[(selected.seed == 0) & selected.comparison.isin(["mse_0.2", "mse_0.01"])]
    largest_error = 0.0
    for row in samples.itertuples():
        snapshot = load_checkpoint(row.run_path, row.step)
        before = snapshot["parameters_before"].double()
        model = ShallowReLU(20, row.width).double()
        assign(model, before)
        indices = data["probe_indices"][0]
        x, y = data["x_train"][indices].double(), data["y_train"][indices].double()
        _, reference_gradient = gradient_blocks(model, x, y)
        saved_gradient = snapshot["probe_gradients"][0].double()
        torch.testing.assert_close(reference_gradient, saved_gradient, atol=2e-7, rtol=3e-5)
        delta = -snapshot["optimizer"]["lr"] * saved_gradient
        x_test, y_test = data["x_test"].double(), data["y_test"].double()
        prediction, moves, _ = prediction_moves(before, delta[None], x_test)
        old_loss = loss(model, x_test, y_test)
        source_before = loss(model, x, y)
        # Separate model forwards, rather than the vectorized decomposition.
        for r in range(row.width):
            isolated = before.clone()
            isolated[r] += delta[r]
            assign(model, isolated)
            error = (model(x_test) - prediction - moves[0, :, r]).abs().max().item()
            largest_error = max(largest_error, error)
            assert error < 1e-12
        assign(model, before + delta)
        exact_change = loss(model, x_test, y_test) - old_loss
        stats, _ = movement_statistics(prediction - y_test, moves)
        torch.testing.assert_close(exact_change, stats["loss_change"][0], atol=1e-12, rtol=1e-10)
        gain = source_improvement(before, delta[None], x[None], y[None])
        torch.testing.assert_close(
            source_before - loss(model, x, y), gain[0], atol=1e-12, rtol=1e-10
        )
    record = {
        "immutable_checkpoints": len(manifest["source_sha256"]),
        "independent_model_checks": len(samples),
        "max_isolated_prediction_error": largest_error,
    }
    (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


def fresh_validation(root, data, output):
    indices = torch.randint(
        len(data["x_train"]), (32, 64), generator=torch.Generator().manual_seed(20261018)
    )
    torch.save(indices, output / "validation_source_indices.pt")
    x_source, y_source = data["x_train"][indices], data["y_train"][indices]
    x_test, y_test = data["x_test"].double(), data["y_test"].double()
    selected = pd.read_csv(output / "selected_states.csv")
    early = selected[(selected.optimizer == "sgd_1") & (selected.comparison == "mse_0.2")]
    records, saved_gradients = [], {}
    metrics = [
        "R",
        "cancellation_fraction",
        "solo_harm_offset_fraction",
        "joint_harm_per_source_gain",
        "joint_benefit_per_source_gain",
    ]
    for seed, pair in early.groupby("seed"):
        states = {}
        for row in pair.itertuples():
            snapshot = load_checkpoint(row.run_path, row.step)
            parameters = snapshot["parameters_before"]
            model = ShallowReLU(20, row.width)
            assign(model, parameters)
            gradients = torch.stack(
                [gradient_blocks(model, x, y)[1] for x, y in zip(x_source, y_source)]
            )
            saved_gradients[f"w{row.width}_seed{seed}"] = gradients
            prediction, _, _ = prediction_moves(
                parameters, torch.zeros_like(parameters)[None], x_test
            )
            states[row.width] = (row, parameters, gradients, prediction - y_test)
        mask = (states[8][-1].abs() <= 0.05) & (states[16][-1].abs() <= 0.05)
        for width, (row, parameters, gradients, residual) in states.items():
            for scale in [0.01, 1.0]:
                delta = -scale * gradients.double()  # SGD lr 1 in the early collection.
                _, moves, _ = prediction_moves(parameters, delta, x_test)
                gain = source_improvement(parameters, delta, x_source, y_source).mean().item()
                for subset, subset_mask in [("all", None), ("well_fit_0.05", mask)]:
                    stats, _ = movement_statistics(residual, moves, subset_mask)
                    means = {k: v.mean().item() for k, v in stats.items()}
                    records.append(
                        {
                            "optimizer": "sgd_1",
                            "comparison": "mse_0.2",
                            "scale": scale,
                            "subset": subset,
                            "width": width,
                            "seed": seed,
                            "train_mse": row.train_mse,
                            "n_examples": len(y_test) if subset_mask is None else mask.sum().item(),
                            "R": means["joint_energy"] / means["individual_energy"],
                            "cancellation_fraction": 1
                            - means["absolute_joint_movement"]
                            / means["absolute_individual_movement"],
                            "solo_harm_offset_fraction": means["offset_solo_harm"]
                            / means["solo_harm"],
                            "joint_harm_per_source_gain": means["joint_harm"] / gain,
                            "joint_benefit_per_source_gain": means["joint_benefit"] / gain,
                        }
                    )
        if seed % 10 == 0:
            print(f"Fresh-batch check: {seed + 1}/40 paired seeds", flush=True)
    torch.save(saved_gradients, output / "validation_source_gradients.pt")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "validation_prediction_changes_seeds.csv", index=False)
    groups = ["optimizer", "comparison", "scale", "subset"]
    estimate(frame, [*groups, "width"], metrics).to_csv(
        output / "validation_prediction_changes_summary.csv", index=False
    )
    effects(frame, groups, metrics).to_csv(
        output / "validation_prediction_changes_effects.csv", index=False
    )
    print("Fresh-batch validation complete.", flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "runs/pier_study/neurons"
    torch.set_num_threads(4)
    data = torch.load(root / "runs/pier_distributions/data.pt", weights_only=True)
    verify(root, data, output)
    fresh_validation(root, data, output)


if __name__ == "__main__":
    main()
