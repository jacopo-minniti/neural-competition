#!/usr/bin/env python3
"""Audit the new study without assuming SGD has a momentum buffer of zero."""

import json
from pathlib import Path

import torch

from neural_competition.diagnostics import batch_curvature_samples, gradient_blocks, probe_losses
from neural_competition.model import ShallowReLU


def restore(blocks, model):
    d = model.input_dim
    with torch.no_grad():
        model.weight.copy_(blocks[:, :d])
        model.bias.copy_(blocks[:, d])
        model.output.copy_(blocks[:, d + 1])


def main():
    root = Path(__file__).resolve().parents[1]
    names = [
        "pier_distributions",
        "pier_optimizer_pilot",
        "pier_early",
        "pier_early_extra",
        "pier_initialization_control",
        "pier_stability_pilot",
        "pier_stability_continuation",
    ]
    torch.set_num_threads(2)
    report = {}
    for name in names:
        folder = root / "runs" / name
        config = json.loads((folder / "config.json").read_text())
        data = torch.load(folder / "data.pt", weights_only=True)
        indices = data["probe_indices"]
        x_probe, y_probe = data["x_train"][indices], data["y_train"][indices]
        runs, snapshots, recomputations, diverged = 0, 0, 0, 0
        for path in sorted(folder.glob("w*/state.json")):
            state = json.loads(path.read_text())
            assert state["status"] in {"complete", "diverged"}, (path, state["status"])
            if state["status"] == "diverged":
                diverged += 1
                continue
            runs += 1
            model = ShallowReLU(config["input_dim"], state["width"])
            checkpoints = sorted((path.parent / "checkpoints").glob("*.pt"))
            assert checkpoints
            for checkpoint in checkpoints:
                snapshot = torch.load(checkpoint, weights_only=True)
                torch.testing.assert_close(
                    snapshot["parameters_after"] - snapshot["parameters_before"],
                    snapshot["actual_step"],
                )
                assert snapshot["probe_gradients"].shape == (
                    config["probe_batches"],
                    state["width"],
                    config["input_dim"] + 2,
                )
                assert torch.isfinite(snapshot["probe_gradients"]).all()
                snapshots += 1
            # Independently recompute one complete pre/post state per run, in a
            # state with learned hidden features, and one curvature batch.
            snapshot = torch.load(checkpoints[-1], weights_only=True)
            restore(snapshot["parameters_before"], model)
            value, gradient = gradient_blocks(model, x_probe[0], y_probe[0])
            torch.testing.assert_close(gradient, snapshot["probe_gradients"][0])
            torch.testing.assert_close(value, snapshot["probe_losses_before"][0])
            numerator, denominator = batch_curvature_samples(model, x_probe[:1], y_probe[:1])
            torch.testing.assert_close(numerator, snapshot["batch_curvature_numerators"][:1])
            torch.testing.assert_close(denominator, snapshot["batch_curvature_denominators"][:1])
            restore(snapshot["parameters_after"], model)
            torch.testing.assert_close(
                probe_losses(model, x_probe, y_probe),
                snapshot["probe_losses_after"],
                atol=1e-7,
                rtol=1e-4,
            )
            final = torch.load(path.parent / "final.pt", weights_only=True)
            torch.testing.assert_close(final["parameters"], snapshot["parameters_after"])
            recomputations += 1
        report[name] = dict(
            complete=runs,
            diverged=diverged,
            snapshots=snapshots,
            independently_recomputed=recomputations,
        )
        print(name, report[name], flush=True)
    continuation = root / "runs/pier_stability_continuation"
    for source in json.loads((continuation / "sources.json").read_text()):
        seed = source["seed"]
        original = torch.load(root / source["path"], weights_only=True)["parameters"]
        for folder in continuation.glob(f"w0016_seed{seed}_*"):
            first = torch.load(folder / "checkpoints/step_000000.pt", weights_only=True)
            torch.testing.assert_close(first["parameters_before"], original)
    tail_folder = root / "runs/pier_study/tail_checks"
    tail_indices = torch.load(tail_folder / "probe_indices.pt", weights_only=True)[0]
    tail_count = 0
    for path in tail_folder.glob("w*.pt"):
        snapshot = torch.load(path, weights_only=True)
        model = ShallowReLU(20, len(snapshot["parameters_before"]))
        restore(snapshot["parameters_before"], model)
        _, gradient = gradient_blocks(
            model, data["x_train"][tail_indices], data["y_train"][tail_indices]
        )
        torch.testing.assert_close(gradient, snapshot["probe_gradients"][0])
        tail_count += 1
    report["fresh_512_probe_states"] = tail_count
    (root / "runs/pier_study/audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
