#!/usr/bin/env python3
"""Check saved gradients, real updates and predictions; never train a network."""

import argparse
import json
from pathlib import Path

import torch

from neural_competition.diagnostics import gradient_blocks, probe_losses
from neural_competition.experiment import write_json
from neural_competition.model import ShallowReLU, loss


def restore(blocks, input_dim):
    model = ShallowReLU(input_dim, len(blocks))
    with torch.no_grad():
        model.weight.copy_(blocks[:, :input_dim])
        model.bias.copy_(blocks[:, input_dim])
        model.output.copy_(blocks[:, input_dim + 1])
    return model


def audit(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    data = torch.load(folder / "data.pt", weights_only=True)
    torch.set_num_threads(config["cpu_threads"])
    expected_steps = sorted(set(config["checkpoint_steps"]) | {0, config["steps"] - 1})
    expected_ids = {
        f"w{width:04d}_seed{seed}_{spec['label']}"
        for width in config["widths"]
        for seed in config["seeds"]
        for spec in config["optimizers"]
    }
    actual_ids = {p.name for p in folder.glob("w*") if p.is_dir()}
    if expected_ids != actual_ids:
        raise ValueError(f"Run inventory differs: {expected_ids ^ actual_ids}")
    checked = 0
    examples = []
    for run_id in sorted(expected_ids):
        run = folder / run_id
        state = json.loads((run / "state.json").read_text())
        assert state["status"] == "complete" and state["updates_completed"] == config["steps"]
        files = sorted((run / "checkpoints").glob("*.pt"))
        assert [int(p.stem.split("_")[1]) for p in files] == expected_steps
        for path in files:
            s = torch.load(path, map_location="cpu", weights_only=True)
            shape = (state["width"], config["input_dim"] + 2)
            assert s["probe_gradients"].shape == (config["probe_batches"], *shape)
            for name in (
                "mean_gradient",
                "sampled_gradient",
                "parameters_before",
                "parameters_after",
                "actual_step",
                "previous_step",
            ):
                assert s[name].shape == shape and torch.isfinite(s[name]).all()
            assert len(s["batch_sharpness_samples"]) == config["sharpness_batches"]
            for name in ("probe_losses_before", "probe_losses_after"):
                assert s[name].shape == (config["probe_batches"],)
                assert torch.isfinite(s[name]).all()
            for name in ("optimizer_before", "optimizer_after"):
                assert set(s[name]) == {"momentum_buffer", "exp_avg", "exp_avg_sq"}
            torch.testing.assert_close(
                s["parameters_after"] - s["parameters_before"], s["actual_step"], rtol=0, atol=0
            )
            # Neuron blocks must exactly account for the global gradient dot.
            g = s["probe_gradients"].double()
            torch.testing.assert_close(
                (g[0] * g[1]).sum(dim=1).sum(), torch.dot(g[0].flatten(), g[1].flatten())
            )
            if s["step"] in {0, 1999, 19999, config["steps"] - 1}:
                model = restore(s["parameters_before"], config["input_dim"])
                value, gradient = gradient_blocks(model, data["x_train"], data["y_train"])
                torch.testing.assert_close(gradient, s["mean_gradient"], atol=1e-7, rtol=2e-4)
                torch.testing.assert_close(
                    value, torch.tensor(s["train_loss_before"]), atol=1e-7, rtol=2e-4
                )
                indices = data["probe_indices"][0]
                value, gradient = gradient_blocks(
                    model, data["x_train"][indices], data["y_train"][indices]
                )
                torch.testing.assert_close(gradient, s["probe_gradients"][0], atol=1e-7, rtol=2e-4)
                torch.testing.assert_close(value, s["probe_losses_before"][0], atol=1e-7, rtol=2e-4)
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(data["x_test"]), s["test_predictions_before"], atol=1e-6, rtol=2e-4
                    )
                after = restore(s["parameters_after"], config["input_dim"])
                indices = data["probe_indices"]
                torch.testing.assert_close(
                    probe_losses(after, data["x_train"][indices], data["y_train"][indices]),
                    s["probe_losses_after"],
                    atol=1e-7,
                    rtol=2e-4,
                )
                torch.testing.assert_close(
                    loss(after, data["x_test"], data["y_test"]),
                    torch.tensor(s["test_loss_after"]),
                    atol=1e-7,
                    rtol=2e-4,
                )
            checked += 1
        final = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
        torch.testing.assert_close(final["parameters"], s["parameters_after"], rtol=0, atol=0)
        assert final["optimizer_state"]["param_groups"][0]["momentum"] == 0
        examples.append(
            {
                "run": run_id,
                "checkpoints": len(files),
                "final_test_mse": 2 * state["final_test_loss"],
            }
        )
    return {
        "status": "passed",
        "runs": len(expected_ids),
        "checkpoints": checked,
        "probe_batches_per_checkpoint": config["probe_batches"],
        "sharpness_samples_per_checkpoint": config["sharpness_batches"],
        "recomputed_steps": [0, 1999, 19999, config["steps"] - 1],
        "training_rerun": False,
        "cases": examples,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders",
        type=Path,
        nargs="*",
        default=[root / "runs/teacher8", root / "runs/teacher8_counterexample"],
    )
    args = parser.parse_args()
    for folder in args.folders:
        result = audit(folder)
        write_json(folder / "audit.json", result)
        print(f"{folder.name}: {result['runs']} runs, {result['checkpoints']} checkpoints verified")


if __name__ == "__main__":
    main()
