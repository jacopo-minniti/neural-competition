"""Load the curated teacher-8 results without running training."""

import json
import math
from pathlib import Path

import pandas as pd
import torch


def load_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for name in ("teacher8", "teacher8_counterexample"):
        folder = Path(root) / "runs" / name
        if not (folder / "config.json").exists():
            raise FileNotFoundError(
                "Original-layout cohort is absent; the curated archive is not a full cohort. "
                "See docs/reproduction.md."
            )
        config = json.loads((folder / "config.json").read_text())
        for run in sorted(folder.glob("w*")):
            state = json.loads((run / "state.json").read_text())
            if state["status"] != "complete":
                continue
            frame = pd.read_csv(run / "metrics.csv")
            frame["collection"] = name
            frame["width"], frame["seed"] = state["width"], state["seed"]
            frame["teacher_width"] = config["teacher_width"]
            frame["run_path"] = str(run)
            frame["updates"] = frame["updates_completed"]
            frame["train_mse"] = 2 * frame["train_loss_after"]
            frame["test_mse"] = 2 * frame["test_loss_after"]
            frame["train_mse_before"] = 2 * frame["train_loss"]
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No complete teacher-8 results; see docs/reproduction.md")
    return pd.concat(frames, ignore_index=True)


def load_checkpoint(run_path: str | Path, step: int) -> dict:
    path = Path(run_path) / "checkpoints" / f"step_{step:06d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def neuron_prediction_changes(snapshot: dict, x: torch.Tensor) -> dict:
    """Finite and first-order prediction changes per neuron, in float64.

    The quadratic term is exact only for neuron/example pairs whose activation
    gate does not change. A negative interaction need not be detrimental.
    """
    before = snapshot["parameters_before"].double()
    after = snapshot["parameters_after"].double()
    delta = after - before
    d = x.shape[-1]
    x = x.reshape(-1, d).double()
    z = x @ before[:, :d].T + before[:, d]
    z_after = x @ after[:, :d].T + after[:, d]
    dz = x @ delta[:, :d].T + delta[:, d]
    gate = z > 0
    scale = math.sqrt(len(before))
    finite = (z_after.relu() * after[:, -1] - z.relu() * before[:, -1]) / scale
    linear = (z.relu() * delta[:, -1] + gate * dz * before[:, -1]) / scale
    quadratic = gate * dz * delta[:, -1] / scale
    return {
        "finite": finite,
        "linear": linear,
        "quadratic_same_region": quadratic,
        "gate_changed": gate != (z_after > 0),
    }
