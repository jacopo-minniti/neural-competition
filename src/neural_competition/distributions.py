"""Distribution diagnostics and seed-level uncertainty for Pier's experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import load_checkpoint
from .diagnostics import alignment_arrays, distribution_summary

LABELS = {
    "sgd_1": "SGD, lr 1",
    "momentum_01": "Momentum, lr 0.1",
    "adam_0001": "Adam, lr 0.001",
    "sgd_01": "SGD, lr 0.1",
    "adam_001": "Adam, lr 0.01",
    "muon_002": "Muon + Adam",
}


def read_study(root, names=("pier_distributions", "pier_optimizer_pilot")):
    """Completed cases only; keep optimizer profiles and budgets separate."""
    frames, outcomes = [], []
    for name in names:
        folder = Path(root) / "runs" / name
        for run in sorted(folder.glob("w*/state.json")):
            state = json.loads(run.read_text())
            if state["status"] != "complete":
                continue
            identity = {
                "collection": name,
                "width": state["width"],
                "seed": state["seed"],
                "optimizer": state["optimizer"]["label"],
                "run_path": str(run.parent),
            }
            frame = pd.read_csv(run.parent / "metrics.csv")
            for key, value in identity.items():
                frame[key] = value
            frame["train_mse"] = 2 * frame["train_loss"]
            frame["test_mse"] = 2 * frame["test_loss"]
            frames.append(frame)
            outcomes.append(
                {
                    **identity,
                    "updates": state["updates_completed"],
                    "train_mse": 2 * state["final_train_loss"],
                    "test_mse": 2 * state["final_test_loss"],
                }
            )
    if not frames:
        raise FileNotFoundError("Run configs/studies/pier_distributions.json first")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(outcomes)


def cosine_with(flat, direction):
    denominator = flat.norm(dim=1) * direction.norm()
    return torch.where(denominator > 0, (flat @ direction) / denominator, float("nan"))


def distribution_arrays(snapshot):
    """Positive step alignment means an infinitesimal realized step helps a probe.

    Pair normalization divides by E_B ||g_B||^2, preserving both mean and shape.
    A moment is one saved buffer projected onto many independent probe gradients;
    it is not a distribution of buffers sampled from independent histories.
    """
    flat = snapshot["probe_gradients"].double().flatten(1)
    pair_dot, pair_cosine = alignment_arrays(flat)
    norm_square = flat.square().sum(dim=1).mean()
    gradient = snapshot["mean_gradient"].double().flatten()
    delta = snapshot["actual_step"].double().flatten()
    kind = snapshot["optimizer"]["name"]
    buffer_key = "exp_avg" if kind == "adam" else "momentum_buffer"
    moment = snapshot["optimizer_after"][buffer_key].double().flatten()
    # Hybrid Muon/Adam has two different moment conventions; do not mix them.
    if kind == "muon":
        moment = torch.zeros_like(moment)
    return {
        "raw_dot": pair_dot.numpy(),
        "scaled_dot": (pair_dot / norm_square).numpy(),
        "cosine": pair_cosine.numpy(),
        "mean_dot": (flat @ gradient / norm_square).numpy(),
        "mean_cosine": cosine_with(flat, gradient).numpy(),
        "step_alignment": cosine_with(flat, -delta).numpy(),
        "moment_alignment": cosine_with(flat, moment).numpy(),
    }


def summarize_arrays(arrays):
    result = {}
    for prefix, values in arrays.items():
        result.update(distribution_summary(prefix, torch.as_tensor(values)))
    return result


def seed_mean_interval(values, *, repetitions=4000, seed=741):
    """Percentile bootstrap interval; resampling unit is a training seed."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = values.mean()
    if len(values) < 2:
        return mean, np.nan, np.nan
    generator = np.random.default_rng(seed)
    boot = values[generator.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
    return mean, *np.quantile(boot, [0.025, 0.975])


def paired_width_effect(frame, metric, small=8, large=16):
    """Large-minus-small; width pairs use the same seed/minibatch order."""
    pivot = frame.pivot(index="seed", columns="width", values=metric)
    if small not in pivot or large not in pivot:
        return {"metric": metric, "n": 0, "mean": np.nan, "low": np.nan, "high": np.nan}
    difference = (pivot[large] - pivot[small]).dropna()
    mean, low, high = seed_mean_interval(difference)
    return {
        "metric": metric,
        "small": small,
        "large": large,
        "n": len(difference),
        "mean": mean,
        "low": low,
        "high": high,
    }


def matched_states(metrics, target=0.01, tolerance=0.25):
    """Closest observed pre-update loss, with explicit eligibility/tolerance.

    The trigger is checked every 100 updates. Not all runs need reach the target.
    Always report counts and attained losses; this is an observational comparison.
    """
    frame = metrics.copy()
    frame["loss_distance"] = np.abs(np.log(frame["train_mse"] / target))
    keys = ["collection", "optimizer", "width", "seed"]
    frame = frame.loc[frame.groupby(keys)["loss_distance"].idxmin()].copy()
    frame["target_mse"] = target
    frame["loss_eligible"] = frame["train_mse"].between(
        target * (1 - tolerance), target * (1 + tolerance)
    )
    return frame


def analyze_states(frame, *, cache_arrays=False):
    rows, arrays = [], {}
    for _, row in frame.iterrows():
        snapshot = load_checkpoint(row["run_path"], int(row["step"]))
        values = distribution_arrays(snapshot)
        record = row.to_dict()
        record.update(summarize_arrays(values))
        record["gradient_signal_ratio"] = (
            snapshot["mean_gradient"].double().square().sum()
            / snapshot["probe_gradients"].double().flatten(1).square().sum(dim=1).mean()
        ).item()
        rows.append(record)
        if cache_arrays:
            arrays[(row["optimizer"], int(row["width"]), int(row["seed"]))] = values
    return pd.DataFrame(rows), arrays


def finite_step_scales(snapshot, x_probe, y_probe, scales=(0.01, 0.1, 0.25, 0.5, 1.0, 2.0)):
    """Exact probe-loss changes along the saved update, evaluated in float64.

    The local quadratic is formed analytically for a shallow ReLU network.
    It excludes gate crossings, whereas the exact finite changes include them.
    This intervention rescales one step; it is not a new training trajectory.
    """
    import math

    before = snapshot["parameters_before"].double()
    delta = snapshot["actual_step"].double()
    x, y = x_probe.double(), y_probe.double()
    d = x.shape[-1]
    z = x @ before[:, :d].T + before[:, d]
    dz = x @ delta[:, :d].T + delta[:, d]
    normalization = math.sqrt(len(before))
    prediction = (z.relu() * before[:, -1]).sum(dim=-1) / normalization
    residual = prediction - y
    gate = z > 0
    first = ((z.relu() * delta[:, -1]) + gate * dz * before[:, -1]).sum(-1)
    first = first / normalization
    second = (gate * dz * delta[:, -1]).sum(-1) / normalization
    linear = (residual * first).mean(dim=1)
    quadratic = (residual * second + 0.5 * first.square()).mean(dim=1)
    rows = []
    for scale in scales:
        parameters = before + scale * delta
        z_new = z + scale * dz
        prediction_new = (z_new.relu() * parameters[:, -1]).sum(-1) / normalization
        change = prediction_new - prediction
        exact = (residual * change + 0.5 * change.square()).mean(dim=1)
        approximation = scale * linear + scale**2 * quadratic
        rows.append(
            {
                "scale": scale,
                "mean_loss_change": exact.mean().item(),
                "first_order_mean": (scale * linear).mean().item(),
                "quadratic_mean": approximation.mean().item(),
                "linear_harm_fraction": (linear > 0).double().mean().item(),
                "quadratic_harm_fraction": (approximation > 0).double().mean().item(),
                "actual_harm_fraction": (exact > 0).double().mean().item(),
                "gate_crossing_fraction": (gate != (z_new > 0)).double().mean().item(),
                "quadratic_absolute_error": (exact - approximation).abs().mean().item(),
            }
        )
    return pd.DataFrame(rows)
