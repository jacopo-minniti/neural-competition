#!/usr/bin/env python3
"""Read-only checkpoint diagnostics for the literature notes; no training.

Run from the repository root:
    .venv/bin/python docs/theory/checks/reanalyze_saved_states.py
Only the two output files beside this script are written. Retained inputs and
the original cohort summaries are never changed.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "runs" / "retained"
OUTPUT = Path(__file__).resolve().parent
RCOND = 1e-12
GROUPS = [
    ("k8_teacher2027", "momentum_01"),
    ("k8_teacher2027", "adam_001"),
    ("k8_teacher2027", "adam_01"),
    ("k8_teacher2027_init01", "adam_001"),
    ("k8_teacher2027_init1", "adam_001"),
]
STATES = [("initial", 0), ("step_010000", 10_000), ("step_050000", 50_000), ("final", 100_000)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def main():
    torch.set_num_threads(1)
    dataset = ARCHIVE / "datasets" / "k8_teacher2027.pt"
    data = load(dataset)
    x = data["x_train"].double().numpy()
    xt = data["x_test"].double().numpy()
    y = data["y_train"].double().numpy()
    yt = data["y_test"].double().numpy()
    hashes = {str(dataset.relative_to(ROOT)): digest(dataset)}
    rows = []
    max_normal_error = 0.0

    for condition, profile in GROUPS:
        for state, completed in STATES:
            path = ARCHIVE / "checkpoints" / "robustness" / condition / profile / f"{state}.pt"
            saved = load(path)
            hashes[str(path.relative_to(ROOT))] = digest(path)
            for i, model in enumerate(saved["models"]):
                width = model["width"]
                p = saved["parameters"][i, :width].double().numpy()
                w, b, a = p[:, :-2], p[:, -2], p[:, -1]
                z, zt = x @ w.T + b, xt @ w.T + b
                phi = np.maximum(z, 0) / math.sqrt(width)
                phit = np.maximum(zt, 0) / math.sqrt(width)
                prediction, prediction_test = phi @ a, phit @ a
                coefficients, _, rank, _ = np.linalg.lstsq(phi, y, rcond=RCOND)
                error = phi @ coefficients - y
                fit_mse = float(np.mean(error**2))
                mse = float(np.mean((prediction - y) ** 2))
                normal_error = float(np.max(np.abs(phi.T @ error / len(y))))
                max_normal_error = max(max_normal_error, normal_error)
                assert fit_mse <= mse + 1e-10, (path, model, fit_mse, mse)
                assert normal_error < 1e-8, (path, model, normal_error)
                active = np.mean(z > 0, axis=0)
                norms = np.linalg.norm(w, axis=1)
                assert np.all(norms > 0), (path, model)
                kink = -b / norms
                # At zero output, -grad_a = E[y ReLU(z)] / sqrt(width).
                correlations = phi.T @ y / len(y)
                rows.append(
                    {
                        "condition": condition,
                        "profile": profile,
                        "checkpoint": state,
                        "completed": completed,
                        "width": width,
                        "seed": model["seed"],
                        "train_mse": mse,
                        "test_mse": float(np.mean((prediction_test - yt) ** 2)),
                        "positive_outputs": int(np.sum(a > 0)),
                        "negative_outputs": int(np.sum(a < 0)),
                        "inactive_on_train": int(np.sum(active == 0)),
                        "always_active_on_train": int(np.sum(active == 1)),
                        "active_below_1pct": int(np.sum(active < 0.01)),
                        "active_above_99pct": int(np.sum(active > 0.99)),
                        "min_active_fraction": float(np.min(active)),
                        "max_active_fraction": float(np.max(active)),
                        "min_kink": float(np.min(kink)),
                        "max_kink": float(np.max(kink)),
                        "readout_rank": int(rank),
                        "refit_train_mse": fit_mse,
                        "refit_test_mse": float(np.mean((phit @ coefficients - yt) ** 2)),
                        "positive_target_correlations": int(np.sum(correlations > 0)),
                        "min_target_correlation": float(np.min(correlations)),
                        "max_target_correlation": float(np.max(correlations)),
                        "readout_normal_equation_error": normal_error,
                    }
                )

    with (OUTPUT / "saved_state_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "scope": "144 saved states, teacher 8 / seed 2027; purposively retained seeds 0 and 3",
        "training_performed": False,
        "precision": "float64 evaluation of saved float32 parameters and data",
        "readout": "unconstrained least squares on full train set; no output bias; hidden weights fixed",
        "least_squares_rcond": RCOND,
        "rows": len(rows),
        "max_readout_normal_equation_error": max_normal_error,
        "nonpositive_predictor_train_mse_bound": float(np.mean(np.maximum(y, 0) ** 2)),
        "inputs_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
    }
    (OUTPUT / "saved_state_checks.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
