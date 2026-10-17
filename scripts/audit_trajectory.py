#!/usr/bin/env python3
"""Independent NumPy oracle, replay, and full-trajectory accounting checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_competition.experiment import write_json
from neural_competition.trajectory import momentum_step


def values(p, x, width):
    z = x @ p[:, :20].T + p[:, 20]
    return np.maximum(z, 0) * p[:, -1] / np.sqrt(width)


def oracle(before, after, y):
    d = after - before
    e = before.sum(1) - y
    end = after.sum(1) - y
    baseline = 0.5 * np.mean(e * e)
    a = np.mean(e[:, None] * d + 0.5 * d * d, axis=0)
    j = d.T @ d / len(d)
    upper = np.triu(np.ones_like(j, dtype=bool), 1)
    helpful = a < -1e-10 * baseline
    eligible = helpful[:, None] & helpful[None, :]
    marginal = a[None, :] + j
    strong = eligible & ~np.eye(len(a), dtype=bool) & (marginal > 1e-10 * baseline)
    out = np.array(
        [
            baseline,
            0.5 * np.mean(end * end),
            np.maximum(-a, 0).sum(),
            np.maximum(a, 0).sum(),
            np.maximum(j[upper], 0).sum(),
            np.maximum(-j[upper], 0).sum(),
            ((j > 1e-10 * baseline) & upper).sum(),
            upper.sum(),
            (eligible & upper).sum(),
            (eligible & upper & (j > 1e-10 * baseline)).sum(),
            strong.sum(),
            marginal[strong].sum(),
            0.5 * np.mean(d * d, axis=0).sum(),
            helpful.sum(),
        ]
    )
    return out, a, j


def main():
    root = Path("runs/pier_study/trajectory")
    data = torch.load("runs/pier_distributions/data.pt", weights_only=True)
    reference = torch.load(root / "reference.pt", weights_only=True)["indices"]
    torch.set_num_threads(4)
    checks = []
    max_metric_error = 0.0
    max_direct_error = 0.0
    source_checks = 0
    for label in ["momentum", "sgd"]:
        folder = root / label
        config = json.loads((folder / "config.json").read_text())
        raw = np.load(folder / "updates.npy", mmap_mode="r")
        p = torch.load(folder / "initial.pt", weights_only=True)["parameters"]
        velocity = torch.zeros_like(p)
        widths = torch.tensor([m["width"] for m in config["models"]])[:, None]
        norm = widths.float().sqrt()
        seeds = torch.tensor([m["seed"] for m in config["models"]])
        n_seeds = len(set(seeds.tolist()))
        generators = [torch.Generator().manual_seed(12026 + s) for s in range(n_seeds)]
        for step in range(200):
            indices = torch.stack([torch.randint(8192, (64,), generator=g) for g in generators])[
                seeds
            ]
            x, y = data["x_train"][indices], data["y_train"][indices]
            q, velocity = momentum_step(p, velocity, x, y, norm, config["lr"], config["momentum"])
            if step in [0, 29, 100, 199]:
                for i, model in enumerate(config["models"]):
                    width = model["width"]
                    pp = p[i, :width].double().numpy()
                    qq = q[i, :width].double().numpy()
                    for v, view in enumerate(["source", "reference"]):
                        xx = (
                            x[i].double().numpy()
                            if v == 0
                            else data["x_train"][reference].double().numpy()
                        )
                        yy = (
                            y[i].double().numpy()
                            if v == 0
                            else data["y_train"][reference].double().numpy()
                        )
                        before, after = values(pp, xx, width), values(qq, xx, width)
                        actual, a, j = oracle(before, after, yy)
                        np.testing.assert_allclose(actual, raw[step, i, v], atol=2e-11, rtol=1e-10)
                        max_metric_error = max(
                            max_metric_error, float(np.abs(actual - raw[step, i, v]).max())
                        )
                        for r, s in [(0, 1), (width - 2, width - 1)]:
                            e = before.sum(1) - yy
                            dr, ds = after[:, r] - before[:, r], after[:, s] - before[:, s]
                            direct = (
                                0.5 * np.mean((e + dr + ds) ** 2)
                                - 0.5 * np.mean((e + dr) ** 2)
                                - 0.5 * np.mean((e + ds) ** 2)
                                + 0.5 * np.mean(e**2)
                            )
                            max_direct_error = max(max_direct_error, float(abs(direct - j[r, s])))
                            source_checks += 1
                        checks.append(dict(optimizer=label, step=step, view=view, **model))
            p = q
        hashes = json.loads((folder / "source_hashes.json").read_text())
        for name, expected in hashes.items():
            content = (folder / "executed_sources" / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == expected
    full = pd.read_csv(root / "analysis/audit.csv")
    assert full.max_local_error.max() < 1e-10
    assert full.telescope_error.abs().max() < 1e-9
    assert full.reference_adjacency_error.max() == 0
    assert max_direct_error < 1e-12
    result = dict(
        independent_state_view_checks=len(checks),
        direct_four_loss_checks=source_checks,
        independent_numpy_max_error=max_metric_error,
        direct_four_loss_max_error=max_direct_error,
        all_updates_local_max_error=full.max_local_error.max(),
        accumulated_reference_telescope_max_error=full.telescope_error.abs().max(),
        reference_adjacency_error=full.reference_adjacency_error.max(),
        frozen_execution_sources_verified=True,
    )
    write_json(root / "analysis/independent_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
