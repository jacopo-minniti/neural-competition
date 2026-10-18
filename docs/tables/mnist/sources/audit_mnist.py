#!/usr/bin/env python3
"""Every-update G/H/total-interaction audit on seed zero at both endpoint widths."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_mnist as study
import torch

from neural_competition.mnist import gradients, initialize, loss_rows, propose


@torch.jit.script
def dense_rollout(p: list[torch.Tensor], first: list[torch.Tensor], second: list[torch.Tensor],
                  x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor, xref: torch.Tensor,
                  yref: torch.Tensor, ce: bool, kind: int, lr: float):
    records = torch.empty((ids.shape[0], 5), dtype=torch.float64)
    norm = float(p[0].shape[0]) ** 0.5
    hidden = torch.relu(xref @ p[0].double().T + p[1].double())
    before = hidden[:, :, None] * p[2].double()[None, :, :] / norm
    z = before.sum(1)
    initial = loss_rows(z, yref, ce).mean()
    prior = initial
    for step in range(ids.shape[0]):
        g = gradients(p, x[ids[step]], y[ids[step]], ce)
        p, first, second = propose(p, g, first, second, kind, lr, step)
        hidden = torch.relu(xref @ p[0].double().T + p[1].double())
        after = hidden[:, :, None] * p[2].double()[None, :, :] / norm
        d = after - before
        if ce:
            alone = (loss_rows(z[:, None, :] + d, yref[:, None, :], True)
                     - loss_rows(z, yref, True)[:, None]).mean(0)
        else:
            alone = ((z - yref)[:, None, :] * d + 0.5 * d.square()).sum(-1).mean(0)
        z = after.sum(1)
        current = loss_rows(z, yref, ce).mean()
        records[step, 0] = current
        records[step, 1] = (-alone).clamp_min(0).sum()
        records[step, 2] = alone.clamp_min(0).sum()
        records[step, 3] = current - prior - alone.sum()
        records[step, 4] = prior - current
        before, prior = after, current
    return p, first, second, records, initial


def audit_one(task):
    root = Path(task["output"]).parents[1]
    output = root / "dense_audit" / Path(task["output"]).name
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    data = study.DATA
    p = initialize(task["width"], task["seed"])
    zeros = [torch.zeros_like(v) for v in p]
    ids = study.training_indices(task["seed"], task["epochs"], task["batch_size"])
    ref = data["train_reference_ids"]
    p, first, second, records, initial = dense_rollout(
        p, zeros, zeros, data["xtrain"], data["ytrain"], ids,
        data["xtrain"][ref].double(), data["ytrain"][ref].double(),
        task["loss"] == "ce", study.KINDS[task["optimizer"]], task["lr"])
    original = torch.load(Path(task["output"]) / f"checkpoint_{len(ids)}.pt", weights_only=True)
    for measured, reference in zip([p, first, second], [original["parameters"], original["first"],
                                                      original["second"]]):
        assert all(torch.equal(a, b) for a, b in zip(measured, reference))
    exact = records.numpy()
    np.savez_compressed(output / "every_update.npz", records=exact, initial=float(initial))
    estimated = pd.read_csv(Path(task["output"]) / "budgets.csv.gz").iloc[-1]
    names = ["gain", "harm", "net_all"]
    result = dict(task=task, replay_bitwise_equal=True,
                  telescoping_error=abs(float(records[:, 4].sum() - initial + records[-1, 0])))
    for name, column in zip(names, [1, 2, 3]):
        truth = float(exact[:, column].sum())
        sampled = float(estimated["cumulative_" + name])
        endpoint = float(estimated.cumulative_net_all_from_endpoints) if name == "net_all" else sampled
        se = float(estimated["sampling_se_signed_alone" if name == "net_all" else
                             "sampling_se_" + name])
        result[name] = dict(exact=truth, sampled=sampled, endpoint_estimator=endpoint,
                            relative_error=(endpoint - truth) / truth,
                            estimated_se=se, error_per_se=(endpoint - truth) / se)
    windows, _ = study.sampling_design(len(ids), task["seed"])
    rng = np.random.default_rng(20261021)
    trials = np.zeros((2000, 3))
    for left, right, _ in windows:
        size = right - left
        first = rng.integers(size, size=2000)
        second = rng.integers(size - 1, size=2000)
        second += second >= first
        trials += size * (exact[left + first, 1:4] + exact[left + second, 1:4]) / 2
    truth = exact[:, 1:4].sum(0)
    result["monte_carlo_relative_bias"] = ((trials.mean(0) - truth) / truth).tolist()
    result["monte_carlo_relative_sd"] = (trials.std(0, ddof=1) / np.abs(truth)).tolist()
    rows = []
    running = np.cumsum(exact[:, 1:4], axis=0)
    for _, right, _ in windows:
        rows.append(dict(completed=right, gain=running[right - 1, 0],
                         harm=running[right - 1, 1], net_all=running[right - 1, 2],
                         reference_loss=exact[right - 1, 0]))
    pd.DataFrame(rows).to_csv(output / "exact_curves.csv.gz", index=False)
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", dest="all_runs")
    parser.add_argument("--input", type=Path, default=study.ROOT / "runs/mnist")
    args = parser.parse_args()
    root = args.input
    tasks = [json.loads(path.read_text())["task"]
             for path in sorted((root / "main").glob("*/complete.json"))]
    if not args.all_runs:
        tasks = [task for task in tasks if task["seed"] == 0 and task["width"] in [16, 128]]
    assert len(tasks) == (120 if args.all_runs else 12)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,
                                               initializer=study.initialize_worker,
                                               initargs=(1,)) as pool:
        futures = [pool.submit(audit_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(loss=result["task"]["loss"],
                                  optimizer=result["task"]["optimizer"],
                                  width=result["task"]["width"],
                                  relative_errors={name: result[name]["relative_error"]
                                                   for name in ["gain", "harm", "net_all"]})),
                  flush=True)
    destination = study.ROOT / "docs/tables/mnist"
    destination.mkdir(parents=True, exist_ok=True)
    name = "dense_full.json" if args.all_runs else "dense_audit.json"
    (destination / name).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
