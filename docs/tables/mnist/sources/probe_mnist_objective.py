#!/usr/bin/env python3
"""Evaluate fresh matched-state updates on all 55,000 training examples."""

import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
import run_mnist as study
import torch

from neural_competition.mnist import gradients, loss_rows, propose, scores


def all_scores(p, x):
    p = [value.double() for value in p]
    return torch.cat([scores(p, x[left:left + 2048].double())
                      for left in range(0, len(x), 2048)])


def probe_one(path):
    checkpoint = torch.load(path, weights_only=True)
    task = checkpoint["task"]
    destination = path.parent / (path.stem + "_full_objective.csv.gz")
    if destination.exists():
        return str(destination)
    p, first, second = checkpoint["parameters"], checkpoint["first"], checkpoint["second"]
    ce, kind = task["loss"] == "ce", study.KINDS[task["optimizer"]]
    data = study.DATA
    baseline = all_scores(p, data["xtrain"])
    y = data["ytrain"].double()
    initial = loss_rows(baseline, y, ce)
    residual = baseline.softmax(-1) - y if ce else baseline - y
    draw = task["seed"] * 100000 + int(round(checkpoint["target_loss"] * 10000))
    metadata = {key: task[key] for key in ["loss", "optimizer", "width", "seed", "lr"]}
    rows = []
    for source in range(8):
        rng = torch.Generator().manual_seed(140000 + draw * 100 + source)
        ids = torch.randint(len(data["xtrain"]), (task["batch_size"],), generator=rng)
        g = gradients(p, data["xtrain"][ids], data["ytrain"][ids], ce)
        q, _, _ = propose(p, g, first, second, kind, task["lr"], checkpoint["completed"])
        for fraction in [1.0, 0.01]:
            moved = [a + fraction * (b - a) for a, b in zip(p, q)]
            following = all_scores(moved, data["xtrain"])
            delta = following - baseline
            progress = float((initial - loss_rows(following, y, ce)).mean())
            work = float(-(residual * delta).sum(-1).mean())
            rows.append(dict(**metadata, target_loss=checkpoint["target_loss"],
                             completed=checkpoint["completed"], source=source,
                             fraction=fraction, train_loss_before=float(initial.mean()),
                             progress=progress, linear_logit_work=work,
                             curvature_cost=work - progress,
                             prediction_rms=float(delta.square().sum(-1).mean().sqrt())))
    pd.DataFrame(rows).to_csv(destination, index=False,
                             compression={"method": "gzip", "mtime": 0})
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=study.ROOT / "runs/mnist")
    args = parser.parse_args()
    root = args.input / "matched"
    checkpoints = []
    for path in sorted(root.glob("*/target_*.pt")):
        if "_w16_" in str(path) or "_w128_" in str(path):
            checkpoints.append(path)
    assert len(checkpoints) == 119  # One SGD target cannot meet the 5% matching tolerance.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,
                                               initializer=study.initialize_worker,
                                               initargs=(1,)) as pool:
        futures = [pool.submit(probe_one, path) for path in checkpoints]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(i, len(checkpoints), future.result(), flush=True)
    frame = pd.concat([pd.read_csv(path.parent / (path.stem + "_full_objective.csv.gz"))
                       for path in checkpoints], ignore_index=True)
    destination = study.ROOT / "docs/tables/mnist/full_objective_probes.csv.gz"
    frame.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})


if __name__ == "__main__":
    main()
