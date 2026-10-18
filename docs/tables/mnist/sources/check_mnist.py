#!/usr/bin/env python3
"""Matched full-training-loss probes and rate controls for the MNIST study."""

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_mnist as study
import torch

from neural_competition.mnist import (
    advance,
    batch_transfer,
    evaluate,
    gradients,
    initialize,
    interaction_statistics,
    propose,
)


def matched_one(task):
    main_output = Path(task["output"])
    collection = main_output.parent.name
    label = "matched" if collection == "main" else "matched_" + collection
    output = main_output.parents[1] / label / main_output.name
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    started = time.monotonic()
    data = study.DATA
    ce, kind = task["loss"] == "ce", study.KINDS[task["optimizer"]]
    width, seed, lr = task["width"], task["seed"], task["lr"]
    targets = [0.6, 0.3] if ce else [0.15, 0.10]
    remaining = targets.copy()
    indices = study.training_indices(seed, task["epochs"], task["batch_size"])
    p = initialize(width, seed)
    first, second = [torch.zeros_like(v) for v in p], [torch.zeros_like(v) for v in p]
    completed = 0
    previous_loss = evaluate(p, data["xtrain"], data["ytrain"], ce)["loss"]
    states = []
    while remaining and completed < len(indices):
        end = min(len(indices), completed + (50 if completed < 1000 else 200))
        old = (p, first, second)
        p, first, second = advance(*old, data["xtrain"], data["ytrain"],
                                   indices[completed:end], ce, kind, lr, completed)
        current_loss = evaluate(p, data["xtrain"], data["ytrain"], ce)["loss"]
        while remaining and current_loss <= remaining[0]:
            target = remaining.pop(0)
            assert previous_loss >= target
            left, right, left_state = completed, end, old
            right_state = (p, first, second)
            best = (abs(math.log(current_loss / target)), end, right_state, current_loss)
            while right - left > 1 and best[0] > math.log(1.025):
                mid = (left + right) // 2
                middle = advance(*left_state, data["xtrain"], data["ytrain"],
                                 indices[left:mid], ce, kind, lr, left)
                value = evaluate(middle[0], data["xtrain"], data["ytrain"], ce)["loss"]
                if abs(math.log(value / target)) < best[0]:
                    best = (abs(math.log(value / target)), mid, middle, value)
                if value > target:
                    left, left_state = mid, middle
                else:
                    right, right_state = mid, middle
            states.append((target, best[1], best[2], best[3]))
        previous_loss, completed = current_loss, end
    # Compare the replay with an original saved checkpoint, including moments.
    checkpoints = sorted(main_output.glob("checkpoint_*.pt"),
                         key=lambda path: int(path.stem.split("_")[-1]))
    audit_path = next(path for path in checkpoints if int(path.stem.split("_")[-1]) >= completed)
    original = torch.load(audit_path, weights_only=True)
    replayed = advance(p, first, second, data["xtrain"], data["ytrain"],
                       indices[completed:original["completed"]], ce, kind, lr, completed)
    max_error = 0.0
    for measured, reference in zip(replayed, [original["parameters"], original["first"],
                                            original["second"]]):
        for a, b in zip(measured, reference):
            max_error = max(max_error, float((a - b).abs().max()))
            assert torch.equal(a, b), (main_output, audit_path, max_error)
    metadata = {key: task[key] for key in ["loss", "optimizer", "width", "seed", "lr"]}
    transfers, interactions, coverage = [], [], []
    views = {}
    for view, split in [("reference", "train"), ("validation", "val")]:
        ids = data[split + "_reference_ids"]
        views[view] = (data["x" + split][ids], data["y" + split][ids])
    for target, step, state, achieved in states:
        matched = abs(achieved / target - 1) <= 0.05
        base = dict(**metadata, target_loss=target, completed=step, train_loss=achieved,
                    matched=matched, relative_loss_error=achieved / target - 1)
        coverage.append(base)
        if not matched:
            continue
        current, moment, variance = state
        draw = seed * 100000 + int(round(target * 10000))
        transfer = batch_transfer(current, moment, variance, data["xtrain"], data["ytrain"],
                                  ce, kind, lr, step, 130000 + draw, probes=64)
        transfers.append(dict(**base, **transfer))
        for source in range(16):
            rng = torch.Generator().manual_seed(140000 + draw * 100 + source)
            ids = torch.randint(len(data["xtrain"]), (task["batch_size"],), generator=rng)
            g = gradients(current, data["xtrain"][ids], data["ytrain"][ids], ce)
            q, _, _ = propose(current, g, moment, variance, kind, lr, step)
            for fraction in [1.0, 0.01]:
                moved = [a + fraction * (b - a) for a, b in zip(current, q)]
                for view, (x, y) in views.items():
                    pair_rng = torch.Generator().manual_seed(150000 + draw * 100 + source)
                    stats, _, _ = interaction_statistics(current, moved, x, y, ce, pair_rng)
                    interactions.append(dict(**base, source=source, fraction=fraction,
                                             view=view, **stats))
        torch.save(dict(parameters=current, first=moment, second=variance, completed=step,
                        task=task, target_loss=target, achieved_loss=achieved),
                   output / f"target_{target:g}.pt")
    for target in remaining:
        coverage.append(dict(**metadata, target_loss=target, matched=False,
                             train_loss=previous_loss, completed=completed,
                             relative_loss_error=previous_loss / target - 1))
    for name, rows in [("transfers", transfers), ("interactions", interactions),
                       ("coverage", coverage)]:
        pd.DataFrame(rows).to_csv(output / (name + ".csv.gz"), index=False,
                                 compression={"method": "gzip", "mtime": 0})
    result = dict(task=task, states=len(states), accepted=sum(row["matched"] for row in coverage),
                  targets=len(targets), replay_bitwise_equal=True, replay_max_error=max_error,
                  replay_checkpoint=str(audit_path), seconds=time.monotonic() - started)
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["matched", "rates"], default="matched")
    parser.add_argument("--input", type=Path, default=study.ROOT / "runs/mnist")
    parser.add_argument("--collection", default="main",
                        choices=["main", "lower_lr", "width_tuned"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    original = [json.loads(path.read_text())["task"]
                for path in sorted((args.input / args.collection).glob("*/complete.json"))]
    assert len(original) == (120 if args.collection == "main" else
                             60 if args.collection == "lower_lr" else 20)
    provenance = dict(sources={str(p.relative_to(study.ROOT)): study.sha256(p) for p in [
        Path(__file__), Path(study.__file__), study.ROOT / "src/neural_competition/mnist.py"]})
    (args.input / (args.phase + "_" + args.collection + "_provenance.json")).write_text(
        json.dumps(provenance, indent=2) + "\n")
    snapshot = args.input / "executed_sources" / Path(__file__).name
    if snapshot.exists():
        assert snapshot.read_bytes() == Path(__file__).read_bytes()
    else:
        snapshot.write_bytes(Path(__file__).read_bytes())
    if args.phase == "matched":
        tasks, function = original, matched_one
    else:
        tasks = []
        tuning = pd.read_csv(args.input / "tuning.csv")
        tuning["log_val_loss"] = np.log(tuning.val_loss)
        means = tuning.groupby(["loss", "optimizer", "width", "lr"]).log_val_loss.mean()
        for task in original:
            if task["width"] not in [16, 128]:
                continue
            best = float(means.loc[task["loss"], task["optimizer"], task["width"]].idxmin())
            choices = [("lower_lr", task["lr"] / 3)]
            if best != task["lr"]:
                choices.append(("width_tuned", best))
            for label, lr in choices:
                moved = {**task, "lr": lr,
                         "output": str(args.input / label / Path(task["output"]).name)}
                tasks.append(moved)
        function = study.run_one
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                               initializer=study.initialize_worker,
                                               initargs=(1,)) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            task = result["task"]
            print(json.dumps(dict(done=k, total=len(tasks), phase=args.phase,
                                  loss=task["loss"], optimizer=task["optimizer"],
                                  width=task["width"], seed=task["seed"],
                                  seconds=round(time.monotonic() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
