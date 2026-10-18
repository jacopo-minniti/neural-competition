#!/usr/bin/env python3
"""Paired MNIST width study, with independent rate tuning and sampled diagnostics."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_competition.mnist import (
    RESOURCES,
    advance,
    batch_transfer,
    evaluate,
    gradients,
    initialize,
    interaction_statistics,
    loss_rows,
    prepare_data,
    propose,
    scores,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = None
KINDS = {"sgd": 0, "momentum": 1, "adam": 2}
GRIDS = {"sgd": [0.03, 0.1, 0.3, 1.0], "momentum": [0.01, 0.03, 0.1, 0.3],
         "adam": [0.0003, 0.001, 0.003, 0.01]}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_worker(threads):
    global DATA
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    DATA = prepare_data(ROOT / "data/mnist")


def training_indices(seed, epochs, batch_size):
    rng = torch.Generator().manual_seed(70000 + seed)
    n = len(DATA["xtrain"])
    per_epoch = n // batch_size
    return torch.stack([torch.randperm(n, generator=rng)[:per_epoch * batch_size]
                        .reshape(per_epoch, batch_size) for _ in range(epochs)]).flatten(0, 1)


def sampling_design(total, seed):
    """Two SRSWOR updates per stratum; dense first 200, then 100-update strata."""
    rng = torch.Generator().manual_seed(80000 + seed)
    windows, selected = [], {}
    left = 0
    while left < total:
        right = min(total, left + (10 if left < 200 else 100))
        length = right - left
        positions = (torch.randperm(length, generator=rng)[:min(2, length)] + left + 1)
        window = len(windows)
        windows.append((left, right, sorted(positions.tolist())))
        for step in positions.tolist():
            selected[step] = dict(stratum=window, weight=length / len(positions))
        left = right
    return windows, selected


def run_one(task):
    start = time.monotonic()
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        assert old["task"] == task, output
        return old
    ce = task["loss"] == "ce"
    kind, lr = KINDS[task["optimizer"]], task["lr"]
    seed, width = task["seed"], task["width"]
    epochs, batch_size = task["epochs"], task["batch_size"]
    tuning = task.get("tuning", False)
    p = initialize(width, seed)
    first, second = [torch.zeros_like(v) for v in p], [torch.zeros_like(v) for v in p]
    indices = training_indices(seed, epochs, batch_size)
    steps_per_epoch = len(indices) // epochs
    total = len(indices)
    refids = DATA["train_reference_ids"]
    xref, yref = DATA["xtrain"][refids], DATA["ytrain"][refids]
    valids = DATA["val_reference_ids"]
    xheld, yheld = DATA["xval"][valids], DATA["yval"][valids]
    initial_reference = float(loss_rows(scores([v.double() for v in p], xref.double()),
                                        yref.double(), ce).mean())
    windows, selected = sampling_design(total, seed) if not tuning else ([], {})
    boundary_windows = {right: k for k, (_, right, _) in enumerate(windows)}
    epochs_at = {k * steps_per_epoch: k for k in range(epochs + 1)}
    probe_epochs = sorted(set([0, 1, 5, 15, epochs]))
    probe_steps = {k * steps_per_epoch for k in probe_epochs if k <= epochs}
    targets = set(epochs_at) if not tuning else {0, total}
    targets |= set(selected) | {s - 1 for s in selected} | set(boundary_windows)
    logs, diagnostics, transfers, tiny_steps, budgets = [], [], [], [], []
    completed = 0
    cumulative = {name: 0.0 for name in ["gain", "harm", "net_all", "pair_plus",
                                       "pair_minus", "pair_net", "strict_harm"]}
    variances = {name: 0.0 for name in ["gain", "harm", "signed_alone", "pair_net"]}
    stratum_records = {}
    diverged = False
    for end in sorted(targets):
        previous = None
        if end > completed:
            if end in selected:
                assert completed == end - 1
                previous = p
            p, first, second = advance(p, first, second, DATA["xtrain"], DATA["ytrain"],
                                       indices[completed:end], ce, kind, lr, completed)
            completed = end
        if not all(torch.isfinite(v).all() for v in p):
            diverged = True
            break
        if end in selected:
            design = selected[end]
            rng = torch.Generator().manual_seed(90000 + seed * 100000 + end)
            stats, _, _ = interaction_statistics(previous, p, xref, yref, ce, rng)
            record = dict(completed=end, **design, **stats)
            diagnostics.append(record)
            stratum_records.setdefault(design["stratum"], []).append(record)
            for name in cumulative:
                cumulative[name] += design["weight"] * stats[name]
        if end in boundary_windows:
            records = stratum_records[boundary_windows[end]]
            length = windows[boundary_windows[end]][1] - windows[boundary_windows[end]][0]
            for name in variances:
                values = [r["harm"] - r["gain"] if name == "signed_alone" else r[name]
                          for r in records]
                if len(values) > 1:
                    variances[name] += length**2 * (1 - len(values) / length) * float(
                        np.var(values, ddof=1)) / len(values)
            current = float(loss_rows(scores([v.double() for v in p], xref.double()),
                                      yref.double(), ce).mean())
            budgets.append(dict(
                completed=end, epoch=end / steps_per_epoch, reference_loss=current,
                initial_reference_loss=initial_reference,
                actual_progress=initial_reference - current,
                **{"cumulative_" + key: value for key, value in cumulative.items()},
                cumulative_net_all_from_endpoints=(current - initial_reference
                                                  - cumulative["harm"] + cumulative["gain"]),
                **{"sampling_se_" + key: math.sqrt(value) for key, value in variances.items()},
            ))
        if end in epochs_at:
            epoch = epochs_at[end]
            train = evaluate(p, DATA["xtrain"], DATA["ytrain"], ce)
            val = evaluate(p, DATA["xval"], DATA["yval"], ce)
            held = evaluate([v.double() for v in p], xheld.double(), yheld.double(), ce)
            logs.append(dict(completed=end, epoch=epoch,
                             train_loss=train["loss"], train_accuracy=train["accuracy"],
                             val_loss=val["loss"], val_accuracy=val["accuracy"],
                             reference_loss=float(loss_rows(
                                 scores([v.double() for v in p], xref.double()),
                                 yref.double(), ce).mean()), heldout_loss=held["loss"]))
        if not tuning and end in probe_steps:
            transfer = batch_transfer(p, first, second, DATA["xtrain"], DATA["ytrain"],
                                      ce, kind, lr, end, 100000 + seed * 100000 + end)
            transfers.append(dict(completed=end, epoch=end / steps_per_epoch, **transfer))
            # Fresh source at each snapshot. Full and 1% moves share one direction.
            rng = torch.Generator().manual_seed(110000 + seed * 100000 + end)
            source = torch.randint(len(DATA["xtrain"]), (batch_size,), generator=rng)
            gradient = gradients(p, DATA["xtrain"][source], DATA["ytrain"][source], ce)
            q, _, _ = propose(p, gradient, first, second, kind, lr, end)
            for fraction in [1.0, 0.01]:
                moved = [a + fraction * (b - a) for a, b in zip(p, q)]
                for view, x, y in [("reference", xref, yref), ("heldout", xheld, yheld)]:
                    pair_rng = torch.Generator().manual_seed(120000 + seed * 100000 + end)
                    stats, _, _ = interaction_statistics(p, moved, x, y, ce, pair_rng)
                    tiny_steps.append(dict(completed=end, epoch=end / steps_per_epoch,
                                           fraction=fraction, view=view, **stats))
            torch.save(dict(parameters=p, first=first, second=second, completed=end,
                            task=task), output / f"checkpoint_{end}.pt")
    test = evaluate(p, DATA["xtest"], DATA["ytest"], ce) if not diverged and not tuning else {
        "loss": None, "accuracy": None}
    for name, records in [("epochs", logs), ("diagnostics", diagnostics),
                          ("transfers", transfers), ("step_controls", tiny_steps),
                          ("budgets", budgets)]:
        if records:
            pd.DataFrame(records).to_csv(output / (name + ".csv.gz"), index=False,
                                         compression={"method": "gzip", "mtime": 0})
    if tuning:
        # Tuning never consults the official test split.
        test = {"loss": None, "accuracy": None}
    final = dict(task=task, diverged=diverged, completed=completed,
                 train=logs[-1] if logs else {}, test=test,
                 seconds=time.monotonic() - start,
                 max_budget_error=max((r["budget_error"] for r in diagnostics), default=0),
                 max_pair_identity_error=max((r["pair_identity_error"] or 0
                                              for r in diagnostics), default=0),
                 max_logit_additivity_error=max((r["logit_additivity_error"]
                                                for r in diagnostics), default=0))
    marker.write_text(json.dumps(final, indent=2) + "\n")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tune", "main"], default="main")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mnist")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.download:
        raw = ROOT / "data/mnist/raw"
        raw.mkdir(parents=True, exist_ok=True)
        for name, checksum in RESOURCES.items():
            path = raw / name
            if not path.exists():
                with urllib.request.urlopen("https://ossci-datasets.s3.amazonaws.com/mnist/" + name,
                                            timeout=60) as response:
                    path.write_bytes(response.read())
            assert hashlib.md5(path.read_bytes()).hexdigest() == checksum, name
    torch.set_num_threads(args.threads)
    data = prepare_data(ROOT / "data/mnist")
    sources = [Path(__file__), ROOT / "src/neural_competition/mnist.py"]
    source_dir = args.output / "executed_sources"
    source_dir.mkdir(exist_ok=True)
    for source in sources:
        destination = source_dir / source.name
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes(), "Execution source changed"
        else:
            destination.write_bytes(source.read_bytes())
    provenance = dict(torch_version=torch.__version__, cpu_count=os.cpu_count(),
                      workers=args.workers, threads_per_worker=args.threads,
                      sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                      prepared_data_sha256=sha256(ROOT / "data/mnist/prepared.pt"),
                      archives=RESOURCES, normalization_mean=data["mean"],
                      normalization_std=data["std"],
                      validation_per_class=500, batch_size=128, drop_last=True,
                      initialization_outgoing_std=0.1, width_scaling="1/sqrt(m)")
    (args.output / (args.phase + "_provenance.json")).write_text(
        json.dumps(provenance, indent=2) + "\n")
    tasks = []
    if args.phase == "tune":
        for loss in ["mse", "ce"]:
            for optimizer, rates in GRIDS.items():
                for width in [16, 128]:
                    for seed in [1000, 1001]:
                        for lr in rates:
                            name = f"{loss}_{optimizer}_w{width}_s{seed}_lr{lr:g}"
                            tasks.append(dict(loss=loss, optimizer=optimizer, width=width,
                                              seed=seed, lr=lr, epochs=5, batch_size=128,
                                              tuning=True,
                                              output=str(args.output / "tuning" / name)))
    else:
        rates = json.loads((args.output / "selected_rates.json").read_text())
        for loss in ["mse", "ce"]:
            for optimizer in KINDS:
                for width in args.widths:
                    for seed in range(args.seeds):
                        name = f"{loss}_{optimizer}_w{width}_s{seed}"
                        tasks.append(dict(loss=loss, optimizer=optimizer, width=width,
                                          seed=seed, lr=rates[loss][optimizer], epochs=args.epochs,
                                          batch_size=128, tuning=False,
                                          output=str(args.output / "main" / name)))
    start = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                               initializer=initialize_worker,
                                               initargs=(args.threads,)) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            task = result["task"]
            row = result["train"]
            print(json.dumps(dict(done=len(results), total=len(tasks), loss=task["loss"],
                                  optimizer=task["optimizer"], width=task["width"],
                                  seed=task["seed"], lr=task["lr"], diverged=result["diverged"],
                                  train_loss=row.get("train_loss"),
                                  val_accuracy=row.get("val_accuracy"),
                                  elapsed=round(time.monotonic() - start, 1))), flush=True)
    if args.phase == "tune":
        rows = []
        for result in results:
            task = result["task"]
            rows.append({**{k: task[k] for k in ["loss", "optimizer", "width", "seed", "lr"]},
                         "val_loss": result["train"].get("val_loss", float("inf"))
                         if not result["diverged"] else float("inf"),
                         "train_loss": result["train"].get("train_loss"),
                         "diverged": result["diverged"]})
        table = pd.DataFrame(rows).sort_values(["loss", "optimizer", "width", "seed", "lr"])
        table.to_csv(args.output / "tuning.csv", index=False)
        table["log_val_loss"] = np.log(table.val_loss)
        means = table.groupby(["loss", "optimizer", "lr"]).log_val_loss.mean()
        chosen = {}
        for loss in ["mse", "ce"]:
            chosen[loss] = {optimizer: float(means.loc[loss, optimizer].idxmin())
                            for optimizer in KINDS}
        (args.output / "selected_rates.json").write_text(json.dumps(chosen, indent=2) + "\n")
        print(json.dumps(chosen), flush=True)


if __name__ == "__main__":
    main()
