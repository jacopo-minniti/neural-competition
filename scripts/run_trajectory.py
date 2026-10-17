#!/usr/bin/env python3
"""Measure every actual training update; retain compact scalar diagnostics."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_competition.experiment import output_lock, save_torch, write_json
from neural_competition.model import ShallowReLU
from neural_competition.trajectory import FIELDS, VIEWS, components, run_block


def evaluate(parameters, widths, x, y):
    """Full empirical loss in chunks, using the diagnostic float64 convention."""
    total = torch.zeros(len(parameters), dtype=torch.float64)
    for start in range(0, len(x), 512):
        xx = x[start : start + 512].double()[None].expand(len(parameters), -1, -1)
        prediction = components(parameters.double(), xx, widths.double().sqrt()).sum(-1)
        total += (prediction - y[None, start : start + 512].double()).square().sum(1)
    return total.numpy() / len(x)


def train(args, label, data, reference_indices):
    output = args.output / label
    output.mkdir(parents=True, exist_ok=True)
    widths = torch.tensor([w for w in [8, 16] for _ in range(args.seeds)])[:, None]
    seeds = torch.arange(args.seeds).repeat(2)
    models = [{"width": int(w), "seed": int(s)} for w, s in zip(widths[:, 0], seeds)]
    lr, momentum = (0.1, 0.9) if label == "momentum" else (1.0, 0.0)
    config = dict(
        steps=args.steps,
        models=models,
        batch_size=64,
        reference_size=args.reference,
        reference_seed=20261020,
        optimizer=label,
        lr=lr,
        momentum=momentum,
        fields=FIELDS,
        views=VIEWS,
        float_training="float32",
        float_metrics="float64",
        teacher_width=8,
        data_seed=2026,
        reference="fixed random training subset",
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
    )
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Existing run has a different configuration")
        if (output / "state.json").exists():
            if json.loads((output / "state.json").read_text())["status"] == "complete":
                print(f"{label}: already complete", flush=True)
                return
    write_json(output / "config.json", config)
    hashes = {
        name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
        for name in ["scripts/run_trajectory.py", "src/neural_competition/trajectory.py"]
    }
    write_json(output / "source_hashes.json", hashes)
    for name in hashes:
        frozen = output / "executed_sources" / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((Path(__file__).resolve().parents[1] / name).read_bytes())
    p = torch.zeros(len(models), 16, 22)
    for i, model in enumerate(models):
        p[i, : model["width"]] = ShallowReLU(20, model["width"], model["seed"]).parameter_blocks()
    velocity = torch.zeros_like(p)
    active = torch.arange(16)[None] < widths
    upper = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)[None]
    upper = upper & active[:, :, None] & active[:, None, :]
    generators = [torch.Generator().manual_seed(2026 + 10_000 + s) for s in range(args.seeds)]
    xref = data["x_train"][reference_indices].double()[None].expand(len(models), -1, -1)
    yref = data["y_train"][reference_indices].double()[None].expand(len(models), -1)
    shape = (args.steps, len(models), len(VIEWS), len(FIELDS))
    completed = 0
    evaluations = []
    elapsed_before = 0.0
    if (output / "resume.pt").exists():
        checkpoint = torch.load(output / "resume.pt", weights_only=True)
        p, velocity, completed = (
            checkpoint["parameters"],
            checkpoint["velocity"],
            checkpoint["completed"],
        )
        elapsed_before = checkpoint["elapsed_seconds"]
        for generator, state in zip(generators, checkpoint["rng_states"]):
            generator.set_state(state)
        records = np.lib.format.open_memmap(output / "updates.npy", mode="r+")
        if (output / "losses.csv").exists():
            old = pd.read_csv(output / "losses.csv")
            evaluations = old[old.updates <= completed].to_dict("records")
    else:
        records = np.lib.format.open_memmap(
            output / "updates.npy", mode="w+", dtype="float64", shape=shape
        )
        save_torch(output / "initial.pt", {"parameters": p, "reference_indices": reference_indices})
    started = time.perf_counter()
    if not evaluations:
        train_mse = evaluate(p, widths, data["x_train"], data["y_train"])
        test_mse = evaluate(p, widths, data["x_test"], data["y_test"])
        evaluations.extend(
            dict(updates=completed, **m, train_mse=float(tr), test_mse=float(te))
            for m, tr, te in zip(models, train_mse, test_mse)
        )
    for start in range(completed, args.steps, args.block):
        count = min(args.block, args.steps - start)
        indices = torch.stack(
            [torch.randint(8192, (count, 64), generator=g) for g in generators], dim=1
        )
        p, velocity, stats = run_block(
            p,
            velocity,
            data["x_train"],
            data["y_train"],
            indices,
            seeds,
            widths,
            xref,
            yref,
            upper,
            lr,
            momentum,
        )
        if not torch.isfinite(stats).all() or not torch.isfinite(p).all():
            raise RuntimeError(f"Nonfinite state at block {start}")
        records[start : start + count] = stats.numpy()
        completed = start + count
        train_mse = evaluate(p, widths, data["x_train"], data["y_train"])
        test_mse = evaluate(p, widths, data["x_test"], data["y_test"])
        evaluations.extend(
            dict(updates=completed, **m, train_mse=float(tr), test_mse=float(te))
            for m, tr, te in zip(models, train_mse, test_mse)
        )
        if completed % 10_000 == 0 or completed == args.steps:
            elapsed = elapsed_before + time.perf_counter() - started
            records.flush()
            pd.DataFrame(evaluations).to_csv(output / "losses.csv", index=False)
            save_torch(
                output / "resume.pt",
                dict(
                    parameters=p,
                    velocity=velocity,
                    completed=completed,
                    rng_states=[g.get_state() for g in generators],
                    elapsed_seconds=elapsed,
                ),
            )
            write_json(
                output / "state.json",
                dict(
                    status="complete" if completed == args.steps else "running",
                    updates=completed,
                    elapsed_seconds=elapsed,
                ),
            )
            medians = [
                np.median(train_mse[i * args.seeds : (i + 1) * args.seeds]) for i in range(2)
            ]
            print(
                f"{label}: {completed:,}/{args.steps:,}, {elapsed:.1f}s, median train MSE {medians}",
                flush=True,
            )
    del records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--reference", type=int, default=512)
    parser.add_argument("--block", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--optimizers", nargs="+", choices=["momentum", "sgd"], default=["momentum", "sgd"]
    )
    parser.add_argument("--output", type=Path, default=Path("runs/pier_study/trajectory"))
    parser.add_argument("--data", type=Path, default=Path("runs/pier_distributions/data.pt"))
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.reference, args.block) <= 0 or args.reference > 8192:
        raise ValueError("Invalid run sizes")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.data, weights_only=True)
    reference_indices = torch.randperm(8192, generator=torch.Generator().manual_seed(20261020))[
        : args.reference
    ]
    with output_lock(args.output):
        reference_path = args.output / "reference.pt"
        if reference_path.exists():
            saved_reference = torch.load(reference_path, weights_only=True)["indices"]
            if not torch.equal(saved_reference, reference_indices):
                raise ValueError("Existing reference differs; use a new output directory")
        else:
            save_torch(reference_path, {"indices": reference_indices})
        for label in args.optimizers:
            train(args, label, data, reference_indices)


if __name__ == "__main__":
    main()
