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
from neural_competition.trajectory import FIELDS, VIEWS, components, run_adam_block, run_block


def evaluate(parameters, widths, x, y):
    """Full empirical loss in chunks, using the diagnostic float64 convention."""
    total = torch.zeros(len(parameters), dtype=torch.float64)
    for start in range(0, len(x), 512):
        xx = x[start : start + 512].double()[None].expand(len(parameters), -1, -1)
        prediction = components(parameters.double(), xx, widths.double().sqrt()).sum(-1)
        total += (prediction - y[None, start : start + 512].double()).square().sum(1)
    return total.numpy() / len(x)


def train(
    args,
    label,
    data,
    reference_indices,
    *,
    student_widths=None,
    teacher_width=8,
    optimizer_spec=None,
    study=None,
    output_scale=0.0,
):
    output = args.output / label
    output.mkdir(parents=True, exist_ok=True)
    student_widths = student_widths or [8, 16]
    widths = torch.tensor([w for w in student_widths for _ in range(args.seeds)])[:, None]
    seeds = torch.arange(args.seeds).repeat(len(student_widths))
    models = [{"width": int(w), "seed": int(s)} for w, s in zip(widths[:, 0], seeds)]
    lr, momentum = (0.1, 0.9) if label == "momentum" else (1.0, 0.0)
    kind = label
    if optimizer_spec is not None:
        kind = optimizer_spec["name"]
        lr = optimizer_spec["lr"]
        momentum = optimizer_spec.get("momentum", 0.0)
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
        teacher_width=teacher_width,
        data_seed=2026,
        reference="fixed random training subset",
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
    )
    if optimizer_spec is not None:
        config["optimizer_spec"] = optimizer_spec
    if study is not None:
        config["study"] = study
    if output_scale:
        config["student_output_scale"] = output_scale
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Existing run has a different configuration")
        if (output / "state.json").exists():
            if json.loads((output / "state.json").read_text())["status"] == "complete":
                print(f"{label}: already complete", flush=True)
                return
    write_json(output / "config.json", config)
    hashes = {
        name: hashlib.sha256((Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest()
        for name in [
            "scripts/studies/run_trajectory.py",
            "src/neural_competition/trajectory.py",
            *getattr(args, "extra_sources", []),
        ]
    }
    write_json(output / "source_hashes.json", hashes)
    for name in hashes:
        frozen = output / "executed_sources" / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((Path(__file__).resolve().parents[2] / name).read_bytes())
    maximum = max(student_widths)
    dim = data["x_train"].shape[-1]
    p = torch.zeros(len(models), maximum, dim + 2)
    for i, model in enumerate(models):
        network = ShallowReLU(dim, model["width"], model["seed"], zero_output=output_scale == 0)
        initial = network.parameter_blocks()
        if output_scale:
            initial[:, -1].mul_(output_scale)
        p[i, : model["width"]] = initial
    velocity = torch.zeros_like(p)
    second = torch.zeros_like(p)
    active = torch.arange(maximum)[None] < widths
    upper = torch.triu(torch.ones(maximum, maximum, dtype=torch.bool), diagonal=1)[None]
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
        if kind == "adam":
            second = checkpoint["second"]
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
            [torch.randint(len(data["x_train"]), (count, 64), generator=g) for g in generators],
            dim=1,
        )
        if kind == "adam":
            p, velocity, second, stats = run_adam_block(
                p,
                velocity,
                second,
                data["x_train"],
                data["y_train"],
                indices,
                seeds,
                widths,
                xref,
                yref,
                upper,
                lr,
                start,
            )
        else:
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
            checkpoint = dict(
                parameters=p,
                velocity=velocity,
                completed=completed,
                rng_states=[g.get_state() for g in generators],
                elapsed_seconds=elapsed,
            )
            if kind == "adam":
                checkpoint["second"] = second
            save_torch(output / "resume.pt", checkpoint)
            if getattr(args, "keep_checkpoints", False):
                folder = output / "checkpoints"
                folder.mkdir(exist_ok=True)
                save_torch(folder / f"step_{completed:06d}.pt", checkpoint)
            write_json(
                output / "state.json",
                dict(
                    status="complete" if completed == args.steps else "running",
                    updates=completed,
                    elapsed_seconds=elapsed,
                ),
            )
            medians = [
                np.median(train_mse[i * args.seeds : (i + 1) * args.seeds])
                for i in range(len(student_widths))
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
    parser.add_argument("--data", type=Path, default=Path("runs/retained/datasets/k8_teacher2027.pt"))
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
