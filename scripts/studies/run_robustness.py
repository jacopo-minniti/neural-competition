#!/usr/bin/env python3
"""Small factorial robustness sweep using the exact trajectory diagnostics."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from run_trajectory import train

from neural_competition.experiment import output_lock, save_torch, write_json
from neural_competition.model import ShallowReLU

PROFILES = {
    "momentum_01": dict(name="momentum", lr=0.1, momentum=0.9),
    "momentum_003": dict(name="momentum", lr=0.03, momentum=0.9),
    "adam_001": dict(name="adam", lr=0.001, betas=[0.9, 0.999], eps=1e-8),
    "adam_01": dict(name="adam", lr=0.01, betas=[0.9, 0.999], eps=1e-8),
}


def make_plan(args):
    jobs = []
    for teacher in [8, 16]:
        for profile in ["momentum_01", "adam_001", "adam_01"]:
            jobs.append(
                dict(
                    teacher_width=teacher,
                    teacher_seed=2027,
                    profile=profile,
                    widths=[teacher, teacher + 2, 2 * teacher, 4 * teacher],
                    role="core",
                )
            )
    jobs.append(
        dict(
            teacher_width=8,
            teacher_seed=2027,
            profile="momentum_003",
            widths=[8, 16, 32],
            role="learning_rate_control",
        )
    )
    for teacher in [8, 16]:
        for profile in ["momentum_01", "adam_001"]:
            jobs.append(
                dict(
                    teacher_width=teacher,
                    teacher_seed=2037,
                    profile=profile,
                    widths=[teacher, 2 * teacher],
                    role="teacher_realization",
                )
            )
    loads = [0] * args.workers
    for job in sorted(jobs, key=lambda j: len(j["widths"]) * max(j["widths"]) ** 2, reverse=True):
        partition = min(range(args.workers), key=lambda i: loads[i])
        job["partition"] = partition
        loads[partition] += len(job["widths"]) * max(job["widths"]) ** 2
        job["id"] = f"k{job['teacher_width']}_teacher{job['teacher_seed']}/{job['profile']}"
        job["optimizer"] = PROFILES[job["profile"]]
    return dict(
        steps=args.steps,
        seeds=args.seeds,
        reference_size=args.reference,
        reference_seed=20261020,
        data_seed=2026,
        batch_size=64,
        workers=args.workers,
        base_data_sha256=hashlib.sha256(args.base_data.read_bytes()).hexdigest(),
        jobs=jobs,
    )


def prepare(args, plan):
    args.output.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output):
        path = args.output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("Existing plan differs; use a new output directory")
        write_json(path, plan)
        base = torch.load(args.base_data, weights_only=True)
        reference = torch.randperm(8192, generator=torch.Generator().manual_seed(20261020))[
            : args.reference
        ]
        for k, seed in sorted({(j["teacher_width"], j["teacher_seed"]) for j in plan["jobs"]}):
            folder = args.output / f"k{k}_teacher{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "data.pt").exists():
                continue
            teacher = ShallowReLU(20, k, seed, zero_output=False)
            with torch.no_grad():
                teacher.output.div_(teacher(base["x_train"]).std().clamp_min(1e-8))
                ytrain, ytest = teacher(base["x_train"]), teacher(base["x_test"])
            if k == 8 and seed == 2027:
                torch.testing.assert_close(ytrain, base["y_train"], rtol=0, atol=0)
            save_torch(
                folder / "data.pt",
                dict(
                    x_train=base["x_train"],
                    x_test=base["x_test"],
                    y_train=ytrain,
                    y_test=ytest,
                    teacher_state=teacher.state_dict(),
                ),
            )
            save_torch(folder / "reference.pt", {"indices": reference})
            write_json(
                folder / "teacher.json",
                dict(
                    teacher_width=k,
                    teacher_seed=seed,
                    target_mean=ytrain.mean().item(),
                    target_std=ytrain.std().item(),
                    negative_target_fraction=(ytrain < 0).float().mean().item(),
                ),
            )
    print(
        f"Prepared {len(plan['jobs'])} groups, {sum(len(j['widths']) * args.seeds for j in plan['jobs'])} runs",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--reference", type=int, default=256)
    parser.add_argument("--block", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--partition", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/pier_study/robustness"))
    parser.add_argument("--base-data", type=Path, default=Path("runs/retained/datasets/k8_teacher2027.pt"))
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.reference, args.block, args.workers) <= 0:
        raise ValueError("Sizes must be positive")
    torch.set_num_threads(args.threads)
    plan = make_plan(args)
    if args.prepare_only or not (args.output / "plan.json").exists():
        prepare(args, plan)
    if json.loads((args.output / "plan.json").read_text()) != plan:
        raise ValueError("Saved plan differs")
    if args.prepare_only:
        return
    if args.partition is not None and not 0 <= args.partition < args.workers:
        raise ValueError("Invalid partition")
    for job in plan["jobs"]:
        if args.partition is not None and job["partition"] != args.partition:
            continue
        settings = copy.copy(args)
        settings.output = args.output / f"k{job['teacher_width']}_teacher{job['teacher_seed']}"
        settings.data = settings.output / "data.pt"
        settings.keep_checkpoints = True
        settings.extra_sources = ["scripts/studies/run_robustness.py"]
        data = torch.load(settings.data, weights_only=True)
        reference = torch.load(settings.output / "reference.pt", weights_only=True)["indices"]
        destination = settings.output / job["profile"]
        destination.mkdir(exist_ok=True)
        print(f"START {job['id']}: widths={job['widths']}, {args.seeds} seeds", flush=True)
        with output_lock(destination):
            train(
                settings,
                job["profile"],
                data,
                reference,
                student_widths=job["widths"],
                teacher_width=job["teacher_width"],
                optimizer_spec=job["optimizer"],
                study={key: job[key] for key in ["teacher_seed", "teacher_width", "role", "id"]},
            )


if __name__ == "__main__":
    main()
