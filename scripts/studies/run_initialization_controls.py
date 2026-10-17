#!/usr/bin/env python3
"""Targeted follow-up to the Adam all-negative-output failure in the core sweep."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from run_trajectory import train

from neural_competition.experiment import output_lock, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/pier_study/robustness"))
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    # Do not mix source versions while the original sweep is still running.
    for job in plan["jobs"]:
        state = args.root / job["id"] / "state.json"
        if not state.exists() or json.loads(state.read_text())["status"] != "complete":
            raise RuntimeError("Finish the planned core sweep first")
    jobs = []
    for scale, tag in [(0.1, "init01"), (1.0, "init1")]:
        folder = args.root / f"k8_teacher2027_{tag}"
        folder.mkdir(exist_ok=True)
        for name in ["data.pt", "reference.pt", "teacher.json"]:
            if not (folder / name).exists():
                shutil.copyfile(args.root / "k8_teacher2027" / name, folder / name)
        jobs.append(
            dict(
                teacher_width=8,
                teacher_seed=2027,
                profile="adam_001",
                widths=[8, 16, 32],
                role="initialization",
                output_scale=scale,
                id=f"{folder.name}/adam_001",
                optimizer=dict(name="adam", lr=0.001, betas=[0.9, 0.999], eps=1e-8),
            )
        )
    write_json(
        args.root / "extension_plan.json",
        dict(
            reason="Follow-up to observed all-negative outgoing coefficients in narrow Adam models",
            jobs=jobs,
        ),
    )
    torch.set_num_threads(4)
    for job in jobs:
        folder = args.root / Path(job["id"]).parent
        data = torch.load(folder / "data.pt", weights_only=True)
        ref = torch.load(folder / "reference.pt", weights_only=True)["indices"]
        settings = SimpleNamespace(
            output=folder,
            data=folder / "data.pt",
            steps=plan["steps"],
            seeds=plan["seeds"],
            reference=plan["reference_size"],
            block=1000,
            keep_checkpoints=True,
            extra_sources=["scripts/studies/run_initialization_controls.py"],
        )
        destination = folder / job["profile"]
        destination.mkdir(exist_ok=True)
        print(f"START {job['id']}: output initialization scale {job['output_scale']}", flush=True)
        with output_lock(destination):
            train(
                settings,
                job["profile"],
                data,
                ref,
                student_widths=job["widths"],
                teacher_width=8,
                optimizer_spec=job["optimizer"],
                study=job,
                output_scale=job["output_scale"],
            )


if __name__ == "__main__":
    main()
