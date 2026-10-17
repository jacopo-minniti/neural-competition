#!/usr/bin/env python3
"""Replay saved updates, verify optimizer states, and inspect neuron polarity."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_trajectory import oracle, values

from neural_competition.experiment import write_json
from neural_competition.model import ShallowReLU, loss
from neural_competition.trajectory import adam_step, batch_gradient, momentum_step


def jobs_at(root):
    jobs = json.loads((root / "plan.json").read_text())["jobs"]
    extension = root / "extension_plan.json"
    if extension.exists():
        jobs += json.loads(extension.read_text())["jobs"]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    root = Path("runs/pier_study/robustness")
    out = root / "analysis"
    torch.set_num_threads(4)
    maximum = dict(oracle=0.0, four_loss=0.0, gradient=0.0, optimizer=0.0)
    n_views = 0
    n_pairs = 0
    n_optimizers = 0
    structures = []
    for job in jobs_at(root):
        folder = root / job["id"]
        config = json.loads((folder / "config.json").read_text())
        state_path = folder / "state.json"
        if not state_path.exists() or json.loads(state_path.read_text())["status"] != "complete":
            if args.completed_only:
                continue
            raise RuntimeError(f"Incomplete: {job['id']}")
        for name, expected in json.loads((folder / "source_hashes.json").read_text()).items():
            assert (
                hashlib.sha256((folder / "executed_sources" / name).read_bytes()).hexdigest()
                == expected
            )
        data = torch.load(folder.parent / "data.pt", weights_only=True)
        ref = torch.load(folder.parent / "reference.pt", weights_only=True)["indices"]
        raw = np.load(folder / "updates.npy", mmap_mode="r")
        models = config["models"]
        initial = torch.load(folder / "initial.pt", weights_only=True)["parameters"]
        scale = config.get("student_output_scale", 0.0)
        for i, model in enumerate(models):
            expected = ShallowReLU(
                20, model["width"], model["seed"], zero_output=scale == 0
            ).parameter_blocks()
            if scale:
                expected[:, -1].mul_(scale)
            torch.testing.assert_close(initial[i, : model["width"]], expected, rtol=0, atol=0)
        widths = torch.tensor([m["width"] for m in models])[:, None]
        norm = widths.double().sqrt().float()
        seeds = torch.tensor([m["seed"] for m in models])
        spec = config["optimizer_spec"]
        for step in [10_000, 50_000, 90_000]:
            checkpoint = torch.load(
                folder / "checkpoints" / f"step_{step:06d}.pt", weights_only=True
            )
            p = checkpoint["parameters"]
            first = checkpoint["velocity"]
            ids = torch.stack(
                [
                    torch.randint(8192, (64,), generator=torch.Generator().set_state(s))
                    for s in checkpoint["rng_states"]
                ]
            )[seeds]
            x, y = data["x_train"][ids], data["y_train"][ids]
            if spec["name"] == "adam":
                q, _, _ = adam_step(
                    p, first, checkpoint["second"], x, y, norm, spec["lr"], step + 1
                )
            else:
                q, _ = momentum_step(p, first, x, y, norm, spec["lr"], spec["momentum"])
            gradient = batch_gradient(p, x, y, norm)
            for i, model in enumerate(models):
                w = model["width"]
                pp = p[i, :w].double().numpy()
                qq = q[i, :w].double().numpy()
                for v in [0, 1]:
                    xx = x[i].double().numpy() if v == 0 else data["x_train"][ref].double().numpy()
                    yy = y[i].double().numpy() if v == 0 else data["y_train"][ref].double().numpy()
                    before, after = values(pp, xx, w), values(qq, xx, w)
                    actual, a, j = oracle(before, after, yy)
                    np.testing.assert_allclose(actual, raw[step, i, v], atol=2e-11, rtol=1e-8)
                    maximum["oracle"] = max(
                        maximum["oracle"], float(np.abs(actual - raw[step, i, v]).max())
                    )
                    e = before.sum(1) - yy
                    dr, ds = after[:, 0] - before[:, 0], after[:, -1] - before[:, -1]
                    direct = (
                        0.5 * np.mean((e + dr + ds) ** 2)
                        - 0.5 * np.mean((e + dr) ** 2)
                        - 0.5 * np.mean((e + ds) ** 2)
                        + 0.5 * np.mean(e**2)
                    )
                    maximum["four_loss"] = max(maximum["four_loss"], float(abs(direct - j[0, -1])))
                    n_views += 1
                    n_pairs += 1
                if model["seed"] == 0:
                    network = ShallowReLU(20, w, 0)
                    with torch.no_grad():
                        network.weight.copy_(p[i, :w, :20])
                        network.bias.copy_(p[i, :w, 20])
                        network.output.copy_(p[i, :w, -1])
                    opt = (
                        torch.optim.Adam(network.parameters(), lr=spec["lr"], foreach=False)
                        if spec["name"] == "adam"
                        else torch.optim.SGD(
                            network.parameters(), lr=spec["lr"], momentum=spec["momentum"]
                        )
                    )
                    loss(network, x[i], y[i]).backward()
                    automatic = network.blocks(tuple(v.grad for v in network.parameters()))
                    maximum["gradient"] = max(
                        maximum["gradient"], float((automatic - gradient[i, :w]).abs().max())
                    )
                    for parameter, g, m, v in zip(
                        network.parameters(),
                        [gradient[i, :w, :20], gradient[i, :w, 20], gradient[i, :w, -1]],
                        [first[i, :w, :20], first[i, :w, 20], first[i, :w, -1]],
                        (
                            [
                                checkpoint["second"][i, :w, :20],
                                checkpoint["second"][i, :w, 20],
                                checkpoint["second"][i, :w, -1],
                            ]
                            if spec["name"] == "adam"
                            else [None] * 3
                        ),
                    ):
                        parameter.grad = g.clone()
                        if spec["name"] == "adam":
                            opt.state[parameter].update(
                                step=torch.tensor(float(step)),
                                exp_avg=m.clone(),
                                exp_avg_sq=v.clone(),
                            )
                        else:
                            opt.state[parameter]["momentum_buffer"] = m.clone()
                    opt.step()
                    error = float((network.parameter_blocks() - q[i, :w]).abs().max())
                    maximum["optimizer"] = max(maximum["optimizer"], error)
                    n_optimizers += 1
        state = torch.load(folder / "resume.pt", weights_only=True)
        for i, model in enumerate(models):
            w = model["width"]
            p = state["parameters"][i, :w].double()
            z = data["x_train"].double() @ p[:, :20].T + p[:, 20]
            active = (z > 0).any(0)
            prediction = z.relu() @ p[:, -1] / w**0.5
            structures.append(
                dict(
                    condition=job["id"],
                    teacher_width=job["teacher_width"],
                    teacher_seed=job["teacher_seed"],
                    profile=job["profile"],
                    role=job["role"],
                    **model,
                    positive_outputs=int((p[:, -1] > 0).sum()),
                    positive_active=int(((p[:, -1] > 0) & active).sum()),
                    inactive_on_train=int((~active).sum()),
                    global_nonpositive=bool((p[:, -1] <= 0).all()),
                    predicted_positive_fraction=float((prediction > 0).double().mean()),
                    target_positive_fraction=float((data["y_train"] > 0).double().mean()),
                    nonpositive_predictor_mse_bound=float(
                        data["y_train"].clamp_min(0).double().square().mean()
                    ),
                    final_mse=float((prediction - data["y_train"].double()).square().mean()),
                )
            )
    accounting = pd.read_csv(out / "audit.csv")
    assert maximum["oracle"] < 1e-9 and maximum["four_loss"] < 1e-10
    assert maximum["gradient"] < 2e-5 and maximum["optimizer"] < 2e-6
    assert accounting.telescope_error.abs().max() < 1e-8
    pd.DataFrame(structures).to_csv(out / "neuron_structure.csv", index=False)
    result = dict(
        independent_state_view_checks=n_views,
        direct_pair_checks=n_pairs,
        restored_optimizer_checks=n_optimizers,
        max_errors=maximum,
        telescoping_error=accounting.telescope_error.abs().max(),
        frozen_sources_verified=True,
    )
    write_json(out / "independent_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
