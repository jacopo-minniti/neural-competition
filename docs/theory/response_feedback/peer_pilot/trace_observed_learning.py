"""Observed-learning ledger: ordinary MNIST SGD and full-distribution toy GD.

New output directories only. Protocol/source snapshots precede training.
Fresh seeds test descriptive identities and measurement errors, not forecasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from explore_conditional_editability import Net
from explore_stable_distinctions import task
from torch.nn import functional as F

from neural_competition.learning_ledger import cross_entropy_ledger, dot, panel_signal

ROOT = Path(__file__).resolve().parents[2]


def snapshot(out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserving existing experiment; select a new output directory")
    sources = [
        Path(__file__),
        ROOT / "src/neural_competition/learning_ledger.py",
        Path(__file__).with_name("explore_conditional_editability.py"),
        Path(__file__).with_name("explore_stable_distinctions.py"),
    ]
    protocol["sources"] = {}
    for path in sources:
        content = path.read_bytes()
        protocol["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        (out / path.name).write_bytes(content)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def state(model):
    return tuple(v.detach().double().clone() for v in model.parameters())


def forward(parameters, x):
    w, b, a, c = parameters
    pre = x @ w.T + b
    act = pre.relu()
    return act @ a.T / np.sqrt(len(w)) + c, pre, act


def gradients(parameters, x, y, cached):
    """Analytic full gradient and per-example norm; independently autograd tested."""
    logits, pre, act = cached
    w, _, a, _ = parameters
    residual = logits.softmax(-1) - F.one_hot(y, 10)
    back = (residual @ a) * (pre > 0) / np.sqrt(len(w))
    n = len(x)
    gradient = (
        back.T @ x / n,
        back.mean(0),
        residual.T @ act / (n * np.sqrt(len(w))),
        residual.mean(0),
    )
    norm2 = back.square().sum(1) * (x.square().sum(1) + 1) + residual.square().sum(1) * (
        act.square().sum(1) / len(w) + 1
    )
    return gradient, norm2


def tangent(parameters, x, cached, delta):
    w, _, a, _ = parameters
    _, pre, act = cached
    dw, db, da, dc = delta
    return (((x @ dw.T + db) * (pre > 0)) @ a.T + act @ da.T) / np.sqrt(len(w)) + dc


def audit_panels(
    out, seed, width, step, parameters, x, y, cached, full_gradient, norm2, repetitions
):
    generator = torch.Generator().manual_seed(283000 + step)
    total = float(dot(full_gradient, full_gradient))
    population_variance = float((norm2.mean() - total) * len(x) / (len(x) - 1))
    # Uniform nested panels within each repetition; repeats are measurement draws.
    for repeat in range(repetitions):
        order = torch.randperm(len(x), generator=generator)
        for size in [64, 256, 1024]:
            ids = order[:size]
            gp, _ = gradients(parameters, x[ids], y[ids], tuple(v[ids] for v in cached))
            own, corrected, estimated_bias = panel_signal(gp, norm2[ids].mean(), size, len(x))
            append(
                out / "panels.jsonl",
                dict(
                    seed=seed,
                    width=width,
                    step=step,
                    repeat=repeat,
                    size=size,
                    full_signal=total,
                    own=float(own),
                    corrected=float(corrected),
                    estimated_bias=float(estimated_bias),
                    expected_bias=(1 / size - 1 / len(x)) * population_variance,
                    cross=float(dot(gp, full_gradient)),
                    population_variance=population_variance,
                ),
            )


def mnist(args):
    out = ROOT / args.output
    seeds = list(range(args.seed, args.seed + args.count))
    protocol = dict(
        study="MNIST finite-step ledger",
        seeds=seeds,
        widths=[32, 128],
        steps=args.steps,
        rate=0.05,
        batch=128,
        training_examples=8192,
        validation_examples=1024,
        panels=[64, 256, 1024],
        panel_repeats=args.repeats,
        audit_steps=[0, 20, 100, 400, 800],
        evaluation="float64 forward on float32 SGD parameters",
        purpose="Descriptive cumulative accounting and panel-error test; no outcome predictor fitted",
        sampling="SGD with replacement; audit panels without replacement, paired across widths",
        hypotheses=[
            "Small-panel own response overstates full-gradient signal by sampling variance.",
            "Unbiased correction may remain noisy, especially for width differences.",
            "Every-step loss ledger closes; first-order sum alone need not close.",
        ],
    )
    snapshot(out, protocol)
    data_path = ROOT / "data/mnist/prepared.pt"
    (out / "data_sha256.txt").write_text(hashlib.sha256(data_path.read_bytes()).hexdigest() + "\n")
    data = torch.load(data_path, weights_only=True)
    x, y = data["xtrain"][:8192], data["train_labels"][:8192]
    xd = x.double()
    vx, vy = data["xval"][:1024].double(), data["val_labels"][:1024]
    for seed in seeds:
        for width in [32, 128]:
            torch.manual_seed(seed)
            model = Net(width)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            generator = torch.Generator().manual_seed(seed + 500)
            old = state(model)
            cached = forward(old, xd)
            rows = []
            started = time.monotonic()
            for step in range(args.steps + 1):
                full_gradient, norm2 = gradients(old, xd, y, cached)
                signal = float(dot(full_gradient, full_gradient))
                if step in protocol["audit_steps"] or step == args.steps:
                    audit_panels(
                        out,
                        seed,
                        width,
                        step,
                        old,
                        xd,
                        y,
                        cached,
                        full_gradient,
                        norm2,
                        args.repeats,
                    )
                    vloss = float(F.cross_entropy(forward(old, vx)[0], vy))
                    append(
                        out / "checkpoints.jsonl",
                        dict(
                            seed=seed,
                            width=width,
                            step=step,
                            loss=float(F.cross_entropy(cached[0], y)),
                            validation_loss=vloss,
                            signal=signal,
                            example_norm2=float(norm2.mean()),
                        ),
                    )
                    torch.save(model.state_dict(), out / f"model_{seed}_{width}_{step}.pt")
                if step == args.steps:
                    break
                ids = torch.randint(len(x), (128,), generator=generator)
                optimizer.zero_grad()
                F.cross_entropy(model(x[ids]), y[ids]).backward()
                batch_gradient = tuple(v.grad.detach().double().clone() for v in model.parameters())
                optimizer.step()
                new = state(model)
                delta = tuple(b - a for a, b in zip(old, new))
                new_cached = forward(new, xd)
                row = cross_entropy_ledger(
                    cached[0], new_cached[0], y, tangent(old, xd, cached, delta)
                )
                row.update(
                    signal=0.05 * signal,
                    sampling=0.05 * float(dot(full_gradient, batch_gradient)) - 0.05 * signal,
                    rounding=float(-dot(full_gradient, delta))
                    - 0.05 * float(dot(full_gradient, batch_gradient)),
                )
                rows.append([row[key] for key in row])
                old, cached = new, new_cached
                if (step + 1) % 100 == 0:
                    print(
                        json.dumps(
                            dict(
                                seed=seed,
                                width=width,
                                step=step + 1,
                                loss=row["next_loss"],
                                seconds=round(time.monotonic() - started, 1),
                            )
                        ),
                        flush=True,
                    )
            np.savez_compressed(
                out / f"ledger_{seed}_{width}.npz",
                columns=np.array(list(row)),
                values=np.array(rows),
            )
    (out / "complete.json").write_text("{}\n")


def toy_forward(s, x):
    w, a, c = s
    pre = w @ x.T
    return c[:, None] + np.mean(a[..., None] * pre.clip(0), axis=1), pre


def toy_velocity(s, x, p, residual):
    w, a, _ = s
    pre = w @ x.T
    return (
        a[..., None] * (((pre > 0) * (p * residual)[:, None]) @ x),
        (pre.clip(0) * (p * residual)[:, None]).sum(2),
        (p * residual).sum(1),
    )


def toy_tangent(s, x, velocity):
    w, a, _ = s
    dw, da, dc = velocity
    pre = w @ x.T
    return dc[:, None] + np.mean(
        da[..., None] * pre.clip(0) + a[..., None] * (pre > 0) * (dw @ x.T), axis=1
    )


def toy(args):
    out = ROOT / args.output
    protocol = dict(
        study="Full-distribution finite-step toy ledger",
        seed=args.seed,
        count=args.count,
        widths=[4, 8],
        dt=args.dt,
        horizon=args.horizon,
        cases=["discrete", "continuous", "no_refinement"],
        scale=0.002,
        purpose="Fresh descriptive component ledgers; task contrasts defined before outcomes",
        columns=[
            "loss",
            "main_loss",
            "detail_loss",
            "other_loss",
            "signal",
            "nonlinear",
            "cost",
            "detail_self",
            "detail_cross",
            "detail_nonlinear",
            "detail_cost",
        ],
        checkpoints=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    )
    snapshot(out, protocol)
    raw = {}
    for name, nodes, spread, detail in [
        ("discrete", 1, 0.0, 2.0),
        ("continuous", 3, 0.15, 2.0),
        ("no_refinement", 1, 0.0, 0.0),
    ]:
        x, p, _ = task(nodes, spread, 0.1)
        y = 2 * x[:, 1] + detail * (x[:, 0] > 0) * x[:, 1]
        vx, vp, _ = task(9, spread, 0.1)
        vy = 2 * vx[:, 1] + detail * (vx[:, 0] > 0) * vx[:, 1]
        z = x[:, 1] / np.sqrt(p @ x[:, 1] ** 2)
        d = ((x[:, 0] > 0) - 0.1) * z / 0.3
        rng = np.random.default_rng(args.seed)
        iw = 0.002 * rng.normal(size=(args.count, 8, 3))
        ia = 0.002 * rng.normal(size=(args.count, 8))
        for width in [4, 8]:
            s = iw[:, :width].copy(), ia[:, :width].copy(), np.zeros(args.count)
            cumulative = np.zeros((args.count, 7))
            trace, times = [], []
            max_closure = 0.0
            for step in range(round(args.horizon / args.dt) + 1):
                f, _ = toy_forward(s, x)
                e = y - f
                loss = 0.5 * (p * e**2).sum(1)
                main = e @ (p * z)
                coefficient = e @ (p * d)
                detail_loss = 0.5 * coefficient**2
                t = step * args.dt
                if (
                    t <= 20
                    and step % max(1, round(0.25 / args.dt)) == 0
                    or any(abs(t - v) < 1e-8 for v in protocol["checkpoints"])
                    or step == round(args.horizon / args.dt)
                ):
                    trace.append(
                        np.column_stack(
                            [
                                loss,
                                0.5 * main**2,
                                detail_loss,
                                loss - 0.5 * main**2 - detail_loss,
                                cumulative.copy(),
                            ]
                        )
                    )
                    times.append(t)
                if step == round(args.horizon / args.dt):
                    break
                velocity = toy_velocity(s, x, p, e)
                linear = args.dt * toy_tangent(s, x, velocity)
                new = tuple(v + args.dt * dv for v, dv in zip(s, velocity))
                change = toy_forward(new, x)[0] - f
                signal = (p * e * linear).sum(1)
                nonlinear = (p * e * (change - linear)).sum(1)
                cost = 0.5 * (p * change**2).sum(1)
                self_velocity = toy_velocity(s, x, p, coefficient[:, None] * d)
                self_change = args.dt * toy_tangent(s, x, self_velocity) @ (p * d)
                dlinear = linear @ (p * d)
                dchange = change @ (p * d)
                cumulative += np.column_stack(
                    [
                        signal,
                        nonlinear,
                        cost,
                        coefficient * self_change,
                        coefficient * (dlinear - self_change),
                        coefficient * (dchange - dlinear),
                        0.5 * dchange**2,
                    ]
                )
                after_loss = 0.5 * (p * (e - change) ** 2).sum(1)
                max_closure = max(
                    max_closure, float(np.max(abs(loss - after_loss - signal - nonlinear + cost)))
                )
                s = new
            key = f"{name}_{width}"
            raw[key + "_trace"] = np.array(trace)
            raw[key + "_times"] = np.array(times)
            raw[key + "_dense_loss"] = 0.5 * (vp * (vy - toy_forward(s, vx)[0]) ** 2).sum(1)
            for label, value in zip(["w", "a", "c"], s):
                raw[key + "_" + label] = value
            raw[key + "_max_closure"] = np.array(max_closure)
            np.savez_compressed(out / "trajectories.npz", **raw)
            print(
                json.dumps(
                    dict(
                        case=name,
                        width=width,
                        mean_loss=float(loss.mean()),
                        max_closure=max_closure,
                    )
                ),
                flush=True,
            )
    (out / "complete.json").write_text("{}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["mnist", "toy"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=283001)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--horizon", type=float, default=2000)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.no_grad() if args.study == "toy" else torch.enable_grad():
        (mnist if args.study == "mnist" else toy)(args)


if __name__ == "__main__":
    main()
