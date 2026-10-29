"""Response-matched implementation interventions under the original squared loss.

No target edits, chosen eigenvectors, or auxiliary protection loss. A fixed metric
matches the complete initial kernel, not merely its trace or residual Rayleigh
quotient. Baseline and concentrated copies differ in subsequent transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from explore_recruitable_workspace import initial, regions, task

ROOT = Path(__file__).resolve().parents[2]


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def model(state, base, copies=1, concentrated=False):
    w, a, c = state
    fractions = np.zeros(copies) if concentrated else np.ones(copies) / copies
    if concentrated:
        fractions[0] = 1
    return dict(
        w=np.tile(w, (copies, 1)),
        a=np.concatenate([v * a for v in fractions]),
        c=c.copy(),
        base=base,
        ma=base / copies,
        mw=base / np.sum(fractions**2),
    )


def clone(s):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s.items()}


def predict(s, x):
    return np.maximum(x @ s["w"].T, 0) @ s["a"] / s["base"] + x @ s["c"]


def velocity(s, x, y):
    z = x @ s["w"].T
    h = z.clip(0)
    e = h @ s["a"] / s["base"] + x @ s["c"] - y
    return dict(
        a=-s["ma"] * (e @ h) / (len(x) * s["base"]),
        w=-s["mw"] * ((e[:, None] * (z > 0)).T @ x) * s["a"][:, None] / (len(x) * s["base"]),
        c=-(e @ x) / len(x),
    )


def step(s, x, y, rate):
    v = velocity(s, x, y)
    for name in v:
        s[name] += rate * v[name]


def kernel(s, x, gate=None):
    z = x @ s["w"].T
    gate = z > 0 if gate is None else gate
    h = z * gate
    ga = gate * s["a"]
    return (
        s["ma"] * h @ h.T / s["base"] ** 2
        + s["mw"] * (ga @ ga.T) * (x @ x.T) / s["base"] ** 2
        + x @ x.T
    ) / len(x)


def kernel_derivative(s, x, v):
    z = x @ s["w"].T
    g = z > 0
    h, dh = z.clip(0), (x @ v["w"].T) * g
    ga, dga = g * s["a"], g * v["a"]
    return (s["ma"] * (dh @ h.T + h @ dh.T) + s["mw"] * (dga @ ga.T + ga @ dga.T) * (x @ x.T)) / (
        s["base"] ** 2 * len(x)
    )


def evaluate(s, x, y, vx, vy, px, demand):
    tr, va = predict(s, x) - y, predict(s, vx) - vy
    k = kernel(s, px)
    vals, vecs = np.linalg.eigh(k)
    vals = np.maximum(vals, 0)
    # Same GD-derived discounted option operator as the existing note, H=I.
    strengths = vals / (vals + 0.5)  # tau=1
    loading = (vecs.T @ demand) ** 2
    return dict(
        train=float(np.mean(tr**2) / 2),
        validation=float(np.mean(va**2) / 2),
        validation_regions=[float(np.mean(va[r] ** 2) / 2) for r in regions(vx)],
        common_strength=float(demand @ k @ demand / (demand @ demand)),
        common_value=float(strengths @ loading / (2 * len(px))),
        option_mass=float(strengths.sum()),
    )


def audit():
    import torch

    x, y = task(44, 47)
    state = initial(45, 9, active=True)
    ordinary = model(state, 9)
    uniform, reserve = model(state, 9, 4), model(state, 9, 4, True)
    k0 = kernel(ordinary, x)
    d0 = kernel_derivative(ordinary, x, velocity(ordinary, x, y))
    result = {}
    for name, s, factor in [("uniform", uniform, 1), ("reserve", reserve, 0.25)]:
        result[name + "_function_error"] = float(np.max(abs(predict(s, x) - predict(ordinary, x))))
        result[name + "_kernel_error"] = float(np.max(abs(kernel(s, x) - k0)))
        result[name + "_transport_identity_error"] = float(
            np.max(abs(kernel_derivative(s, x, velocity(s, x, y)) - factor * d0))
        )
        # Independent autodiff checks for the complete metric kernel and velocity.
        w, a, c = [
            torch.tensor(s[k], dtype=torch.float64, requires_grad=True) for k in ["w", "a", "c"]
        ]
        xt = torch.tensor(x)

        def f(w, a, c):
            return torch.relu(xt @ w.T) @ a / 9 + xt @ c

        j = torch.autograd.functional.jacobian(f, (w, a, c))
        gram = sum(
            metric * q.reshape(len(x), -1) @ q.reshape(len(x), -1).T
            for metric, q in zip([s["mw"], s["ma"], 1], j)
        ) / len(x)
        result[name + "_autodiff_kernel_error"] = float(np.max(abs(gram.numpy() - kernel(s, x))))
        loss = ((f(w, a, c) - torch.tensor(y)) ** 2).mean() / 2
        loss.backward()
        v = velocity(s, x, y)
        result[name + "_autodiff_velocity_error"] = max(
            float(np.max(abs(v[key] + metric * p.grad.numpy())))
            for key, metric, p in zip(["w", "a", "c"], [s["mw"], s["ma"], 1], [w, a, c])
        )
    for _ in range(100):
        step(ordinary, x, y, 0.06)
        step(uniform, x, y, 0.06)
    result["uniform_100_step_error"] = float(
        np.max(abs(predict(ordinary, x) - predict(uniform, x)))
    )
    assert max(result.values()) < 1e-11, result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--affine", action="store_true")
    p.add_argument("--recovery", type=int, default=3000)
    args = p.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Refusing to replace existing records")
    protocol = dict(
        created=datetime.now(timezone.utc).isoformat(),
        seeds=args.seeds,
        widths=[9, 36],
        copies=4,
        affine=args.affine,
        active_initialization=True,
        train_n=256,
        validation_n=2048,
        panel_n=64,
        rate=0.06,
        start=200,
        exposure=400,
        recovery=args.recovery,
        histories=["full", "peer"],
        peer="seed modulo 3, equal input intervals",
        arms=["ordinary", "reserve", "double_rate (width 9 only)"],
        objective="Original labels throughout; peer history temporarily changes sampling only",
        predictions=[
            "Function and complete initial optimizer kernel match exactly for reserve vs ordinary.",
            "Smooth initial kernel velocity is reduced fourfold, with either sign preserved.",
            "Gate-mediated weakening is not generally reduced fourfold.",
            "Slack conjecture: reserve reduces peer-induced residual-option weakening and later excess loss, especially at width 9.",
            "Competing prediction: reserve also slows beneficial feature creation and may impede full-task improvement.",
            "Affine control should have small absolute reserve effects.",
        ],
        inference="No endpoint-fitted coefficients. Seed/task pair is the sampling unit; all horizons correlated.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256=hashlib.sha256(
            Path(__file__).with_name("explore_recruitable_workspace.py").read_bytes()
        ).hexdigest(),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "audit.json").write_text(json.dumps(audit(), indent=2) + "\n")
    for seed in args.seeds:
        x, y = task(seed, 256, affine=args.affine)
        vx, vy = task(seed, 2048, validation=True, affine=args.affine)
        px, py = vx[:64], vy[:64]
        base = {m: model(initial(seed, m, active=True), m) for m in [9, 36]}
        for s in base.values():
            for _ in range(200):
                step(s, x, y, 0.06)
        demand = py - sum(predict(s, px) for s in base.values()) / 2
        states = dict(x=x, y=y, vx=vx, vy=vy, demand=demand)
        for m, s0 in base.items():
            for key in ["w", "a", "c"]:
                states[f"base_{m}_{key}"] = s0[key]
            reserve = model(tuple(s0[key] for key in ["w", "a", "c"]), m, 4, True)
            for history in ["full", "peer"]:
                mask = np.ones(len(x), dtype=bool) if history == "full" else regions(x)[seed % 3]
                ex, ey = x[mask], y[mask]
                for arm in ["ordinary", "reserve"] + (["double_rate"] if m == 9 else []):
                    s = clone(reserve if arm == "reserve" else s0)
                    rate = 0.12 if arm == "double_rate" else 0.06
                    k0 = kernel(s, px)
                    v = velocity(s, ex, ey)
                    kd = kernel_derivative(s, px, v)
                    forecast = dict(
                        seed=seed,
                        width=m,
                        history=history,
                        arm=arm,
                        kernel_match_error=float(np.max(abs(k0 - kernel(s0, px)))),
                        function_match_error=float(np.max(abs(predict(s, vx) - predict(s0, vx)))),
                        smooth_demand_change=float(rate * demand @ kd @ demand / (demand @ demand)),
                    )
                    append(out / "forecasts.jsonl", forecast)
                    oldgate = px @ s["w"].T > 0
                    local = clone(s)
                    step(local, ex, ey, rate)
                    kf = kernel(local, px, oldgate)
                    ka = kernel(local, px)
                    append(
                        out / "local_outcomes.jsonl",
                        dict(
                            seed=seed,
                            width=m,
                            history=history,
                            arm=arm,
                            smooth_finite=float(demand @ (kf - k0) @ demand / (demand @ demand)),
                            gate=float(demand @ (ka - kf) @ demand / (demand @ demand)),
                            actual=float(demand @ (ka - k0) @ demand / (demand @ demand)),
                        ),
                    )
                    checkpoints = {0, 20, 100, 400, 420, 500, 800, 1600, 400 + args.recovery}
                    for t in range(400 + args.recovery + 1):
                        if t in checkpoints:
                            append(
                                out / "trajectory.jsonl",
                                dict(
                                    seed=seed,
                                    width=m,
                                    history=history,
                                    arm=arm,
                                    step=t,
                                    **evaluate(s, x, y, vx, vy, px, demand),
                                ),
                            )
                        if t == 400 + args.recovery:
                            break
                        step(s, ex if t < 400 else x, ey if t < 400 else y, rate)
                    for key in ["w", "a", "c"]:
                        states[f"end_{m}_{history}_{arm}_{key}"] = s[key]
        np.savez_compressed(out / f"states_{seed}.npz", **states)
        print(json.dumps(dict(seed=seed, complete=True)), flush=True)
    (out / "complete.json").write_text(json.dumps(dict(seeds=args.seeds)) + "\n")


if __name__ == "__main__":
    main()
