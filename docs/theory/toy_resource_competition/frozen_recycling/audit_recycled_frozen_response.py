"""Replay recycling starts and audit exact frozen responses at the remaining budget."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from recycle_toy_resources import recycle
from toy_option_resources import forward, kernel, task

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/theory/toy_resource_competition"


def frozen(state, data, horizon, dt):
    eig, vec = np.linalg.eigh(kernel(state, data)[0])
    eig = np.maximum(eig, 0)
    e = (data["y"] - forward(state, data["x"])[0]) * data["root"]
    load = (vec.T @ e) ** 2
    flow = 0.5 * np.sum(load * (-np.expm1(-2 * horizon * eig)))
    assert np.max(dt * eig) < 2, "Frozen Euler response is unstable"
    steps = round(horizon / dt)
    gd = 0.5 * np.sum(load * (1 - (1 - dt * eig) ** (2 * steps)))
    return float(flow), float(gd)


def main():
    protocol = json.loads((BASE / "recycle_confirmation/protocol.json").read_text())
    outcomes = [
        json.loads(line)
        for line in (BASE / "recycle_confirmation/outcomes.jsonl").read_text().splitlines()
    ]
    rows = {
        (r["case"], r["width"], r["seed"], r["repeat"], r["arm"], r["time"]): r for r in outcomes
    }
    raw_records, summaries = [], []
    for case in protocol["cases"]:
        data, dense = task(case), task(case, True)
        for m in [data["capacity"], 2 * data["capacity"]]:
            bank = np.load(ROOT / protocol["input"] / f"{case}_{m}_states.npz")
            state = tuple(bank[f"ordinary_20_{key}"] for key in ["w", "a", "c"])
            seed_means = collections.defaultdict(list)
            for seed in range(len(state[0])):
                for repeat in range(protocol["repeats"]):
                    result = recycle(
                        tuple(v[seed : seed + 1] for v in state),
                        data,
                        dense,
                        np.random.default_rng(295900 + 1000 * seed + repeat),
                    )
                    if result is None:
                        continue
                    control, treatment, _ = result
                    c, t = [frozen(s, data, 1980, 0.05) for s in [control, treatment]]
                    initial = rows[case, m, seed, repeat, "control", 20]["loss"]
                    recomputed = 0.5 * np.sum(
                        data["p"] * (data["y"] - forward(control, data["x"])[0]) ** 2
                    )
                    assert abs(initial - recomputed) < 1e-10
                    gain = (
                        rows[case, m, seed, repeat, "control", 2000]["loss"]
                        - rows[case, m, seed, repeat, "recycled", 2000]["loss"]
                    )
                    predicted = np.array(t) - np.array(c)
                    raw_records.append(
                        dict(
                            case=case,
                            width=m,
                            seed=seed,
                            repeat=repeat,
                            actual_gain=gain,
                            flow_gain=predicted[0],
                            gd_gain=predicted[1],
                        )
                    )
                    seed_means[seed].append([gain, *predicted])
            if not seed_means:
                continue
            values = np.array([np.mean(v, axis=0) for v in seed_means.values()])
            summaries.append(
                dict(
                    case=case,
                    width=m,
                    eligible=len(values),
                    mean_actual=float(values[:, 0].mean()),
                    mean_frozen_flow=float(values[:, 1].mean()),
                    mean_frozen_gd=float(values[:, 2].mean()),
                    correlation_flow=None
                    if min(np.std(values[:, 0]), np.std(values[:, 1])) < 1e-12
                    else float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
                    rmse_frozen_flow=float(np.sqrt(np.mean((values[:, 0] - values[:, 1]) ** 2))),
                )
            )
    out = BASE / "frozen_recycling"
    out.mkdir(exist_ok=True)
    assert not (out / "protocol.json").exists()
    record = dict(
        input_protocol_hash=hashlib.sha256(
            (BASE / "recycle_confirmation/protocol.json").read_bytes()
        ).hexdigest(),
        sources={},
        horizon=1980,
        dt=0.05,
        status="Post-outcome model audit, no fitted parameters; not a prospective forecast.",
    )
    for name in [Path(__file__).name, "toy_option_resources.py", "recycle_toy_resources.py"]:
        source = Path(__file__).with_name(name)
        record["sources"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (out / name).write_bytes(source.read_bytes())
    (out / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "audit.json").write_text(
        json.dumps(dict(summary=summaries, trials=raw_records), indent=2) + "\n"
    )
    (out / "complete.json").write_text("{}\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
