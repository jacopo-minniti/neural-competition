"""Test the old completion-reserve prediction while relaxing one premise at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from competition_reserve import (
    capability,
    common_flow,
    protected_reserve,
    reserve,
)
from reserve_bridge import factor_response_power, forward, task, velocity

SETTINGS = [
    dict(name="anchor", mass=.04, cooccurrence=0., spread=0., gamma=1.5),
    dict(name="cooccurrence", mass=.04, cooccurrence=1., spread=0., gamma=1.5),
    dict(name="more_other_pressure", mass=.15, cooccurrence=1., spread=0., gamma=1.5),
    dict(name="boundary_30", mass=.30, cooccurrence=1., spread=0., gamma=1.5),
    dict(name="boundary_60", mass=.60, cooccurrence=1., spread=0., gamma=1.5),
    dict(name="bounded_observations", mass=.04, cooccurrence=1., spread=.1, gamma=1.5),
    dict(name="remove_opposition", mass=.04, cooccurrence=1., spread=0., gamma=1.),
    dict(name="associated_factors", mass=.04, cooccurrence=1., spread=0., gamma=1.5, association=1.),
    dict(name="associated_more_pressure", mass=.15, cooccurrence=1., spread=0., gamma=1.5,
         association=1.),
]


def metric_norm(update, width):
    w, a, c = update
    return np.sqrt((np.sum(w**2, axis=(-2, -1)) + np.sum(a**2, axis=-1)) / width + c**2)


def describe(state, train, audit, functional, q):
    f = forward(state, train["x"])
    coefficients = (train["p"] * f) @ (functional["directions"] / np.sqrt(
        np.where(train["p"] > 0, train["p"], 1))[:, None])
    target = (-1.)**np.arange(len(q)) * np.sqrt(q)
    power = factor_response_power(state, train, functional)
    zero = power == 0
    return dict(prototype_response=capability(state, audit).tolist(),
                factor_response=capability(state, functional).tolist(),
                factor_demand=(target - coefficients).tolist(), exact_zero=zero.tolist(),
                loss=float(.5 * np.sum(train["p"] * (train["y"] - f)**2)),
                closed_factor_loss_floor=float(.5 * q[zero].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--seed-bank", type=int, default=170731)
    parser.add_argument("--rate", type=float, default=.01)
    parser.add_argument("--followup-steps", type=int, default=5000)
    parser.add_argument("--settings", nargs="+", default=[s["name"] for s in SETTINGS])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior cohorts")
    widths = [4, 8, 16]
    settings = [s for s in SETTINGS if s["name"] in args.settings]
    protocol = dict(**vars(args), settings_spec=settings, widths=widths, factors=4,
                    target="unchanged global 4 ReLU(x0) + linear attribute",
                    primary="half-response loss over two successive half-completion phases",
                    primary_audit="unchanged uniform single-factor prototype panel",
                    secondary="population factor corrections and exact missing-factor loss floor",
                    optimizer="full objective GD, M_w=M_a=m; no frozen parameters or gates",
                    prediction="unchanged analytic common-completion reserve; no fitted parameters",
                    phase_endpoint="first ordinary step crossing half remaining head correction",
                    sources={})
    (out / "sources").mkdir()
    for source in [Path(__file__), Path(__file__).with_name("reserve_bridge.py"),
                   Path(__file__).with_name("competition_reserve.py"),
                   Path(__file__).with_name("probabilistic_option_transport.py")]:
        data = source.read_bytes()
        protocol["sources"][source.name] = hashlib.sha256(data).hexdigest()
        (out / "sources" / source.name).write_bytes(data)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, followup = [], []
    for setting in settings:
        kwargs = {k: v for k, v in setting.items() if k != "name"}
        tasks = [task(seed=args.seed_bank + seed, **kwargs) for seed in range(args.seeds)]
        x = np.array([t[0]["x"] for t in tasks])
        y = np.array([t[0]["y"] for t in tasks])
        p, q = tasks[0][0]["p"], tasks[0][3]["q"]
        wbank, abank = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(args.seed_bank + 10000 + seed)
            wbank.append(.2 * rng.normal(size=(16, x.shape[-1])))
            abank.append(.2 * rng.normal(size=16))
        for width in widths:
            states = (np.array(wbank)[:, :width].copy(), np.array(abank)[:, :width].copy(),
                      np.zeros(args.seeds))
            elapsed = np.zeros(args.seeds, dtype=int)
            for phase in range(2):
                phase_rows = []
                initial_head = states[2] + np.mean(states[1] * np.maximum(states[0][:, :, 0], 0), axis=1)
                target = initial_head + .5 * (4 - initial_head)
                for seed in range(args.seeds):
                    st = tuple(v[seed] for v in states)
                    train, audit, functional, meta = tasks[seed]
                    initial = describe(st, train, audit, functional, q)
                    predicted, exposure = reserve(st, audit, .5 * (4 - initial_head[seed]))
                    predicted_factor = capability(common_flow(st, exposure), functional)
                    safe, _ = protected_reserve(st, train, audit)
                    predicted_state = common_flow(st, exposure)
                    full_update = velocity(st, train["x"], train["y"], train["p"])
                    head_update = velocity(st, train["x"], train["y"], train["p"] * train["head"])
                    other_update = tuple(v - h for v, h in zip(full_update, head_update))
                    record = dict(setting=setting["name"], seed=seed, width=width, phase=phase,
                                  start_step=int(elapsed[seed]), q=q.tolist(),
                                  p_empty=meta["p_empty"], cooccurrence_mass=meta["cooccurrence_mass"],
                                  initial=initial, predicted=predicted.tolist(),
                                  predicted_factor=predicted_factor.tolist(), safe=safe.tolist(),
                                  initial_other_to_head=float(metric_norm(other_update, width)
                                      / max(metric_norm(head_update, width), 1e-30)),
                                  predicted_zero=(factor_response_power(predicted_state, train,
                                                                       functional) == 0).tolist())
                    phase_rows.append(record)
                with (out / "predictions.jsonl").open("a") as handle:
                    for row in phase_rows:
                        handle.write(json.dumps(row) + "\n")
                done = np.zeros(args.seeds, dtype=bool)
                ends = tuple(v.copy() for v in states)
                length_head, length_other = np.zeros(args.seeds), np.zeros(args.seeds)
                for step in range(1, 20001):
                    full = velocity(states, x, y, p)
                    head_weights = p * tasks[0][0]["head"]
                    head = velocity(states, x, y, head_weights)
                    length_head += (~done) * args.rate * metric_norm(head, width)
                    length_other += (~done) * args.rate * metric_norm(
                        tuple(f - h for f, h in zip(full, head)), width)
                    states = tuple(v + args.rate * dv for v, dv in zip(states, full))
                    value = states[2] + np.mean(states[1] * np.maximum(states[0][:, :, 0], 0), axis=1)
                    hit = (~done) & (value >= target)
                    for seed in np.flatnonzero(hit):
                        for dest, value_state in zip(ends, states):
                            dest[seed] = value_state[seed]
                        elapsed[seed] += step
                    done |= hit
                    if done.all():
                        break
                if not done.all():
                    raise RuntimeError("A completion phase did not finish; preserve the cohort")
                states = ends
                for seed, row in enumerate(phase_rows):
                    train, audit, functional, _ = tasks[seed]
                    row["actual"] = describe(tuple(v[seed] for v in states), train, audit, functional, q)
                    row["end_step"] = int(elapsed[seed])
                    row["other_to_head_update_length"] = float(length_other[seed] / length_head[seed])
                    rows.append(row)
            # Continue unchanged full training to a common, ordinary step budget.
            if args.followup_steps:
                if np.any(elapsed >= args.followup_steps):
                    raise RuntimeError("Follow-up budget must exceed both phase endpoints")
                ends = tuple(v.copy() for v in states)
                for _ in range(args.followup_steps - int(elapsed.min())):
                    update = velocity(states, x, y, p)
                    states = tuple(v + args.rate * dv for v, dv in zip(states, update))
                    elapsed += 1
                    hit = elapsed == args.followup_steps
                    for seed in np.flatnonzero(hit):
                        for dest, value_state in zip(ends, states):
                            dest[seed] = value_state[seed]
                for seed in range(args.seeds):
                    train, audit, functional, _ = tasks[seed]
                    followup.append(dict(setting=setting["name"], width=width, seed=seed,
                        step=args.followup_steps, **describe(tuple(v[seed] for v in ends),
                                                            train, audit, functional, q)))
        (out / "results.json").write_text(json.dumps(dict(protocol=protocol, rows=rows,
                                                         followup=followup), separators=(",", ":")) + "\n")
        print(json.dumps(dict(setting=setting["name"], complete=True)), flush=True)


if __name__ == "__main__":
    main()
