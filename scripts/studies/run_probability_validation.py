"""Fresh conditional forecasts locked before the relevant training continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from implementation_moments import moments, observe
from probabilistic_options import clock, features, probability, weighted_task
from toy_option_resources import advance, velocity

ROOT = Path(__file__).resolve().parents[2]


def configurations():
    cases = [
        (1, "geometric", 1.0, 0.0, "same"),
        (1, "geometric", 1.0, 1.0, "same"),
        (2, "geometric", 0.25, 0.0, "same"),
        (4, "geometric", 0.25, 0.0, "same"),
        (4, "geometric", 0.1, 0.0, "same"),
        (4, "geometric", 0.1, 1.0, "same"),
        (3, "zipf", 1.0, 0.0, "transfer"),
        (4, "zipf", 1.0, 0.0, "transfer"),
        (4, "zipf", 2.0, 0.5, "transfer"),
    ]
    return [
        dict(n=n, family=f, parameter=p, overlap=o, split=s, case=f"n{n}_{f}{p:g}_o{o:g}")
        for n, f, p, o, s in cases
    ]


def rows(state, data, config, m, t, bank):
    obs = observe(state, data)
    for seed in range(len(state[0])):
        resource = moments(tuple(v[seed : seed + 1] for v in state), data)
        yield dict(
            **config,
            ratio=data["ratio"],
            q=data["q"].tolist(),
            width=m,
            seed=seed,
            bank=bank,
            time=t,
            gate_groups=len(resource["groups"]),
            **{k: np.asarray(v[seed]).tolist() for k, v in obs.items()},
        )


def append(path, records):
    with path.open("a") as file:
        for row in records:
            file.write(json.dumps(row, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=297101)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--end", type=float, default=2000)
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    model_path = ROOT / "docs/theory/probabilistic_options/locked/models.json"
    models = json.loads(model_path.read_text())["selected"]
    configs = configurations()
    protocol = dict(
        count=args.count,
        seed=args.seed,
        dt=args.dt,
        end=args.end,
        factors=args.factors,
        configs=configs,
        scale=0.002,
        model_path=str(model_path.relative_to(ROOT)),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        design="All initial closure forecasts precede training of their own run. All conditional opening forecasts are saved at t=20, before any run continues beyond t=20. No model refitting. New seeds, dt=.025; Zipf and N=3/intermediate alignment are declared transfer tests.",
        thresholds=dict(coverage=0.95, mastery=0.1),
        sources={},
    )
    for name in [
        Path(__file__).name,
        "probabilistic_options.py",
        "implementation_moments.py",
        "toy_option_resources.py",
    ]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "models.json").write_bytes(model_path.read_bytes())
    for config in configs:
        data = weighted_task(config["n"], config["family"], config["parameter"], config["overlap"])
        rng = np.random.default_rng(args.seed)
        maxm = 4 * data["capacity"]
        iw = 0.002 * rng.normal(size=(args.count, maxm, data["x"].shape[1]))
        ia = 0.002 * rng.normal(size=(args.count, maxm))
        for factor in args.factors:
            m = factor * data["capacity"]
            state = (iw[:, :m].copy(), ia[:, :m].copy(), np.zeros(args.count))
            initial = tuple(v.copy() for v in state)
            for row in rows(state, data, config, m, 0, args.seed):
                append(out / "trajectory.jsonl", [row])
                for j in range(data["n"]):
                    if row["coverage"][j] < 0.95:
                        continue
                    feature = features(row, j, data["q"][j])
                    pred = {
                        k: float(probability(model, [feature])[0])
                        for k, model in models.items()
                        if k.startswith("closure")
                    }
                    append(
                        out / "forecasts.jsonl",
                        [
                            dict(
                                case=config["case"],
                                split=config["split"],
                                width=m,
                                seed=row["seed"],
                                component=j + 1,
                                weight=1 / data["n"],
                                event="closure",
                                horizon=20,
                                features=feature,
                                predictions=pred,
                            )
                        ],
                    )
            times = {round(t / args.dt): t for t in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]}
            for i in range(1, round(20 / args.dt) + 1):
                state = advance(state, velocity(state, data), args.dt)
                if i in times:
                    append(
                        out / "trajectory.jsonl", rows(state, data, config, m, times[i], args.seed)
                    )
            for row in rows(state, data, config, m, 20, args.seed):
                for j in range(data["n"]):
                    if row["coverage"][j] >= 0.95 or abs(row["gap_error"][j]) <= 0.1:
                        continue
                    feature = features(row, j, data["q"][j])
                    for horizon in [50.0, 100.0, 500.0, 2000.0]:
                        if horizon > args.end:
                            continue
                        pred = {
                            k: float(
                                probability(model, [feature], [clock(horizon, model["clock"])])[0]
                            )
                            for k, model in models.items()
                            if k.startswith("opening")
                        }
                        append(
                            out / "forecasts.jsonl",
                            [
                                dict(
                                    case=config["case"],
                                    split=config["split"],
                                    width=m,
                                    seed=row["seed"],
                                    component=j + 1,
                                    weight=1 / data["n"],
                                    event="opening",
                                    horizon=horizon,
                                    features=feature,
                                    predictions=pred,
                                )
                            ],
                        )
            np.savez_compressed(
                out / f"{config['case']}_m{m}_landmark.npz",
                **{
                    f"{time}_{key}": v
                    for time, ss in [(0, initial), (20, state)]
                    for key, v in zip(["w", "a", "c"], ss)
                },
            )
    (out / "forecasts_locked.json").write_text(
        json.dumps(
            dict(
                forecast_sha256=hashlib.sha256((out / "forecasts.jsonl").read_bytes()).hexdigest(),
                status="All forecasts saved; no continuation beyond t=20 has run.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("All forecasts locked; beginning continuations", flush=True)
    for config in configs:
        data = weighted_task(config["n"], config["family"], config["parameter"], config["overlap"])
        for factor in args.factors:
            m = factor * data["capacity"]
            raw = np.load(out / f"{config['case']}_m{m}_landmark.npz")
            state = tuple(raw[f"20_{key}"].copy() for key in ["w", "a", "c"])
            times = {
                round((t - 20) / args.dt): t
                for t in [30.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, args.end]
                if 20 < t <= args.end
            }
            for i in range(1, round((args.end - 20) / args.dt) + 1):
                state = advance(state, velocity(state, data), args.dt)
                if i in times:
                    append(
                        out / "trajectory.jsonl", rows(state, data, config, m, times[i], args.seed)
                    )
            np.savez_compressed(
                out / f"{config['case']}_m{m}_final.npz",
                **{key: v for key, v in zip(["w", "a", "c"], state)},
            )
            print(config["case"], m, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
