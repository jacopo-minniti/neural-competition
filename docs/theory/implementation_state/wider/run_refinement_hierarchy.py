"""Natural widths on systematically varied conditional-refinement hierarchies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from implementation_moments import hierarchy, moments, observe
from toy_option_resources import advance, velocity

ROOT = Path(__file__).resolve().parents[2]


def configurations():
    return [
        dict(n=n, ratio=r, overlap=o)
        for n, ratios in [(1, [1.0]), (2, [1.0, 0.25]), (4, [1.0, 0.25, 0.1])]
        for r in ratios
        for o in [1.0, 0.0]
    ]


def key(config):
    return f"n{config['n']}_r{config['ratio']:g}_o{config['overlap']:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=296101)
    parser.add_argument("--end", type=float, default=4000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    configs = [c for c in configurations() if args.only is None or key(c) in args.only]
    protocol = dict(
        configs=configs,
        count=args.count,
        seed=args.seed,
        end=args.end,
        dt=args.dt,
        factors=args.factors,
        scale=0.002,
        task="All contexts always present; total refinement frequency .2; q_j proportional to ratio**j; alternating coefficients +/-2; overlap is attribute overlap, not an assumed measured implementation overlap.",
        widths="(2+2N)*factor, exact target representation at narrow width",
        analysis="Per-refinement structural coverage >= .95, residual gap <= .1 of target gap; report continuous values and sensitivities. Checkpoint transitions are sampled, not exact event times.",
        sources={},
    )
    for name in [Path(__file__).name, "implementation_moments.py", "toy_option_resources.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    times = sorted(
        set(
            [
                0.0,
                1.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                15.0,
                20.0,
                30.0,
                50.0,
                100.0,
                200.0,
                500.0,
                1000.0,
                2000.0,
                4000.0,
                args.end,
            ]
        )
    )
    checkpoints = {round(t / args.dt): t for t in times if t <= args.end}
    for config in configs:
        name = key(config)
        data = hierarchy(**config)
        mmax = data["capacity"] * max(args.factors)
        rng = np.random.default_rng(args.seed)
        # Always draw up to 4x capacity so nested banks are unchanged in factor ablations.
        iw = 0.002 * rng.normal(size=(args.count, data["capacity"] * 4, data["x"].shape[1]))
        ia = 0.002 * rng.normal(size=(args.count, data["capacity"] * 4))
        assert mmax <= iw.shape[1]
        for factor in args.factors:
            m = data["capacity"] * factor
            state = (iw[:, :m].copy(), ia[:, :m].copy(), np.zeros(args.count))
            raw = {}
            for i in range(round(args.end / args.dt) + 1):
                if i in checkpoints:
                    t = checkpoints[i]
                    obs = observe(state, data)
                    with (out / "trajectory.jsonl").open("a") as f:
                        for seed in range(args.count):
                            resource = moments(tuple(v[seed : seed + 1] for v in state), data)
                            row = dict(
                                case=name,
                                **config,
                                width=m,
                                factor=factor,
                                seed=seed,
                                time=t,
                                gate_groups=len(resource["groups"]),
                                largest_group=max(g["count"] for g in resource["groups"]),
                                **{k: np.asarray(v[seed]).tolist() for k, v in obs.items()},
                            )
                            f.write(json.dumps(row, allow_nan=False) + "\n")
                    if t in [0, 2, 10, 20, 100, 500, 2000, args.end]:
                        for label, v in zip(["w", "a", "c"], state):
                            raw[f"t{int(t)}_{label}"] = v.copy()
                if i < round(args.end / args.dt):
                    state = advance(state, velocity(state, data), args.dt)
            np.savez_compressed(out / f"{name}_m{m}.npz", **raw)
            print(name, m, float(np.mean(obs["loss"])), flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
