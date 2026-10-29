"""Natural-state forecasts and interventions that preserve implementation moments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from implementation_moments import (
    functional_particles,
    hierarchy,
    mix_within_gate,
    moment_kernel,
    moment_output,
    moment_step,
    moments,
    particle_output,
    particle_step,
    response_from_kernel,
)
from run_refinement_hierarchy import key
from toy_option_resources import advance, forward, kernel, velocity

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end", type=float, default=2000)
    parser.add_argument("--probe-count", type=int, default=12)
    parser.add_argument("--mix-time", type=int, default=20)
    parser.add_argument("--skip-forecasts", action="store_true")
    args = parser.parse_args()
    src = ROOT / args.input
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    configs = [
        dict(n=1, ratio=1.0, overlap=0.0),
        dict(n=4, ratio=0.1, overlap=0.0),
        dict(n=4, ratio=0.1, overlap=1.0),
    ]
    protocol = dict(
        input=args.input,
        end=args.end,
        configs=configs,
        probe_count=args.probe_count,
        dt=0.05,
        probe_times=[] if args.skip_forecasts else [2, 20, 100],
        mix_time=args.mix_time,
        probe_horizons=[0.05, 0.5, 5.0],
        intervention="One random within-gate orthogonal pair mixing at declared mix_time; same full joint moments, predictions, full K, width and optimizer. Largest feasible predetermined angle; no outcome selection.",
        sources={},
        input_hashes={},
    )
    for name in [
        Path(__file__).name,
        "implementation_moments.py",
        "toy_option_resources.py",
        "run_refinement_hierarchy.py",
    ]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    for config in configs:
        for m in [2 + 2 * config["n"], 4 + 4 * config["n"]]:
            file = src / f"{key(config)}_m{m}.npz"
            protocol["input_hashes"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for config in configs:
        data = hierarchy(**config)
        name = key(config)
        for m in [data["capacity"], 2 * data["capacity"]]:
            raw = np.load(src / f"{name}_m{m}.npz")
            for t0 in protocol["probe_times"]:
                states = tuple(
                    raw[f"t{t0}_{label}"][: args.probe_count].copy() for label in ["w", "a", "c"]
                )
                resources = [
                    moments(tuple(v[j : j + 1] for v in states), data)
                    for j in range(len(states[0]))
                ]
                particles = functional_particles(states, data)
                k0 = kernel(states, data)
                frozen = forward(states, data["x"]) * data["root"]
                gates = states[0] @ data["x"].T > 0
                crossed = np.zeros(len(states[0]), dtype=bool)
                for step in range(1, 101):
                    resources = [moment_step(r, data, 0.05) for r in resources]
                    particles = particle_step(particles, data, 0.05)
                    frozen += 0.05 * np.einsum("sij,sj->si", k0, data["root"] * data["y"] - frozen)
                    states = advance(states, velocity(states, data), 0.05)
                    crossed |= np.any((states[0] @ data["x"].T > 0) != gates, axis=(1, 2))
                    if step in [1, 10, 100]:
                        kk = kernel(states, data)
                        ff = forward(states, data["x"]) * data["root"]
                        fp = particle_output(particles, data)
                        with (out / "forecasts.jsonl").open("a") as f:
                            for j, r in enumerate(resources):
                                b = response_from_kernel(kk[j])
                                br = response_from_kernel(moment_kernel(r, data))
                                b0 = response_from_kernel(k0[j])
                                fr = moment_output(r, data)
                                row = dict(
                                    case=name,
                                    width=m,
                                    seed=j,
                                    start=t0,
                                    horizon=step * 0.05,
                                    crossed=bool(crossed[j]),
                                    groups=len(r["groups"]),
                                    moment_function_error=float(np.linalg.norm(ff[j] - fr)),
                                    frozen_function_error=float(np.linalg.norm(ff[j] - frozen[j])),
                                    particle_function_error=float(np.linalg.norm(ff[j] - fp[j])),
                                    moment_response_error=float(np.linalg.norm(b - br)),
                                    frozen_response_error=float(np.linalg.norm(b - b0)),
                                )
                                f.write(json.dumps(row, allow_nan=False) + "\n")
            base = tuple(raw[f"t{args.mix_time}_{label}"].copy() for label in ["w", "a", "c"])
            starts = []
            meta = []
            for j in range(len(base[0])):
                original = tuple(v[j : j + 1] for v in base)
                mixed = mix_within_gate(original, data, np.random.default_rng(296301 + j))
                if mixed is None:
                    with (out / "ineligible.jsonl").open("a") as f:
                        f.write(json.dumps(dict(case=name, width=m, seed=j)) + "\n")
                    continue
                state, info = mixed
                orig = moments(original, data)
                other = moments(state, data)
                error = max(
                    np.max(abs(a[k] - b[k]))
                    for a, b in zip(orig["groups"], other["groups"])
                    for k in ["s", "q", "R"]
                )
                assert error < 1e-10
                starts.extend([original, state])
                meta.extend(
                    [
                        dict(
                            case=name,
                            width=m,
                            seed=j,
                            arm=arm,
                            moment_matching_error=float(error),
                            **info,
                        )
                        for arm in ["control", "mixed"]
                    ]
                )
            if not starts:
                continue
            state = tuple(np.concatenate([s[k] for s in starts]) for k in range(3))
            initial_gates = state[0] @ data["x"].T > 0
            first = np.full(len(starts), -1.0)
            divergence = np.full(len(starts) // 2, -1.0)
            for i in range(round((args.end - args.mix_time) / 0.05) + 1):
                if i in [0, 1, 10, 100, 1600, round((args.end - args.mix_time) / 0.05)]:
                    pred = forward(state, data["x"])
                    loss = 0.5 * np.sum(data["p"] * (pred - data["y"]) ** 2, axis=1)
                    with (out / "mixed_outcomes.jsonl").open("a") as f:
                        for j, row in enumerate(meta):
                            f.write(
                                json.dumps(
                                    dict(
                                        **row,
                                        time=args.mix_time + i * 0.05,
                                        loss=float(loss[j]),
                                        first_gate_change=float(first[j]),
                                        first_gate_divergence=float(divergence[j // 2]),
                                    ),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                if i == round((args.end - args.mix_time) / 0.05):
                    break
                state = advance(state, velocity(state, data), 0.05)
                changed = np.any((state[0] @ data["x"].T > 0) != initial_gates, axis=(1, 2))
                first[(first < 0) & changed] = (i + 1) * 0.05
                current_gates = state[0] @ data["x"].T > 0
                different = np.any(current_gates[::2] != current_gates[1::2], axis=(1, 2))
                divergence[(divergence < 0) & different] = (i + 1) * 0.05
            # Independently replayed control must reproduce the original saved endpoint.
            expected = tuple(raw[f"t{int(args.end)}_{label}"] for label in ["w", "a", "c"])
            expected_loss = 0.5 * np.sum(
                data["p"] * (forward(expected, data["x"]) - data["y"]) ** 2, axis=1
            )
            assert (
                max(
                    abs(loss[j] - expected_loss[row["seed"]])
                    for j, row in enumerate(meta)
                    if row["arm"] == "control"
                )
                < 1e-9
            )
            print(name, m, "eligible", len(starts) // 2, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
