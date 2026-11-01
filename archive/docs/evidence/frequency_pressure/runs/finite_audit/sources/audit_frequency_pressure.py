"""Finite gate-aware attribution of ordinary one-step SGD continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_competition import frequency_pressure as fp
from neural_competition import parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, nargs="+", default=[6000, 10000])
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--seed-bank", type=int, default=1157100)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior audits")
    training = json.loads((run / "protocol.json").read_text())
    protocol = dict(**vars(args), training=training, sources={},
                    attribution="straight parameter path; signed margin shares at gate jumps",
                    interpretation="observational allocation, not unique causal mediation")
    (out / "sources").mkdir()
    for source in [Path(__file__), Path(fp.__file__)]:
        content = source.read_bytes()
        protocol["sources"][source.name] = hashlib.sha256(content).hexdigest()
        (out / "sources" / source.name).write_bytes(content)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    q = np.array(training["q"])
    pop, _ = parity.panels(q)
    x, y, p = fp.inputs(pop["x"], q, training["encoding"]), pop["y"], pop["p"]
    phi, incidence = fp.product_basis(pop["x"], q)
    allocation = fp.factor_allocation(incidence)
    u = np.sqrt(p)[:, None] * phi
    checkpoints = np.load(run / "checkpoints.npz")
    for step in args.steps:
        for seed in range(args.seeds):
            for width in args.widths:
                state = tuple(checkpoints[f"{step}_{width}_{name}"][seed] for name in ["w", "a", "c"])
                e = y - fp.forward(state, x)
                _, pressures = fp.residual_pressures(e, phi, p, allocation)
                components = fp.velocity(state, x, pressures, p)
                current = np.sqrt(p) * e / np.sqrt(np.sum(p * e**2))
                probes = np.column_stack([u[:, [1 << i for i in range(len(q))]], current])
                rng = np.random.default_rng(args.seed_bank + step * 100 + seed)
                totals, smooth, gates, losses, opposed = [], [], [], [], []
                max_error = 0.
                for _ in range(args.samples):
                    ids = rng.choice(len(p), size=training["batch"], p=p)
                    update = fp.velocity(state, x[ids], y[ids] - fp.forward(state, x[ids]),
                                         1 / training["batch"])
                    end = fp.advance(state, update, training["rate"])
                    displacement = tuple(training["rate"] * np.concatenate(
                        [part, (full - part.sum(0))[None]], axis=0)
                        for part, full in zip(components, update))
                    finite = fp.finite_pressure_transport(state, end, x, p, probes, displacement)
                    exact = fp.kernel_change_ledger(state, end, x, p, probes)
                    max_error = max(max_error, float(np.max(np.abs(
                        finite["total"].sum(0) - exact["total"]))))
                    totals.append(finite["total"])
                    smooth.append(finite["smooth"])
                    gates.append(finite["gates"])
                    losses.append(fp.step_ledger(state, end, x, y, p, update, training["rate"]))
                    z0, z1 = state[0] @ x.T, end[0] @ x.T
                    margin = displacement[0][1:len(q) + 1] @ x.T
                    opposed_factors = (margin.min(0) < 0) & (margin.max(0) > 0)
                    changed = (z0 > 0) != (z1 > 0)
                    mass = np.sum(changed * p)
                    opposed.append(float(np.sum(changed * opposed_factors * p) / max(mass, 1e-300)))
                record = dict(step=step, seed=seed, width=width, max_identity_error=max_error,
                              transport=np.array(totals).tolist(), smooth=np.array(smooth).tolist(),
                              gates=np.array(gates).tolist(), opposed_crossing_fraction=opposed,
                              loss_terms={k: [float(v[k]) for v in losses] for k in losses[0]})
                with (out / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
        print(json.dumps(dict(step=step, complete=True)), flush=True)


if __name__ == "__main__":
    main()
