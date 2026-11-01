"""Ordinary fixed-distribution SGD on parity, with observational pressure audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from neural_competition import frequency_pressure as fp
from neural_competition import parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-bank", type=int, default=731100)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--maximum", type=float, default=.4)
    parser.add_argument("--profile", choices=["power", "geometric", "uniform_activity",
                                             "uniform_label", "balanced"], default="power")
    parser.add_argument("--encoding", choices=["raw", "centered"], default="raw")
    parser.add_argument("--initialization", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--scale", type=float, default=.3)
    parser.add_argument("--rate", type=float, default=.05)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--diagnostic-seeds", type=int, default=8)
    parser.add_argument("--diagnostic-steps", type=int, nargs="+",
                        default=[0, 100, 500, 1000, 2000, 4000, 6000, 10000, 20000])
    parser.add_argument("--forecast-steps", type=int, nargs="+", default=[500, 2000, 6000, 10000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--response-steps", type=int, default=1000)
    parser.add_argument("--cohort", choices=["development", "confirmation", "control"],
                        default="development")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior cohorts; choose a new output directory")
    q = parity.frequencies(args.features, args.alpha, args.maximum)
    if args.profile == "geometric":
        q = np.geomspace(q[0], q[-1], len(q))
    elif args.profile == "uniform_activity":
        q = np.full(len(q), q.mean())
    elif args.profile == "uniform_label":
        q = np.full(len(q), (1 - np.prod(1 - 2 * q)**(1 / len(q))) / 2)
    elif args.profile == "balanced":
        q = np.full(len(q), .5)
    pop, audit = parity.panels(q)
    x, y, p = fp.inputs(pop["x"], q, args.encoding), pop["y"], pop["p"]
    phi, incidence = fp.product_basis(pop["x"], q)
    allocation = fp.factor_allocation(incidence)
    u = np.sqrt(p)[:, None] * phi
    target_coefficients = fp.parity_coefficients(q)
    interaction = incidence.sum(1) >= 2
    reference = u @ (target_coefficients * interaction)
    reference /= np.linalg.norm(reference)
    centered_target = np.sqrt(p) * (y - p @ y)
    centered_target /= np.linalg.norm(centered_target)
    fixed_probes = np.column_stack([u, reference, centered_target])
    protocol = dict(**vars(args), q=q.tolist(), expected_activity=float(q.sum()),
                    positive_label_probability=float(p @ y),
                    architecture="c + mean_j a_j ReLU(w_j @ [x,1])",
                    optimizer="ordinary minibatch gradient descent; M_w=M_a=m, M_c=1",
                    updates="all parameters, all steps, fixed product distribution, no interventions",
                    response_tau=args.rate * args.response_steps,
                    pressure_allocation="mean plus equal shares of ANOVA interactions to participants",
                    exploratory=args.cohort != "confirmation",
                    primary_comparison="paired width 8 minus width 32 population loss at step 20000",
                    primary_transport="negative off-diagonal factor-to-singleton response share at step 10000",
                    weakening_relative_threshold=.1, demand_energy_fraction=.001, sources={})
    (out / "sources").mkdir()
    for source in [Path(__file__), Path(fp.__file__), Path(parity.__file__)]:
        data = source.read_bytes()
        protocol["sources"][source.name] = hashlib.sha256(data).hexdigest()
        (out / "sources" / source.name).write_bytes(data)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    bank_w, bank_a = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed_bank + seed)
        shape = (max(args.widths), x.shape[1])
        if args.initialization == "gaussian":
            bank_w.append(args.scale * rng.normal(size=shape))
            bank_a.append(args.scale * rng.normal(size=max(args.widths)))
        else:
            bank_w.append(args.scale * rng.uniform(-np.sqrt(3), np.sqrt(3), size=shape))
            bank_a.append(args.scale * rng.uniform(-np.sqrt(3), np.sqrt(3), max(args.widths)))
    bank_w, bank_a = np.array(bank_w), np.array(bank_a)
    states = {m: (bank_w[:, :m].copy(), bank_a[:, :m].copy(), np.zeros(args.seeds))
              for m in args.widths}
    rng = np.random.default_rng(args.seed_bank + 200000)
    exposures = np.zeros((args.seeds, args.features), dtype=np.int64)
    forecast_steps = {s for s in args.forecast_steps if s + max(args.horizons) <= args.steps}
    diagnostic_steps = {s for s in args.diagnostic_steps if s <= args.steps}
    evaluation_steps = set(range(0, args.steps + 1, args.stride)) | {args.steps}
    evaluation_steps |= {s + h for s in forecast_steps for h in args.horizons}
    evaluation_steps |= diagnostic_steps | forecast_steps
    curves, records, snapshots, pending = [], [], [], {}
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step in evaluation_steps:
            loss, audit_loss, conditional = [], [], []
            for m in args.widths:
                error2 = (y - fp.forward(states[m], x))**2 / 2
                loss.append(error2 @ p)
                audit_loss.append(error2.mean(1))
                conditional.append((error2 * p) @ pop["x"] / q)
            curves.append((step, np.array(loss), np.array(audit_loss),
                           np.array(conditional), exposures.copy()))
        for m in args.widths:
            state = states[m]
            if (step, m) in pending:
                for record, probes, before in pending.pop((step, m)):
                    seed = record["seed"]
                    st = tuple(v[seed] for v in state)
                    record["next_strength"] = fp.response_strength(
                        st, x, p, probes, args.rate * args.response_steps).tolist()
                    record["gate_flip_mass"] = float(np.sum(p * np.mean(
                        (st[0] @ x.T > 0) != record.pop("previous_gates"), axis=0)))
                    change = fp.kernel_change_ledger(before, st, x, p, probes)
                    record["kernel_change"] = {k: v.tolist() for k, v in change.items()}
                    realized_update = tuple((v1 - v0) / args.rate for v0, v1 in zip(before, st))
                    record["finite_loss"] = {k: float(v) for k, v in fp.step_ledger(
                        before, st, x, y, p, realized_update, args.rate).items()}
                    record["finite_rate"] = {k: float(v) for k, v in fp.rate_ledger(
                        before, st, x, y, p).items()}
            if step in diagnostic_steps or step in forecast_steps:
                snapshots.append((step, m, *(v.copy() for v in state)))
            if step in forecast_steps:
                with (out / "forecasts.jsonl").open("a") as handle:
                    for seed in range(args.seeds):
                        st = tuple(v[seed] for v in state)
                        forecast = fp.local_loss_forecasts(st, x, y, p, args.rate,
                                                          args.batch, args.horizons)
                        # Persist before taking any of the forecasted training steps.
                        handle.write(json.dumps(dict(step=step, width=m, seed=seed,
                                                     **forecast)) + "\n")
        if step == args.steps:
            ids = None
        else:
            ids = rng.choice(len(p), size=(args.seeds, args.batch), p=p)
        if step in diagnostic_steps:
            for m in args.widths:
                state = states[m]
                for seed in range(min(args.diagnostic_seeds, args.seeds)):
                    st = tuple(v[seed] for v in state)
                    ledger = fp.pressure_ledger(st, x, y, p, phi, allocation)
                    residual = y - fp.forward(st, x)
                    _, pressures = fp.residual_pressures(residual, phi, p, allocation)
                    if ids is not None:
                        realized = np.bincount(ids[seed], minlength=len(p)) / args.batch
                        pressures = np.vstack([pressures, residual * (realized / p - 1)])
                    current_demand = np.sqrt(p) * residual
                    current_demand /= np.linalg.norm(current_demand)
                    probes = np.column_stack([fixed_probes, current_demand])
                    strength, transport = fp.response_transport(
                        st, x, p, p, probes, pressures, args.rate * args.response_steps)
                    kernel_strength, kernel_signed = fp.kernel_transport(st, x, p, probes, pressures)
                    record = dict(step=step, width=m, seed=seed,
                                  **{k: float(ledger[k]) for k in ["loss", "rate", "modal_self",
                                     "modal_cancellation", "factor_cancellation"]},
                                  coefficients=ledger["coefficients"].tolist(),
                                  modal_rate=ledger["modal_rate"].tolist(),
                                  source_gram=ledger["source_gram"].tolist(),
                                  strength=strength.tolist(), transport=transport.tolist(),
                                  kernel_strength=kernel_strength.tolist(),
                                  kernel_transport=kernel_signed.tolist())
                    if ids is not None:
                        record["previous_gates"] = st[0] @ x.T > 0
                        pending.setdefault((step + 1, m), []).append((record, probes, st))
                    records.append(record)
            print(json.dumps(dict(step=step, seconds=round(time.monotonic() - started, 1),
                                  mean_loss={m: float(curves[-1][1][i].mean())
                                             for i, m in enumerate(args.widths)})), flush=True)
        if ids is None:
            break
        xb, yb = x[ids], y[ids]
        for m in args.widths:
            residual = yb - fp.forward(states[m], xb)
            update = fp.velocity(states[m], xb, residual, 1 / args.batch)
            states[m] = fp.advance(states[m], update, args.rate)
        exposures += pop["x"][ids].sum(1).astype(np.int64)
    np.savez_compressed(out / "curves.npz", steps=np.array([c[0] for c in curves]),
                        loss=np.array([c[1] for c in curves]),
                        audit_loss=np.array([c[2] for c in curves]),
                        conditional_loss=np.array([c[3] for c in curves]),
                        exposures=np.array([c[4] for c in curves]))
    snapshot_arrays = {}
    for step, width, w, a, c in snapshots:
        snapshot_arrays.update({f"{step}_{width}_w": w, f"{step}_{width}_a": a,
                                f"{step}_{width}_c": c})
    np.savez_compressed(out / "checkpoints.npz", **snapshot_arrays)
    (out / "diagnostics.json").write_text(json.dumps(records, separators=(",", ":")) + "\n")
    print(json.dumps(dict(complete=True, seconds=round(time.monotonic() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
