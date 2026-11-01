"""Prospective conditional probability tests on 24 power-law refinements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from power_law_tasks import (
    block_forecast,
    dataset,
    forward,
    frozen_forecast,
    influence,
    initialize,
    local_forecast,
    response,
    sgd_step,
    specification,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-bank", type=int, default=73100)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 100, 500])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--widths", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--initializations", nargs="+", default=[
        "small_gaussian", "fanin_gaussian", "fanin_uniform"])
    parser.add_argument("--cases", nargs="+", default=["prototype", "associated"])
    parser.add_argument("--rate", type=float, default=.025)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--uncentered", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Do not overwrite a previous cohort")
    protocol = dict(**vars(args), factors=24, alpha=1.15, tau=5,
                    changes=[0.05, 0.1, 0.25],
                    availability_resolutions=[0.05, 0.1, 0.25], block_size=4,
                    prediction="Frozen-checkpoint force, exact multinomial aggregate exposure",
                    uncertainty="Independent forecast and validation batch streams; no fitted parameters",
                    audit_measure="Fixed balanced factor-intervention panel, not population risk",
                    capacity="Exact width 50 witness for both entire bounded task supports",
                    sources={})
    for name in [Path(__file__).name, "power_law_tasks.py"]:
        source = Path(__file__).with_name(name).read_bytes()
        protocol["sources"][name] = hashlib.sha256(source).hexdigest()
        (out / "sources").mkdir(exist_ok=True)
        (out / "sources" / name).write_bytes(source)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for case_index, case in enumerate(args.cases):
        spec = specification(richer=case == "associated", centered=not args.uncentered)
        for seed in range(args.seeds):
            bank_seed = args.seed_bank + seed
            train, validation, audit, _ = dataset(spec, bank_seed + 10000)
            np.savez_compressed(out / f"{case}_{seed}_data.npz", q=spec["q"],
                                train_marginal=train["p"] @ train["indicator"],
                                validation_marginal=validation["p"] @ validation["indicator"],
                                factor_covariance=np.cov(train["indicator"], rowvar=False,
                                                         aweights=train["p"]))
            for init_index, kind in enumerate(args.initializations):
                for width in args.widths:
                    state = initialize(width, train["x"].shape[1], kind, bank_seed)
                    training_rng = np.random.default_rng(bank_seed + 20000)
                    for step in range(max(args.checkpoints) + 1):
                        if step in args.checkpoints:
                            key = f"{case}_{kind}_s{seed}_m{width}_t{step}"
                            start = response(state, audit)
                            forecast = {"initial_strength": start}
                            inf = influence(state, train, audit)
                            forecast.update({k: inf[k] for k in ["mean", "covariance", "diversity"]})
                            # Persist all forecasts before drawing continuation outcomes.
                            base_rng_seed = bank_seed + 100000 * (case_index + 1) + 1000 * init_index
                            for horizon in args.horizons:
                                rng = np.random.default_rng(base_rng_seed + 31 * step + horizon)
                                pred = frozen_forecast(state, train, horizon, args.rate,
                                                       args.batch, args.samples, rng)
                                forecast[f"h{horizon}"] = response(pred, audit)
                                refreshed = block_forecast(
                                    state, train, horizon, args.rate, args.batch, args.samples,
                                    np.random.default_rng(base_rng_seed + 800000 + 31 * step + horizon))
                                forecast[f"block_h{horizon}"] = response(refreshed, audit)
                                for method, bilinear in [("linear", False), ("bilinear", True)]:
                                    local = local_forecast(
                                        state, train, horizon, args.rate, args.batch, args.samples,
                                        np.random.default_rng(base_rng_seed + (1600000 if bilinear else 1200000)
                                                              + 31 * step + horizon), bilinear=bilinear)
                                    forecast[f"{method}_h{horizon}"] = response(local, audit)
                            np.savez_compressed(out / f"{key}_forecast.npz", **forecast)
                            samples = tuple(np.broadcast_to(s, (args.samples,) + s.shape).copy()
                                            for s in state)
                            actual = {}
                            rng = np.random.default_rng(base_rng_seed + 9000000 + step)
                            for horizon in range(1, max(args.horizons) + 1):
                                samples = sgd_step(samples, train, args.rate, args.batch, rng)
                                if horizon in args.horizons:
                                    actual[f"h{horizon}"] = response(samples, audit)
                            np.savez_compressed(out / f"{key}_actual.npz", **actual)
                            loss = .5 * np.sum(validation["p"] * (
                                forward(state, validation["x"]) - validation["y"]) ** 2)
                            tail = start[16:]
                            record = dict(key=key, case=case, initialization=kind, width=width,
                                          seed=seed, step=step, validation_loss=float(loss),
                                          mean_strength=float(start.mean()),
                                          tail_strength=float(tail.mean()),
                                          mean_diversity=float(inf["diversity"].mean()),
                                          tail_diversity=float(inf["diversity"][16:].mean()))
                            with (out / "states.jsonl").open("a") as f:
                                f.write(json.dumps(record) + "\n")
                            print(json.dumps(record), flush=True)
                        if step < max(args.checkpoints):
                            state = sgd_step(state, train, args.rate, args.batch, training_rng)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
