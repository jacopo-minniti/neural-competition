"""Test current reserve measures over phases of common learning, not fixed horizons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from competition_reserve import (
    capability,
    common_flow,
    forward,
    head_prediction,
    learn_to_gain,
    protected_reserve,
    reserve,
    task,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-bank", type=int, default=210731)
    parser.add_argument("--factors", type=int, default=24)
    parser.add_argument("--rate", type=float, default=.005)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("Preserve prior cohorts")
    protocol = dict(**vars(args), widths=[4, 8, 16], scale=.2,
                    initializations=["gaussian", "uniform"], rare_masses=[.04, .15],
                    spreads=[0., .15], stages=2, remaining_head_residual_fraction=.5,
                    tau=20, material_loss_fraction=.5,
                    state_quantities=["capability", "protected_reserve", "completion_reserve",
                                      "implementation_participation", "instantaneous_slope"],
                    target="Capability and vulnerable task mass after specified common prediction gain",
                    fitted_parameters=0, sources={})
    (out / "sources").mkdir(exist_ok=True)
    for name in [Path(__file__).name, "competition_reserve.py", "probabilistic_option_transport.py"]:
        source = Path(__file__).with_name(name).read_bytes()
        protocol["sources"][name] = hashlib.sha256(source).hexdigest()
        (out / "sources" / name).write_bytes(source)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for spread in protocol["spreads"]:
        for mass in protocol["rare_masses"]:
            for seed in range(args.seeds):
                train, validation, audit, meta = task(args.seed_bank + seed, factors=args.factors,
                                                       total_rare=mass, spread=spread)
                for kind in protocol["initializations"]:
                    rng = np.random.default_rng(args.seed_bank + 10000 + seed)
                    if kind == "gaussian":
                        bank_w = .2 * rng.normal(size=(16, train["x"].shape[1]))
                        bank_a = .2 * rng.normal(size=16)
                    else:
                        bank_w = .2 * rng.uniform(-np.sqrt(3), np.sqrt(3), (16, train["x"].shape[1]))
                        bank_a = .2 * rng.uniform(-np.sqrt(3), np.sqrt(3), 16)
                    for width in protocol["widths"]:
                        state = (bank_w[:width].copy(), bank_a[:width].copy(), np.array(0.))
                        for stage in range(2):
                            current, diversity = capability(state, audit, diversity=True)
                            protected, mask = protected_reserve(state, train, audit)
                            gain = .5 * (meta["target"] - head_prediction(state))
                            completion, exposure = reserve(state, audit, gain)
                            small = common_flow(state, 1e-5)
                            slope = (capability(small, audit) - current) / (
                                head_prediction(small) - head_prediction(state))
                            record = dict(spread=spread, rare_mass=mass, seed=seed,
                                          initialization=kind, width=width, stage=stage,
                                          current=current.tolist(), protected=protected.tolist(),
                                          completion=completion.tolist(),
                                          linear=(current + gain * slope).tolist(),
                                          diversity=diversity.tolist(), protected_units=int(mask.sum()),
                                          common_gain=gain, pressure_budget=exposure,
                                          initial_head_prediction=head_prediction(state))
                            # Save the state description before measuring the actual learning phase.
                            with (out / "state_descriptions.jsonl").open("a") as f:
                                f.write(json.dumps(record) + "\n")
                            end, diagnostics = learn_to_gain(state, train, gain, rate=args.rate)
                            observed = capability(end, audit)
                            record.update(actual=observed.tolist(), diagnostics=diagnostics,
                                          loss=float(.5 * np.sum(validation["p"] * (
                                              validation["y"] - forward(end, validation["x"]))**2)))
                            rows.append(record)
                            state = end
                print(json.dumps(dict(spread=spread, mass=mass, seed=seed)), flush=True)
    result = dict(protocol=protocol, rows=rows)
    (out / "results.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
