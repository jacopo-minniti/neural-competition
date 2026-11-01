"""Attribute actual gate-induced losses/recoveries of refinement support to pressures.

Replay ordinary trajectories; audit every gate-changing update. Counterfactuals
start at the same state and use only the conditional-mean or centered residual.
The large-event threshold selects changes, not terminal training outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from toy_option_resources import advance, common, forward, jacobian, response, task, velocity

ROOT = Path(__file__).resolve().parents[2]


def coverage(state, data):
    d = jacobian(state, data)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    u, sv, _ = np.linalg.svd(d, full_matrices=False)
    keep = sv > 1e-9 * np.maximum(sv[:, :1], 1)
    fixed = data["root"] * data["ref_y"]
    fixed /= np.linalg.norm(fixed)
    return np.sum(np.einsum("sni,n->si", u, fixed) ** 2 * keep, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end", type=float, default=100)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "discrete",
            "continuous",
            "rarer",
            "centered",
            "separate_attribute",
            "two_refinements",
        ],
    )
    args = parser.parse_args()
    src, out = ROOT / args.input, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    source_protocol = json.loads((src / "protocol.json").read_text())
    dt = source_protocol["dt"]
    protocol = dict(
        input=args.input,
        end=args.end,
        dt=dt,
        threshold=args.threshold,
        cases=args.cases,
        selection="Every ordinary update with a gate change; record absolute structural coverage change above threshold. No terminal-outcome selection.",
        sources={},
        input_hashes={},
    )
    for name in [Path(__file__).name, "toy_option_resources.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    for name in args.cases:
        for m in [task(name)["capacity"], 2 * task(name)["capacity"]]:
            f = src / f"{name}_{m}_states.npz"
            protocol["input_hashes"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in args.cases:
        data = task(name)
        fixed = data["root"] * data["ref_y"]
        fixed /= np.linalg.norm(fixed)
        for m in [data["capacity"], 2 * data["capacity"]]:
            raw = np.load(src / f"{name}_{m}_states.npz")
            state = tuple(raw[f"ordinary_0_{key}"] for key in ["w", "a", "c"])
            recorded = 0
            for i in range(round(args.end / dt)):
                v = velocity(state, data)
                after = advance(state, v, dt)
                changed = np.any(
                    (state[0] @ data["x"].T > 0) != (after[0] @ data["x"].T > 0), axis=(1, 2)
                )
                ids = np.flatnonzero(changed)
                if len(ids):
                    before_s = tuple(s[ids] for s in state)
                    after_s = tuple(s[ids] for s in after)
                    b, f = coverage(before_s, data), coverage(after_s, data)
                    selected = np.flatnonzero(abs(f - b) > args.threshold)
                    if len(selected):
                        bs = tuple(s[selected] for s in before_s)
                        fs = tuple(s[selected] for s in after_s)
                        e = data["y"] - forward(bs, data["x"])
                        ec = common(e, data)
                        cs = advance(bs, velocity(bs, data, ec), dt)
                        rs = advance(bs, velocity(bs, data, e - ec), dt)
                        covc, covr = coverage(cs, data), coverage(rs, data)
                        strengths = [
                            np.einsum("i,sij,j->s", fixed, response(s, data), fixed)
                            for s in [bs, cs, rs, fs]
                        ]
                        for j, selected_id in enumerate(selected):
                            row = dict(
                                case=name,
                                width=m,
                                seed=int(ids[selected_id]),
                                time=i * dt,
                                before=float(b[selected_id]),
                                full=float(f[selected_id]),
                                common=float(covc[j]),
                                refinement=float(covr[j]),
                                common_energy=float(0.5 * np.sum(data["p"] * ec[j] ** 2)),
                                refinement_energy=float(
                                    0.5 * np.sum(data["p"] * (e[j] - ec[j]) ** 2)
                                ),
                                strength_before=float(strengths[0][j]),
                                strength_common=float(strengths[1][j]),
                                strength_refinement=float(strengths[2][j]),
                                strength_full=float(strengths[3][j]),
                            )
                            with (out / "events.jsonl").open("a") as fh:
                                fh.write(json.dumps(row, allow_nan=False) + "\n")
                            recorded += 1
                state = after
            if args.end == 100:
                assert (
                    max(
                        np.max(abs(s - raw[f"ordinary_100_{key}"]))
                        for s, key in zip(state, ["w", "a", "c"])
                    )
                    < 1e-12
                )
            print(name, m, "events", recorded, flush=True)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
