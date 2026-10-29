"""Independent initial-width probabilities; latent support and finite response differ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_toy_option_events import coverage
from toy_option_resources import response, task

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=295501)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "protocol.json").exists()
    names = ["discrete", "continuous", "separate_attribute", "two_refinements"]
    protocol = dict(
        count=args.count,
        seed=args.seed,
        scale=0.002,
        cases=names,
        widths=[2, 4, 6, 8, 12, 16],
        sources={},
    )
    for name in [Path(__file__).name, "toy_option_resources.py", "audit_toy_option_events.py"]:
        path = Path(__file__).with_name(name)
        protocol["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / name).write_bytes(path.read_bytes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    raw = {}
    for name in names:
        data = task(name)
        rng = np.random.default_rng(args.seed)
        w = 0.002 * rng.normal(size=(args.count, 16, data["x"].shape[1]))
        a = 0.002 * rng.normal(size=(args.count, 16))
        fixed = data["root"] * data["ref_y"]
        fixed /= np.linalg.norm(fixed)
        for m in protocol["widths"]:
            cov, strengths = [], []
            for i in range(0, args.count, 128):
                state = (w[i : i + 128, :m], a[i : i + 128, :m], np.zeros(len(w[i : i + 128])))
                cov.extend(coverage(state, data))
                strengths.extend(np.einsum("i,sij,j->s", fixed, response(state, data), fixed))
            raw[f"{name}_{m}_coverage"] = cov
            raw[f"{name}_{m}_strength"] = strengths
        print(name, flush=True)
    np.savez_compressed(out / "initial.npz", **raw)
    (out / "complete.json").write_text("{}\n")


if __name__ == "__main__":
    main()
