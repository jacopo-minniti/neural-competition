"""Independent invariants and provenance for the implementation-slack studies."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from explore_recruitable_workspace import audit, direction, predict
from probe_workspace_recovery import exact_target

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/recruitable_workspace"


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        analytic_audit=audit(),
        cohorts={},
        hashes={},
    )
    for folder in ["pilot", "confirmation", "extra_refinement", "affine", "active_initialization"]:
        p = OUT / folder
        protocol = json.loads((p / "protocol.json").read_text())
        assert (
            hashlib.sha256((p / "source.py").read_bytes()).hexdigest() == protocol["source_sha256"]
        )
        assays = read(p / "assays.jsonl")
        interventions = read(p / "interventions.jsonl")
        forecasts = read(p / "forecasts.jsonl")
        ordinary = read(p / "ordinary.jsonl")
        assert len(assays) == len(protocol["seeds"]) * 8 * 5
        assert len(interventions) == len(protocol["seeds"]) * 5 * 5
        assert len(ordinary) == len(protocol["seeds"]) * 2 * 5
        max_origin = max_clone = max_match = max_demand = max_rep = 0.0
        for seed in protocol["seeds"]:
            a = [r for r in assays if r["seed"] == seed]
            for field in ["train", "validation"]:
                origins = [r[field] for r in a if r["step"] == 0]
                max_origin = max(max_origin, max(origins) - min(origins))
                for r in a:
                    if r["width"] == 9 and r["history"] == "peer":
                        cr = next(
                            q
                            for q in interventions
                            if q["seed"] == seed and q["arm"] == "clone" and q["step"] == r["step"]
                        )
                        max_clone = max(max_clone, abs(r[field] - cr[field]))
            z = np.load(p / f"states_{seed}.npz")
            states = {m: tuple(z[f"{m}_pause_{k}"] for k in ["w", "a", "c"]) for m in [9, 36]}
            for prefix, xkey, ykey, dkey in [
                ("", "x", "y", "demand"),
                ("v", "vx", "vy", "vdemand"),
            ]:
                d = z[ykey] - (predict(states[9], z[xkey]) + predict(states[36], z[xkey])) / 2
                max_demand = max(max_demand, float(np.max(np.abs(d - z[dkey]))))
            pp = next(
                q["initial_gain_rate"]
                for q in forecasts
                if q["seed"] == seed and q.get("width") == 9 and q.get("history") == "peer"
            )
            rr = next(
                q for q in forecasts if q["seed"] == seed and q.get("arm") == "fresh_gain_matched"
            )
            max_match = max(max_match, abs(pp - rr["initial_gain_rate"] * rr["rate_multiplier"]))
            if not protocol["affine"]:
                max_rep = max(
                    max_rep, float(np.max(np.abs(predict(exact_target(seed), z["vx"]) - z["vy"])))
                )
            if protocol.get("active_init"):
                w = z["9_initial_w"]
                assert np.all(np.abs(w[:, 1]) < np.abs(w[:, 0]))
            # Check the recorded initial forecast with the saved state and data.
            peer = tuple(z[f"9_peer_{k}"] for k in ["w", "a", "c"])
            _, power = direction(peer, z["x"], z["demand"] + predict(peer, z["x"]))
            assert abs(pp - power) < 1e-12
        assert max(max_origin, max_clone, max_match, max_demand, max_rep) < 1e-11
        report["cohorts"][folder] = dict(
            n=len(protocol["seeds"]),
            max_origin_error=max_origin,
            max_clone_error=max_clone,
            max_gain_match_error=max_match,
            max_demand_error=max_demand,
            max_representation_error=max_rep,
        )
    for folder in ["recovery", "recovery_active"]:
        p = OUT / folder
        protocol = json.loads((p / "protocol.json").read_text())
        assert (
            hashlib.sha256((p / "source.py").read_bytes()).hexdigest() == protocol["source_sha256"]
        )
        audits = json.loads((p / "audits.json").read_text())
        assert max(r["recycled_function_error"] for r in audits) == 0
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ["verification.json", "summary.png"]:
            report["hashes"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in [
        "explore_recruitable_workspace",
        "probe_workspace_recovery",
        "analyze_recruitable_workspace",
        "audit_restart_evidence",
        "verify_workspace_evidence",
    ]:
        p = ROOT / "scripts/studies" / f"{name}.py"
        report["hashes"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    report["all_checks_passed"] = True
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(all_checks_passed=True, cohorts=report["cohorts"]), indent=2))


if __name__ == "__main__":
    main()
