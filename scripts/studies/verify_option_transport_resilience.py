"""Verify saved provenance, complete cohorts, matching, and declared bound tests."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from analyze_option_transport_resilience import read

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/option_transport_resilience"


def main():
    results = {}
    for name in [
        "main",
        "affine",
        "delayed",
        "delayed_confirmation",
        "delayed_half_step",
        "delayed_intervention",
    ]:
        folder = OUT / name
        protocol = json.loads((folder / "protocol.json").read_text())
        actual = hashlib.sha256((folder / "source.py").read_bytes()).hexdigest()
        assert actual == protocol["source_sha256"], name
        assert (folder / "complete.json").exists(), name
        compile((folder / "source.py").read_text(), str(folder / "source.py"), "exec")
        results[name] = {"executed_source_hash": actual, "complete": True}
        if name.startswith("delayed") and name != "delayed_intervention":
            rows = read(folder / "outcomes.jsonl")
            assert len(rows) == protocol["count"] * len(protocol["cases"]) * 2
            assert len({(r["case"], r["width"], r["seed"]) for r in rows}) == len(rows)
            assert len(read(folder / "forecasts.jsonl")) == 2 * len(rows)
            assert all(np.isfinite(r["loss"]) and r["loss"] >= 0 for r in rows)
            results[name]["endpoints"] = len(rows)
        elif name in ["main", "affine"]:
            rows = read(folder / "forecasts.jsonl")
            assert len(rows) == 10 * len(protocol["seeds"])
            assert max(r["kernel_match_error"] for r in rows) < 1e-12
            assert max(r["function_match_error"] for r in rows) < 1e-12
            results[name]["branches"] = len(rows)
    analysis = json.loads((OUT / "delayed_analysis.json").read_text())
    for cohort, cases in analysis.items():
        for case, widths in cases.items():
            for width in ["4", "8"]:
                for forecasts in widths[width]["forecasts"].values():
                    for record in forecasts.values():
                        assert record["endpoint_floor_violations"] == 0
                        assert record["exit_bound_violations"] == 0
    intervention = json.loads((OUT / "intervention_analysis.json").read_text())
    assert intervention["clone_max_loss_difference"] < 1e-12
    assert max(intervention["matching"].values()) < 1e-12
    results["intervention_matching"] = intervention["matching"]
    results["clone_max_loss_difference"] = intervention["clone_max_loss_difference"]
    results["numerical_bound_violations"] = 0
    note = ROOT / "docs/theory/delayed_demand_option_recovery.md"
    for target in re.findall(r"\]\(([^)]+)\)", note.read_text()):
        if not target.startswith("https://"):
            assert (note.parent / target).exists(), target
    results["note_local_links"] = "all exist"
    (OUT / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
