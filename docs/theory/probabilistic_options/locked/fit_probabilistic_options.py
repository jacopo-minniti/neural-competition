"""Development-only model selection followed by frozen fresh-run forecasts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from probabilistic_options import FEATURES, clock, development_paths, episodes, fit, probability

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/theory/probabilistic_options/locked"


def score(model, rows, event):
    exposure = None if event == "closure" else [clock(r["horizon"], model["clock"]) for r in rows]
    pred = probability(model, [r["features"] for r in rows], exposure)
    y = np.array([r["event"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    return float(np.average((pred - y) ** 2, weights=w))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "models.json").exists()
    paths, sources = development_paths(ROOT / "docs/theory/implementation_state")
    close, open_rows, queries = episodes(paths)
    result = dict(
        protocol="Fit only historical data. Four folds hold out seed indices across all families, widths and banks. Select smallest feature set within one standard error of minimum CV Brier. Opening clock also chosen on development. Fresh tests are frozen before training.",
        sources={},
        candidates={},
        selected={},
    )
    for path in sources:
        result["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for event, train, test in [
        ("closure", close, close),
        ("opening", open_rows, [r for r in queries if r["horizon"] <= 500]),
    ]:
        candidates = []
        for name, columns in FEATURES.items():
            for kind in ["log", "linear"] if event == "opening" else ["log"]:
                scores = []
                for fold in range(4):
                    tr = [r for r in train if r["seed"] % 4 != fold]
                    te = [r for r in test if r["seed"] % 4 == fold]
                    model = fit(tr, columns, kind)
                    scores.append(score(model, te, event))
                candidates.append(
                    dict(
                        name=name,
                        clock=kind,
                        scores=scores,
                        mean=float(np.mean(scores)),
                        se=float(np.std(scores, ddof=1) / 2),
                    )
                )
        best = min(candidates, key=lambda r: r["mean"])
        eligible = [r for r in candidates if r["mean"] <= best["mean"] + best["se"]]
        chosen = min(eligible, key=lambda r: (len(FEATURES[r["name"]]), r["mean"]))
        result["candidates"][event] = candidates
        result["selected"][event] = dict(
            name=chosen["name"], **fit(train, FEATURES[chosen["name"]], chosen["clock"])
        )
        result["selected"][event + "_constant"] = fit(train, [], chosen["clock"])
        # Save all comparators now; none may be refit after fresh outcomes.
        result["selected"][event + "_task"] = fit(train, FEATURES["task"], chosen["clock"])
        result["selected"][event + "_width"] = fit(train, FEATURES["width"], chosen["clock"])
        result["selected"][event + "_options"] = fit(train, FEATURES["options"], chosen["clock"])
        result["selected"][event + "_resource"] = fit(train, FEATURES["resource"], chosen["clock"])
        result["selected"][event + "_no_width"] = fit(
            train, [k for k in FEATURES[chosen["name"]] if k != "log_width"], chosen["clock"]
        )
    result["counts"] = dict(
        closure=len(close),
        closure_events=sum(r["event"] for r in close),
        recovery_intervals=len(open_rows),
        recovery_events=sum(r["event"] for r in open_rows),
        recovery_queries=len(queries),
    )
    (OUT / "models.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in [
        Path(__file__).name,
        "probabilistic_options.py",
        "implementation_moments.py",
        "toy_option_resources.py",
    ]:
        path = Path(__file__).with_name(name)
        (OUT / name).write_bytes(path.read_bytes())
    print(
        json.dumps(
            dict(
                counts=result["counts"],
                candidates=result["candidates"],
                selected=result["selected"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
