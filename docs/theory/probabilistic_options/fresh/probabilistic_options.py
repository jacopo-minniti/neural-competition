"""Small conditional hazard models; spectral options remain the capability definition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from implementation_moments import hierarchy

FEATURES = {
    "constant": [],
    "task": ["log_frequency", "alignment", "log_components"],
    "width": ["log_frequency", "alignment", "log_components", "log_width"],
    "options": [
        "log_frequency",
        "alignment",
        "log_components",
        "log_width",
        "log_strength",
        "coverage",
    ],
    "demand": [
        "log_frequency",
        "alignment",
        "log_components",
        "log_width",
        "log_strength",
        "coverage",
        "log_demand",
    ],
    "resource": [
        "log_frequency",
        "alignment",
        "log_components",
        "log_width",
        "log_strength",
        "coverage",
        "log_demand",
        "group_fraction",
    ],
}
FEATURES.update(
    {
        "minimal_width": ["alignment", "log_components", "log_width"],
        "width_only": ["alignment", "log_width"],
        "state": ["coverage", "log_strength", "log_demand"],
        "state_task": ["coverage", "log_strength", "log_demand", "log_components"],
        "compact": ["coverage", "log_demand", "log_components"],
        "strength": ["log_strength", "log_demand", "log_components"],
        "compact_width": ["coverage", "log_demand", "log_components", "log_width"],
    }
)


def weighted_task(n, family, parameter, overlap):
    data = hierarchy(n, 1.0, overlap)
    weights = (
        parameter ** np.arange(n) if family == "geometric" else (1 + np.arange(n)) ** (-parameter)
    )
    q = 0.2 * weights / weights.sum()
    p = np.r_[1 - q.sum(), q][data["context"]] / 4
    root = np.sqrt(p)
    refs = (
        np.stack([(data["context"] == j + 1) - q[j] for j in range(n)], axis=1)
        * (data["z"] @ data["v"])[:, None]
    )
    columns = root[:, None] * refs
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return dict(
        data,
        p=p,
        root=root,
        q=q,
        directions=columns / np.linalg.norm(columns, axis=0),
        u=u[:, s > 1e-12],
        gram=root[:, None] * (data["x"] @ data["x"].T) * root[None, :],
        family=family,
        parameter=parameter,
    )


def features(row, j, q):
    return dict(
        log_frequency=float(np.log(q)),
        alignment=float(row["overlap"]),
        log_components=float(np.log(row["n"])),
        log_width=float(np.log(row["width"] / (2 + 2 * row["n"]))),
        log_strength=float(np.log(max(row["strength_1000"][j], 1e-9))),
        coverage=float(row["coverage"][j]),
        log_demand=float(np.log(max(abs(row["gap_error"][j]), 1e-3))),
        group_fraction=float(row["gate_groups"] / row["width"]),
    )


def clock(t, kind):
    return np.log(t / 20.0) if kind == "log" else (t - 20.0) / 20.0


def probability(model, feature_list, exposure=None):
    x = np.array([[f[k] for k in model["features"]] for f in feature_list], dtype=float).reshape(
        len(feature_list), len(model["features"])
    )
    xx = np.column_stack(
        [np.ones(len(x)), (x - np.array(model["center"])) / np.array(model["scale"])]
    )
    rate = np.exp(np.clip(xx @ model["coef"], -30, 20))
    return -np.expm1(-rate * (1 if exposure is None else np.array(exposure)))


def loss_gradient_hessian(eta, y):
    rate = np.exp(np.clip(eta, -30, 20))
    loss = np.where(y > 0, -np.log(np.maximum(-np.expm1(-rate), 1e-300)), rate)
    # Stable derivatives of the Bernoulli complementary-log-log likelihood.
    e = np.exp(-rate)
    den = np.maximum(-np.expm1(-rate), 1e-300)
    grad = np.where(y > 0, -rate * e / den, rate)
    hess = np.where(y > 0, rate * e * (rate - den) / (den**2), rate)
    return loss, grad, np.maximum(hess, 0)


def fit(rows, feature_names, kind="log", penalty=0.01):
    x = np.array([[r["features"][k] for k in feature_names] for r in rows]).reshape(
        len(rows), len(feature_names)
    )
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    xx = np.column_stack([np.ones(len(x)), (x - center) / scale])
    y = np.array([r["event"] for r in rows], dtype=float)
    weight = np.array([r["weight"] for r in rows])
    weight /= weight.sum()
    offset = np.array([r.get(f"offset_{kind}", 0) for r in rows])
    coef = np.zeros(xx.shape[1])
    coef[0] = np.log(-np.log(1 - np.clip(np.average(y, weights=weight), 0.01, 0.99)))
    ridge = np.r_[0.0, np.full(len(feature_names), penalty)]

    def value(b):
        return np.sum(weight * loss_gradient_hessian(xx @ b + offset, y)[0]) + 0.5 * np.sum(
            ridge * b * b
        )

    for _ in range(100):
        _, g, h = loss_gradient_hessian(xx @ coef + offset, y)
        grad = xx.T @ (weight * g) + ridge * coef
        H = xx.T @ ((weight * h)[:, None] * xx) + np.diag(ridge) + np.eye(len(coef)) * 1e-9
        step = np.linalg.solve(H, grad)
        old = value(coef)
        alpha = 1.0
        while alpha > 1e-8 and value(coef - alpha * step) > old:
            alpha *= 0.5
        coef -= alpha * step
        if np.linalg.norm(alpha * step) < 1e-8:
            break
    return dict(
        features=feature_names,
        center=center.tolist(),
        scale=scale.tolist(),
        coef=coef.tolist(),
        clock=kind,
        penalty=penalty,
    )


def episodes(paths):
    """Landmark first-passage targets; all features precede their prediction window."""
    closure = []
    recovery = []
    queries = []
    for identity, path in paths.items():
        path = sorted(path, key=lambda r: r["time"])
        start = path[0]
        at20 = next((r for r in path if r["time"] == 20), None)
        if at20 is None:
            continue
        q = np.array(start.get("q", hierarchy(start["n"], start["ratio"], start["overlap"])["q"]))
        for j in range(start["n"]):
            ident = dict(
                bank=start["bank"],
                seed=start["seed"],
                case=start["case"],
                width=start["width"],
                component=j + 1,
                weight=1 / start["n"],
            )
            if start["coverage"][j] >= 0.95:
                early = [r for r in path if r["time"] <= 20]
                mastery = next(
                    (r["time"] for r in early if abs(r["gap_error"][j]) <= 0.1), float("inf")
                )
                event = any(r["time"] < mastery and r["coverage"][j] < 0.95 for r in early)
                closure.append(dict(**ident, features=features(start, j, q[j]), event=int(event)))
            if at20["coverage"][j] >= 0.95 or abs(at20["gap_error"][j]) <= 0.1:
                continue
            feature = features(at20, j, q[j])
            previous = 20.0
            first = None
            for row in path:
                t = row["time"]
                if t <= 20:
                    continue
                event = row["coverage"][j] >= 0.95
                recovery.append(
                    dict(
                        **ident,
                        features=feature,
                        event=int(event),
                        offset_log=float(np.log(clock(t, "log") - clock(previous, "log"))),
                        offset_linear=float(np.log(clock(t, "linear") - clock(previous, "linear"))),
                    )
                )
                previous = t
                if event:
                    first = t
                    break
            for t in [50.0, 100.0, 500.0, 2000.0]:
                if t <= path[-1]["time"]:
                    queries.append(
                        dict(
                            **ident,
                            features=feature,
                            horizon=t,
                            event=int(first is not None and first <= t),
                        )
                    )
    return closure, recovery, queries


def development_paths(root):
    paths = {}
    sources = []
    # Seed bank is part of identity; finest available ordinary path wins per run.
    for collection in ["pilot", "confirmation", "wider", "step_check", "finer_step"]:
        folder = Path(root) / collection
        protocol = json.loads((folder / "protocol.json").read_text())
        sources.append(folder / "trajectory.jsonl")
        local = {}
        for line in (folder / "trajectory.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["bank"] = protocol["seed"]
            local.setdefault((row["bank"], row["case"], row["width"], row["seed"]), []).append(row)
        paths.update(local)
    return paths, sources
