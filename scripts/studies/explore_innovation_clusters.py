"""Falsification probe: thicken common prototypes in the novel direction.

Choose delta=epsilon/(4*k), a quarter of the representability boundary.
Two endpoints per common cluster have target zero; the exception target is -1.
No orthant projection or iid survival law is assumed in this extension.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def train(w, x, masses, target, horizon, dt=0.2):
    w = w.copy()
    for _ in range(round(horizon / dt)):
        z = w @ x.T
        residual = -np.maximum(z, 0).mean(axis=1) - target
        update = ((z > 0) * (masses * residual)[:, None, :]) @ x
        w += dt * update
    residual = -np.maximum(w @ x.T, 0).mean(axis=1) - target
    return w, 0.5 * (masses * residual**2).sum(axis=1)


def main():
    k, epsilon, q, seeds = 4, 1.0, 0.1, 128
    delta = epsilon / (4 * k)
    common = np.repeat(np.eye(k + 1)[:k], 2, axis=0)
    common[:, -1] = np.tile([-delta, delta], k)
    v = np.ones(k + 1) / np.sqrt(k + epsilon**2)
    x = np.concatenate([common, v[None]])
    target = np.zeros(2*k + 1)
    target[-1] = -1
    masses = np.full(2*k + 1, (1-q)/(2*k))
    masses[-1] = q
    rng = np.random.default_rng(9421)
    initial = rng.normal(size=(seeds, 16, k + 1))
    initial *= 0.2 / np.linalg.norm(initial, axis=-1, keepdims=True)
    rows = []
    # Explicit width-one witness. Common activation <=0, exception activation =1.
    witness = np.full(k+1, -delta)
    witness[-1] = 1
    witness /= witness @ v
    assert np.max(common @ witness) < 1e-14
    assert abs(witness @ v - 1) < 1e-14
    assert np.all(common @ v > 0)  # no exception activation can revive
    for width in (1, 4, 16):
        w0 = initial[:, :width]
        pre, preloss = train(w0, common, np.full(2*k, 1/(2*k)),
                             np.zeros(2*k), 1000)
        absent0 = (w0 @ v <= 0).all(axis=1)
        absent = (pre @ v <= 0).all(axis=1)
        after, loss = train(pre, x, masses, target, 2000)
        after_absent = (after @ v <= 0).all(axis=1)
        assert np.all(after_absent[absent])
        rows.append(dict(width=width, preloss=float(preloss.mean()),
            initial_absence=float(absent0.mean()), before_release_absence=float(absent.mean()),
            dynamic_absence=float(np.mean(absent & ~absent0)),
            mean_loss=float(loss.mean()), seed_specific_lower_bound=float(q/2*absent.mean()),
            loss_on_predicted_failures=float(loss[absent].mean()) if absent.any() else None,
            loss_on_other_starts=float(loss[~absent].mean()) if (~absent).any() else None,
            final_absence=float(after_absent.mean())))
    result = dict(k=k, epsilon=epsilon, delta=delta, mass=q, seeds=seeds, seed=9421,
                  dt=0.2, pretraining_horizon=1000, final_horizon=2000,
                  representable_at_width=1, witness=witness.tolist(), rows=rows)
    output = Path(__file__).resolve().parents[2] / "docs/theory/innovation_clusters_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
