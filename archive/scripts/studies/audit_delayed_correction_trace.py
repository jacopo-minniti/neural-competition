"""Retrospective tracing of each final model-discovered residual, not a forecast.

Local main-versus-rest pulses audit which force closes that same response.
Selection uses endpoints and therefore cannot establish prospective mediation.
"""

import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import train

ROOT = Path(__file__).resolve().parents[2] / "docs/evidence/option_transport_resilience"


def predict(s, x):
    w, a, c = s
    return c[:, None] + np.mean(a[..., None] * np.maximum(w @ x.T, 0), axis=1)


def strength(s, x, p, d):
    w, a, _ = s
    z = w @ x.T
    ja = np.sum(z.clip(0) * (p * d)[:, None], axis=2)
    jw = a[..., None] * (((z > 0) * (p * d)[:, None]) @ x)
    return (
        (np.sum(ja**2, axis=1) + np.sum(jw**2, axis=(1, 2))) / a.shape[1]
        + np.sum(p * d, axis=1) ** 2
    ) / np.sum(p * d**2, axis=1)


def main():
    ar = np.load(ROOT / "delayed_confirmation/trajectories.npz")
    x, p, y = task(1, 0.0, 0.1)
    end = tuple(ar[f"delayed_4_final_{k}"] for k in ["w", "a", "c"])
    residual = y - predict(end, x)
    selected = np.flatnonzero(0.5 * np.sum(p * residual**2, axis=1) > 1e-7)
    d = residual[selected]
    rng = np.random.default_rng(282941)
    iw = 0.002 * rng.normal(size=(64, 8, 3))
    ia = 0.002 * rng.normal(size=(64, 8))
    s = (iw[selected, :4].copy(), ia[selected, :4].copy(), np.zeros(len(selected)))
    done = np.zeros(len(selected), dtype=bool)
    init = strength(s, x, p, d)
    rows = []
    for i in range(400):
        current = predict(s, x)
        error = y - current
        shared = (error @ (p * x[:, 1]))[:, None] * x[:, 1] / (p @ x[:, 1] ** 2)
        old = strength(s, x, p, d)
        nxt = train(s, x, p, y, 0.05)
        new = strength(nxt, x, p, d)
        event = (new < 1e-12) & ~done
        if event.any():
            shared_only = train(s, x, p, current + shared, 0.05)
            rest_only = train(s, x, p, current + error - shared, 0.05)
            sm = strength(shared_only, x, p, d)
            sr = strength(rest_only, x, p, d)
            for j in np.flatnonzero(event):
                rows.append(
                    dict(
                        seed=int(selected[j]),
                        time=(i + 1) * 0.05,
                        initial_strength=float(init[j]),
                        before=float(old[j]),
                        full_after=float(new[j]),
                        shared_only_after=float(sm[j]),
                        rest_only_after=float(sr[j]),
                    )
                )
        done |= event
        s = nxt
    result = dict(
        interpretation="Endpoint-selected retrospective fixed-demand trace; pulses are local counterfactuals, not long-trajectory rescue tests.",
        selected=len(selected),
        found=int(done.sum()),
        shared_alone_closes=sum(r["shared_only_after"] < 1e-12 for r in rows),
        rest_alone_closes=sum(r["rest_only_after"] < 1e-12 for r in rows),
        rows=rows,
    )
    (ROOT / "retrospective_correction_trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
