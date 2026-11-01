"""Run controlled experiments on standardized bipolar parity with online episode detection."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from competition_reserve import (
    capability,
    detect_dominant_episode,
    directional_prediction,
    online_reserve,
    protected_reserve,
)


def make_parity_task(factors=4, alpha=1.15, q0=0.35, standardized=True, bipolar=True):
    """Generate exact product distribution for parity with specified encoding and targets."""
    z = np.array(list(itertools.product([0., 1.], repeat=factors)))
    q = q0 * (np.arange(1, factors + 1, dtype=float) ** (-alpha))
    p = np.prod(np.where(z == 1, q, 1 - q), axis=1)

    if bipolar:
        y = (-1.) ** z.sum(axis=1)
    else:
        y = (z.sum(axis=1) % 2).astype(float)

    std = np.sqrt(q * (1 - q))
    if standardized:
        x = (z - q) / std
        empty_idx = 0
        x_empty = x[empty_idx]
        ell = float(np.linalg.norm(x_empty))
        u_true = x_empty / ell
    else:
        # Raw binary with intercept column
        x = np.column_stack([z, np.ones(len(z))])
        empty_idx = 0
        x_empty = x[empty_idx]
        ell = float(np.linalg.norm(x_empty))
        u_true = x_empty / ell

    # Audit directions: Fourier basis
    directions = np.zeros((len(z), factors))
    for i in range(factors):
        directions[:, i] = ((-1.) ** z[:, i]) / np.sqrt(len(z))

    audit = dict(x=x, p=np.full(len(z), 1.0 / len(z)), directions=directions)
    head_center = x_empty.copy()
    head_radius = np.zeros_like(head_center)
    train = dict(x=x, y=y, p=p, head_center=head_center, head_radius=head_radius)

    return train, audit, dict(q=q, z=z, empty_idx=empty_idx, u_true=u_true, ell_true=ell)


def forward(state, x):
    w, a, c = state
    pre = w @ x.T
    if np.ndim(c) == 0:
        return float(c) + np.mean(a[:, None] * np.maximum(pre, 0), axis=0)
    return c[..., None] + np.mean(a[..., None] * np.maximum(pre, 0), axis=-2)


def velocity(state, x, y, p):
    w, a, _ = state
    pre = w @ x.T
    pe = p * (y - forward(state, x))
    return (
        a[..., None] * (((pre > 0) * pe[..., None, :]) @ x),
        np.sum(np.maximum(pre, 0) * pe[..., None, :], axis=-1),
        pe.sum(-1),
    )


SETTINGS = [
    dict(name="primary_d4", factors=4, alpha=1.15, q0=0.35, standardized=True, bipolar=True, seeds=128),
    dict(name="dimension_d6", factors=6, alpha=1.15, q0=0.35, standardized=True, bipolar=True, seeds=64),
    dict(name="flatter_alpha05", factors=4, alpha=0.50, q0=0.35, standardized=True, bipolar=True, seeds=64),
    dict(name="uncentered_control", factors=4, alpha=1.15, q0=0.35, standardized=False, bipolar=True, seeds=64),
    dict(name="zero_label_control", factors=4, alpha=1.15, q0=0.35, standardized=True, bipolar=False, seeds=64),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/natural_parity")
    parser.add_argument("--seed-bank", type=int, default=50731)
    parser.add_argument("--rate", type=float, default=0.01)
    parser.add_argument("--followup-steps", type=int, default=2500)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists() or (out / "results.json").exists():
        raise RuntimeError("Preserve prior results; choose a new output directory")

    widths = [4, 8, 16, 32]
    protocol = dict(
        output=str(out),
        seed_bank=args.seed_bank,
        rate=args.rate,
        followup_steps=args.followup_steps,
        widths=widths,
        settings=SETTINGS,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    all_rows = []

    for setting in SETTINGS:
        name = setting["name"]
        factors = setting["factors"]
        n_seeds = setting["seeds"]
        print(f"\n--- Running Setting: {name} ({factors} factors, {n_seeds} seeds) ---", flush=True)

        train, audit, meta = make_parity_task(
            factors=factors,
            alpha=setting["alpha"],
            q0=setting["q0"],
            standardized=setting["standardized"],
            bipolar=setting["bipolar"],
        )
        x, y, p = train["x"], train["y"], train["p"]
        empty_idx = meta["empty_idx"]
        u_oracle = meta["u_true"]
        ell_oracle = meta["ell_true"]
        q = meta["q"]

        # Pre-generate initialization banks
        w_bank = []
        a_bank = []
        for s in range(n_seeds):
            rng = np.random.default_rng(args.seed_bank + s)
            w_bank.append(0.2 * rng.normal(size=(max(widths), x.shape[-1])))
            a_bank.append(0.2 * rng.normal(size=max(widths)))

        for width in widths:
            print(f"  Width {width:2d}...", end=" ", flush=True)
            w_init = np.array([w[:width].copy() for w in w_bank])
            a_init = np.array([a[:width].copy() for a in a_bank])
            c_init = np.zeros(n_seeds)
            states = (w_init, a_init, c_init)

            # 1. Predictions for each seed before training
            initial_records = []
            target_f = np.zeros(n_seeds)
            for s in range(n_seeds):
                st = (states[0][s], states[1][s], float(states[2][s]))
                beta0 = capability(st, audit)

                # SVD online episode detection
                u_det, coherence, strength, ell_det = detect_dominant_episode(st, x, y, p)

                # Check target gain: 50% remaining error on dominant empty pattern
                f_emp0 = directional_prediction(st, u_oracle, ell_oracle)
                target_val = y[empty_idx]
                gain = 0.5 * (target_val - f_emp0)
                if abs(gain) < 1e-4:
                    gain = 0.5  # fallback for zero label control
                target_f[s] = f_emp0 + gain

                # Online detected prediction
                pred_beta_det, exp_det = online_reserve(st, audit, u_det, max(gain, 1e-4), norm_x=ell_det)

                # Oracle prediction
                pred_beta_ora, exp_ora = online_reserve(st, audit, u_oracle, max(gain, 1e-4), norm_x=ell_oracle)

                # Protected floor
                safe_floor, _ = protected_reserve(st, train, audit)

                cos_sim = float(abs(np.dot(u_det, u_oracle)))
                initial_records.append(dict(
                    setting=name,
                    seed=s,
                    width=width,
                    q=q.tolist(),
                    beta0=beta0.tolist(),
                    u_coherence=coherence,
                    u_strength=strength,
                    u_cosine_to_oracle=cos_sim,
                    gain=float(gain),
                    pred_det=pred_beta_det.tolist(),
                    exp_det=float(exp_det),
                    pred_ora=pred_beta_ora.tolist(),
                    exp_ora=float(exp_ora),
                    safe_floor=safe_floor.tolist(),
                ))

            # 2. Phase 1 training (until each seed reaches target_f)
            cur_states = (states[0].copy(), states[1].copy(), states[2].copy())
            done = np.zeros(n_seeds, dtype=bool)
            phase1_steps = np.zeros(n_seeds, dtype=int)
            phase1_states = [None] * n_seeds

            for step in range(1, 4001):
                vw, va, vc = velocity(cur_states, x, y, p)
                cur_states = (
                    cur_states[0] + args.rate * vw,
                    cur_states[1] + args.rate * va,
                    cur_states[2] + args.rate * vc,
                )
                preds_empty = np.array([
                    directional_prediction((cur_states[0][s], cur_states[1][s], cur_states[2][s]), u_oracle, ell_oracle)
                    for s in range(n_seeds)
                ])
                hit = (~done) & (preds_empty >= target_f)
                for s in np.flatnonzero(hit):
                    phase1_steps[s] = step
                    phase1_states[s] = (cur_states[0][s].copy(), cur_states[1][s].copy(), float(cur_states[2][s]))
                done |= hit
                if done.all():
                    break

            # For any seed not hitting by max steps, save current state
            for s in range(n_seeds):
                if phase1_states[s] is None:
                    phase1_steps[s] = 4000
                    phase1_states[s] = (cur_states[0][s].copy(), cur_states[1][s].copy(), float(cur_states[2][s]))

            # 3. Extended training to follow-up budget (Episode 2 / Convergence)
            rem_steps = max(args.followup_steps - int(phase1_steps.mean()), 500)
            for _ in range(rem_steps):
                vw, va, vc = velocity(cur_states, x, y, p)
                cur_states = (
                    cur_states[0] + args.rate * vw,
                    cur_states[1] + args.rate * va,
                    cur_states[2] + args.rate * vc,
                )

            # Record outcomes
            for s in range(n_seeds):
                rec = initial_records[s]
                st_p1 = phase1_states[s]
                beta_p1 = capability(st_p1, audit)
                f_p1 = forward(st_p1, x)
                loss_p1 = float(0.5 * np.sum(p * (y - f_p1)**2))

                st_end = (cur_states[0][s], cur_states[1][s], float(cur_states[2][s]))
                beta_end = capability(st_end, audit)
                f_end = forward(st_end, x)
                loss_end = float(0.5 * np.sum(p * (y - f_end)**2))
                acc_end = float(np.mean(np.sign(f_end) == y)) if setting["bipolar"] else float(np.mean((f_end > 0.5) == y))

                rec.update(dict(
                    phase1_step=int(phase1_steps[s]),
                    actual_p1=beta_p1.tolist(),
                    loss_p1=loss_p1,
                    actual_end=beta_end.tolist(),
                    loss_end=loss_end,
                    acc_end=acc_end,
                ))
                all_rows.append(rec)

            print(f"done (mean p1 steps: {phase1_steps.mean():.1f})", flush=True)

    (out / "results.json").write_text(json.dumps(dict(protocol=protocol, rows=all_rows), separators=(",", ":")) + "\n")
    print(f"\nAll experiments complete. Results saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
