"""Check derived probabilities and an explicit trap; no fitted parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from probabilistic_option_transport import (
    blind_joint_probability,
    blind_probability,
    transition_probabilities,
    transport,
)


def advance(w, a, c, x, p, y, dt):
    pre = w @ x.T
    h = np.maximum(pre, 0)
    e = y - c - np.mean(a[:, None] * h, axis=0)
    pe = p * e
    return (w + dt * a[:, None] * ((pre > 0) * pe) @ x,
            a + dt * (h @ pe), c + dt * pe.sum())


def run():
    rng = np.random.default_rng(20261029)
    draws = 400_000
    initial_w = rng.normal(size=(draws, 2))
    initial_a = rng.normal(size=draws)
    rows = []
    for theta in [np.pi / 12, np.pi / 3]:
        probes = np.array([[-np.cos(theta), np.sin(theta)],
                           [-np.cos(theta), -np.sin(theta)]])
        initial_blind = np.all(initial_w @ probes.T <= 0, axis=1)
        for exposure in [0.5, 1.0, 2.0]:
            w, _ = transport(initial_w, initial_a, exposure)
            blind = np.all(w @ probes.T <= 0, axis=1)
            predicted = blind_probability(theta, exposure)
            joint = blind_joint_probability(theta, exposure)
            se = np.sqrt(predicted * (1 - predicted) / draws)
            joint_se = np.sqrt(joint * (1 - joint) / draws)
            observed = float(np.mean(blind))
            observed_joint = float(np.mean(blind & initial_blind))
            assert abs(observed - predicted) < 6 * se
            assert abs(observed_joint - joint) < 6 * joint_se
            rows.append(dict(theta_over_pi=theta / np.pi, exposure=exposure,
                             predicted_block_blind=predicted, observed_block_blind=observed,
                             binomial_se=se, predicted_joint_blind=joint,
                             observed_joint_blind=observed_joint,
                             widths={str(m): transition_probabilities(theta, exposure, m)
                                     for m in [4, 8]}))

    theta, q, target, exposure = np.pi / 3, 0.01, 4.0, 1.5
    x = np.array([[1., 0.], [-np.cos(theta), np.sin(theta)],
                  [-np.cos(theta), -np.sin(theta)]])
    p, y = np.array([1 - q, q / 2, q / 2]), np.array([target, 1., -1.])
    w0 = np.array([[.2, .4], [.2, -.4], [.2, .4], [.2, -.4]])
    a0 = np.full(4, .4)
    pre = w0 @ x.T
    incoming = a0[:, None, None] * (pre > 0)[..., None] * x[None]
    jac = np.concatenate([np.maximum(pre, 0).T, incoming.transpose(1, 0, 2).reshape(3, -1),
                          np.ones((3, 1))], axis=1)
    assert np.linalg.matrix_rank(jac) == 3
    wt, at = transport(w0, a0, exposure)
    ct = exposure
    eu = target - ct - np.mean(at * wt[:, 0])
    assert eu > 0 and np.max(wt @ x[1:].T) < 0 and np.min(at) > 0

    # Integrate the exact clock, then compare full-task training from the start.
    grid, weight = np.polynomial.legendre.leggauss(128)
    pressure = exposure * (grid + 1) / 2
    along = .2 * np.cosh(pressure) + .4 * np.sinh(pressure)
    readout = .4 * np.cosh(pressure) + .2 * np.sinh(pressure)
    elapsed = exposure / 2 * np.sum(weight / ((1 - q) * (target - pressure - along * readout)))
    natural = []
    for step in [.002, .001]:
        n = int(np.ceil(elapsed / step))
        dt = elapsed / n
        w, a, c = w0.copy(), a0.copy(), 0.
        for _ in range(n):
            w, a, c = advance(w, a, c, x, p, y, dt)
        e_common = target - c - np.mean(a * w[:, 0])
        rare_margin = -float(np.max(w @ x[1:].T))
        assert rare_margin > 0 and np.min(a) > 0 and c > 0 and e_common > 0
        natural.append(dict(step=dt, common_residual=float(e_common), intercept=float(c),
                            minimum_readout=float(a.min()), rare_inactive_margin=rare_margin))

    # Check the exact trap's full-objective continuation at two Euler resolutions.
    continuations = []
    release_q = .1
    release_p = np.array([1 - release_q, release_q / 2, release_q / 2])
    for step in [.01, .005]:
        w, a, c = wt.copy(), at.copy(), ct
        max_rare_pre = float(np.max(w @ x[1:].T))
        for _ in range(round(50 / step)):
            w, a, c = advance(w, a, c, x, release_p, y, step)
            max_rare_pre = max(max_rare_pre, float(np.max(w @ x[1:].T)))
        e = y - c - np.mean(a[:, None] * np.maximum(w @ x.T, 0), axis=0)
        loss = float(.5 * np.sum(release_p * e**2))
        assert max_rare_pre < 0 and loss >= release_q / 2 - 1e-12
        continuations.append(dict(step=step, time=50., loss=loss, proven_floor=release_q / 2,
                                  maximum_rare_preactivation=max_rare_pre))

    result = dict(
        status="Mathematical and numerical consistency audit; not a fit or general validation.",
        seed=20261029, gaussian_draws=draws, probability_checks=rows,
        trap=dict(initial_jacobian_rank=3, width=4, common_exposure=exposure,
                  common_branch_elapsed=float(elapsed), common_branch_residual=float(eu),
                  natural_full_task_q=q, natural_full_task_checkpoints=natural,
                  release_q=release_q, continuations=continuations),
    )
    dest = Path(__file__).resolve().parents[2] / "docs/evidence/probabilistic_competition/audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
