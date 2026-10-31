"""Capability retained after a declared amount of coherent common learning.

The state variable is common prediction gain, not elapsed training steps.
Single-prototype common transport is exact, including common-gate absorption.
"""

from __future__ import annotations

import numpy as np
from probabilistic_option_transport import transport


def forward(state, x):
    w, a, c = state
    return c + np.mean(a[:, None] * np.maximum(w @ x.T, 0), axis=0)


def velocity(state, data, common_only=False):
    w, a, _ = state
    x, p, y = data["x"], data["p"].copy(), data["y"]
    if common_only:
        p[~data["head"]] = 0
    pre = w @ x.T
    e = y - forward(state, x)
    pe = p * e
    return (a[:, None] * (((pre > 0) * pe) @ x), np.maximum(pre, 0) @ pe, pe.sum())


def common_flow(state, exposure):
    w, a = transport(state[0], state[1], exposure)
    return w, a, state[2] + exposure


def head_prediction(state):
    return float(state[2] + np.mean(state[1] * np.maximum(state[0][:, 0], 0)))


def exposure_for_gain(state, gain):
    """Invert the monotone common response; no future time integration."""
    if gain < 0:
        raise ValueError("This branch concerns positive common correction.")
    if gain == 0:
        return 0.
    target = head_prediction(state) + gain
    # The intercept alone contributes exposure, so gain brackets the root.
    low, high = 0., gain
    for _ in range(60):
        middle = (low + high) / 2
        if head_prediction(common_flow(state, middle)) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def capability(state, audit, tau=20., diversity=False):
    w, a, _ = state
    x, p, directions = audit["x"], audit["p"], audit["directions"]
    root = np.sqrt(p)
    pre = w @ x.T
    h, gate = np.maximum(pre, 0), pre > 0
    ga = a[:, None] * gate
    k = ((h.T @ h + (ga.T @ ga) * (x @ x.T)) / len(a) + 1) * root[:, None] * root
    lam = .5 / tau
    screened = np.linalg.solve(k + lam * np.eye(len(p)), directions)
    beta = np.maximum(1 - lam * np.sum(directions * screened, axis=0), 0)
    if not diversity:
        return beta
    screen = screened * root[:, None]
    readout = h @ screen
    incoming = np.einsum("ji,ir,id->jrd", gate, screen, x)
    weights = (readout**2 + a[:, None]**2 * np.sum(incoming**2, axis=-1)) / len(a)
    effective = weights.sum(0)**2 / np.maximum(np.sum(weights**2, axis=0), 1e-300)
    return beta, effective


def reserve(state, audit, gain):
    exposure = exposure_for_gain(state, gain)
    return capability(common_flow(state, exposure), audit), exposure


def protected_reserve(state, train, audit, tau=20.):
    """Response supplied by units inactive on the entire declared head support.

    Keep the original 1/m normalization, even when some blocks are removed.
    Returning a subnetwork of fewer units with its own denominator would be wrong.
    """
    w, a, _ = state
    maximum = w @ train["head_center"] + np.abs(w) @ train["head_radius"]
    protected = maximum < 0
    altered_w, altered_a = w.copy(), a.copy()
    altered_w[~protected] = 0
    altered_a[~protected] = 0
    return capability((altered_w, altered_a, state[2]), audit, tau), protected


def directional_lifetimes(state, directions, theta):
    """Exact closing exposures in the monotone positive-readout sector.

    Rows are units, columns rare contrasts. Common-inactive units have infinite
    lifetime. A negative readout is outside this monotone special case.
    """
    w, a, _ = state
    if np.any(a < 0):
        raise ValueError("Monotone lifetime formula requires nonnegative readouts.")
    target = np.tan(theta) * np.abs(w[:, 1:] @ directions.T)
    along = w[:, 0, None]
    readout = a[:, None]
    # Solve (W+A) exp(2s) - 2 target exp(s) + (W-A) = 0.
    radicand = target**2 + readout**2 - along**2
    root = (target + np.sqrt(np.maximum(radicand, 0))) / np.maximum(along + readout, 1e-300)
    lifetime = np.log(np.maximum(root, 1))
    lifetime = np.where(target <= along, 0, lifetime)
    return np.where(along <= 0, np.inf, lifetime)


def task(seed, factors=24, total_rare=.04, spread=0., association=.5, target=4.):
    """Long-tailed refinement directions around one frequent context.

    Correlation is in context geometry. A shared linear refinement plus a common
    hinge has an exact three-unit witness, including continuous observations.
    """
    rng = np.random.default_rng(seed)
    nuisance = 4
    d = 1 + factors + 4 + nuisance
    theta = np.pi / 3
    directions = np.zeros((factors, d - 1))
    directions[np.arange(factors), np.arange(factors)] = np.sqrt(1 - association)
    directions[np.arange(factors), factors + np.arange(factors) % 4] = np.sqrt(association)
    q = np.arange(1, factors + 1, dtype=float)**-1.15
    q *= total_rare / q.sum()
    coefficient = (-1.) ** np.arange(factors)
    teacher = np.zeros(d)
    teacher[1:1 + factors] = coefficient / (np.sin(theta) * np.sqrt(1 - association))
    prototypes = np.zeros((1 + 2 * factors, d))
    prototypes[0, 0] = 1
    for r in range(factors):
        for offset, sign in enumerate([1., -1.]):
            prototypes[1 + 2*r + offset, 0] = -np.cos(theta)
            prototypes[1 + 2*r + offset, 1:] = sign * np.sin(theta) * directions[r]
    weights = np.r_[1 - total_rare, np.repeat(q / 2, 2)]

    def make_panel(nodes):
        x = np.repeat(prototypes, nodes, axis=0)
        if spread:
            x *= rng.uniform(1 - spread, 1 + spread, (len(x), 1))
            x[:, -nuisance:] += rng.uniform(-spread, spread, (len(x), nuisance))
            # Small continuous variation within the rare attribute subspace.
            x[:, 1:1 + factors] += spread * .1 / np.sqrt(factors) * rng.uniform(
                -1, 1, (len(x), factors))
        y = target * np.maximum(x[:, 0], 0) + x @ teacher
        return dict(x=x, y=y, p=np.repeat(weights / nodes, nodes),
                    head=np.repeat(np.arange(len(prototypes)) == 0, nodes))

    train = make_panel(8 if spread else 1)
    validation = make_panel(32 if spread else 1)
    center = np.zeros(d)
    center[0] = 1
    radius = np.zeros(d)
    radius[0] = spread
    radius[1:1 + factors] = spread * .1 / np.sqrt(factors)
    radius[-nuisance:] = spread
    train.update(head_center=center, head_radius=radius)
    probes = np.zeros((len(prototypes), factors))
    probes[1 + 2*np.arange(factors), np.arange(factors)] = 1 / np.sqrt(2)
    probes[2 + 2*np.arange(factors), np.arange(factors)] = -1 / np.sqrt(2)
    audit = dict(x=prototypes, p=np.full(len(prototypes), 1 / len(prototypes)), directions=probes)
    w = np.zeros((3, d))
    w[0, 0], w[1], w[2] = 1., teacher, -teacher
    witness = (w, np.array([3 * target, 3., -3.]), 0.)
    for data in [train, validation]:
        np.testing.assert_allclose(forward(witness, data["x"]), data["y"], atol=1e-12)
    return train, validation, audit, dict(q=q, directions=directions, theta=theta,
                                         target=target, capacity=3)


def learn_to_gain(state, train, gain, rate=.01, max_steps=10000, common_only=False):
    """Measure a natural phase ending at functional progress, not a fixed time."""
    result = tuple(np.asarray(v).copy() for v in state)
    target = head_prediction(state) + gain
    length_common, length_rest, elapsed = 0., 0., 0.
    for step in range(max_steps):
        common = velocity(result, train, common_only=True)
        full = common if common_only else velocity(result, train)
        trial = tuple(s + rate * v for s, v in zip(result, full))
        fraction = 1.
        if head_prediction(trial) >= target:
            low, high = 0., 1.
            for _ in range(45):
                middle = (low + high) / 2
                sample = tuple(s + middle * rate * v for s, v in zip(result, full))
                if head_prediction(sample) < target:
                    low = middle
                else:
                    high = middle
            fraction = (low + high) / 2
        width = len(state[1])
        metric_norm = lambda v: np.sqrt((np.sum(v[0]**2) + np.sum(v[1]**2)) / width + v[2]**2)
        length_common += fraction * rate * metric_norm(common)
        length_rest += fraction * rate * metric_norm(tuple(f - c for f, c in zip(full, common)))
        elapsed += fraction * rate
        result = tuple(s + fraction * rate * v for s, v in zip(result, full))
        if head_prediction(result) >= target - 1e-10:
            return result, dict(completed=True, steps=step + 1, elapsed=elapsed,
                                integrated_other_to_common=length_rest / max(length_common, 1e-30))
    return result, dict(completed=False, steps=max_steps, elapsed=elapsed,
                        integrated_other_to_common=length_rest / max(length_common, 1e-30))
