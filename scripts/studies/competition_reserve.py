"""Capability retained after a declared amount of coherent common learning.

The state variable is common prediction gain, not elapsed training steps.
Single-prototype common transport is exact, including common-gate absorption.
"""

from __future__ import annotations

import numpy as np


def transport(w, a, exposure):
    """Flow under one unit-norm common input (the first coordinate).

    Use the absorbing zero-subgradient solution at a common-gate hit.
    Other coordinates stay fixed. Arrays can have any leading batch shape.
    """
    if exposure < 0:
        raise ValueError("Exposure must be nonnegative.")
    w, a = np.asarray(w, dtype=float), np.asarray(a, dtype=float)
    out_w, out_a = w.copy(), a.copy()
    active = w[..., 0] > 0
    along = w[..., 0] * np.cosh(exposure) + a * np.sinh(exposure)
    readout = a * np.cosh(exposure) + w[..., 0] * np.sinh(exposure)
    hit = active & (along <= 0)
    out_w[..., 0] = np.where(active, np.maximum(along, 0), w[..., 0])
    out_a[...] = np.where(active, readout, a)
    out_a[...] = np.where(hit, -np.sqrt(np.maximum(a * a - w[..., 0] ** 2, 0)), out_a)
    return out_w, out_a




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
    """Exact closing exposures in the monotone strictly positive-readout sector.

    Rows are units, columns rare contrasts. Common-inactive units have infinite
    lifetime if they currently respond. Nonpositive readouts are outside this case.
    """
    w, a, _ = state
    if np.any(a <= 0):
        raise ValueError("Monotone lifetime formula requires strictly positive readouts.")
    target = np.tan(theta) * np.abs(w[:, 1:] @ directions.T)
    along = w[:, 0, None]
    readout = a[:, None]
    # Solve (W+A) exp(2s) - 2 target exp(s) + (W-A) = 0.
    radicand = target**2 + readout**2 - along**2
    root = (target + np.sqrt(np.maximum(radicand, 0))) / np.maximum(along + readout, 1e-300)
    lifetime = np.log(np.maximum(root, 1))
    lifetime = np.where(target <= along, 0, lifetime)
    protected_response = (along < 0) | ((along == 0) & (target > 0))
    return np.where(protected_response, np.inf, lifetime)


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
        def metric_norm(v):
            return np.sqrt((np.sum(v[0]**2) + np.sum(v[1]**2)) / width + v[2]**2)
        length_common += fraction * rate * metric_norm(common)
        length_rest += fraction * rate * metric_norm(tuple(f - c for f, c in zip(full, common)))
        elapsed += fraction * rate
        result = tuple(s + fraction * rate * v for s, v in zip(result, full))
        if head_prediction(result) >= target - 1e-10:
            return result, dict(completed=True, steps=step + 1, elapsed=elapsed,
                                integrated_other_to_common=length_rest / max(length_common, 1e-30))
    return result, dict(completed=False, steps=max_steps, elapsed=elapsed,
                        integrated_other_to_common=length_rest / max(length_common, 1e-30))


def vector_transport(w, a, u, exposure):
    """Exact hyperbolic transport along an arbitrary unit vector u in input space.

    Decomposes incoming weights into parallel and orthogonal components relative to u.
    Parallel components and readouts follow the exact 2D hyperbolic flow with
    ReLU gate absorption when along <= 0.
    Orthogonal components remain strictly invariant.
    """
    if exposure < 0:
        raise ValueError("Exposure must be nonnegative.")
    u = np.asarray(u, dtype=float)
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        raise ValueError("Direction u must have positive norm.")
    u = u / norm
    w, a = np.asarray(w, dtype=float), np.asarray(a, dtype=float)
    w_along = w @ u
    w_perp = w - w_along[..., None] * u
    active = w_along > 0
    along = w_along * np.cosh(exposure) + a * np.sinh(exposure)
    readout = a * np.cosh(exposure) + w_along * np.sinh(exposure)
    hit = active & (along <= 0)
    new_along = np.where(active, np.maximum(along, 0), w_along)
    new_a = np.where(active, readout, a)
    new_a = np.where(hit, -np.sqrt(np.maximum(a**2 - w_along**2, 0)), new_a)
    new_w = w_perp + new_along[..., None] * u
    return new_w, new_a


def vector_flow(state, u, exposure, norm_x=1.0):
    """Full network state flow under dominant pressure along unit vector u."""
    w, a = vector_transport(state[0], state[1], u, exposure)
    return w, a, state[2] + exposure / norm_x


def directional_prediction(state, u, norm_x=1.0):
    """Network output along direction norm_x * u."""
    w, a, c = state
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    return float(c + norm_x * np.mean(a * np.maximum(w @ u, 0)))


def vector_exposure_for_gain(state, u, gain, norm_x=1.0):
    """Invert the monotone response along direction u for a given gain."""
    if gain < 0:
        raise ValueError("gain must be nonnegative.")
    if gain == 0:
        return 0.0
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    target = directional_prediction(state, u, norm_x) + gain
    low, high = 0.0, max(gain * norm_x, 1.0)
    while directional_prediction(vector_flow(state, u, high, norm_x), u, norm_x) < target:
        high *= 2.0
    for _ in range(60):
        mid = (low + high) / 2
        if directional_prediction(vector_flow(state, u, mid, norm_x), u, norm_x) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def detect_dominant_episode(state, x, y, p=None):
    """Detect dominant pressure direction, coherence, and scale from gradient covariance/SVD.

    Uses the singular value decomposition of the incoming layer gradient matrix G_W
    (the uncentered parameter gradient covariance across neurons).

    Returns:
        direction (np.ndarray): Unit vector u in input space along the dominant gradient axis.
        coherence (float): Ratio of dominant eigenvalue energy sigma_1^2 / sum(sigma_i^2) in [0, 1].
        strength (float): Dominant singular value sigma_1.
        scale (float): Characteristic input projection scale ell along u.
    """
    w, a, c = state
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p is None:
        p = np.full(len(x), 1.0 / len(x))
    else:
        p = np.asarray(p, dtype=float)

    pre = w @ x.T
    f = c + np.mean(a[:, None] * np.maximum(pre, 0), axis=0)
    e = y - f
    pe = p * e

    # Incoming weight gradient matrix G_W of shape (width, input_dim)
    # G_W[j] = a[j] * sum_k p_k * e_k * 1{w_j^T x_k > 0} * x_k
    gw = a[:, None] * (((pre > 0) * pe) @ x)

    # SVD of G_W: dominant right singular vector is the input-space direction
    # that maximizes total squared update across neurons in the layer
    _, s, vt = np.linalg.svd(gw, full_matrices=False)
    u = vt[0]

    # Coherence ratio: fraction of gradient variance along the dominant axis
    total_energy = np.sum(s**2)
    coherence = float(s[0]**2 / max(total_energy, 1e-30))

    # Orient u so that positive residual pressure corresponds to u^T x > 0
    # i.e., sum_k p_k * e_k * (x_k^T u) > 0
    pressure_projection = float(np.sum(pe * (x @ u)))
    if pressure_projection < 0:
        u = -u

    # Characteristic scale ell: root-mean-square projection of inputs weighted by squared residual
    scale = float(np.sqrt(np.sum(pe**2 * ((x @ u)**2)) / max(np.sum(pe**2), 1e-30)))

    return u, coherence, float(s[0]), max(scale, 1e-12)


def online_reserve(state, audit, u, gain, norm_x=1.0, tau=20.):
    """Predict capability reserve along detected direction u for a given gain."""
    exposure = vector_exposure_for_gain(state, u, gain, norm_x)
    flow_state = vector_flow(state, u, exposure, norm_x)
    return capability(flow_state, audit, tau), exposure
