"""Product-distribution pressure decompositions and shallow-network diagnostics.

All functions are observational: no training distribution, gates, or parameters
are frozen. Directions in this module are probes, not spectral option counts.
"""

from __future__ import annotations

import numpy as np


def product_basis(x, q):
    """Return normalized Bernoulli ANOVA functions and their factor incidence.

    Column ``s`` is the product over bits selected by integer mask ``s``.
    On the complete product panel, phi.T @ diag(p) @ phi is the identity.
    """
    x, q = np.asarray(x), np.asarray(q)
    incidence = (np.arange(2 ** len(q))[:, None] >> np.arange(len(q))) & 1
    z = (x - q) / np.sqrt(q * (1 - q))
    phi = np.prod(np.where(incidence[None] != 0, z[:, None], 1.), axis=-1)
    return phi, incidence


def parity_coefficients(q):
    """Exact ANOVA coefficients of the ordinary {0,1} parity target."""
    q = np.asarray(q)
    incidence = (np.arange(2 ** len(q))[:, None] >> np.arange(len(q))) & 1
    factors = np.where(incidence, -2 * np.sqrt(q * (1 - q)), 1 - 2 * q)
    coefficients = -.5 * np.prod(factors, axis=1)
    coefficients[0] += .5
    return coefficients


def factor_allocation(incidence):
    """Mean plus symmetric allocation of each interaction to its participants.

    Rows sum to one. The resulting factor pressures add exactly but need not be
    orthogonal. Equal interaction shares are an explicit attribution convention.
    """
    order = incidence.sum(1)
    allocation = np.zeros((len(order), incidence.shape[1] + 1))
    allocation[0, 0] = 1
    allocation[:, 1:] = incidence / np.maximum(order[:, None], 1)
    return allocation


def inputs(x, q, encoding="raw"):
    if encoding not in {"raw", "centered"}:
        raise ValueError("encoding must be raw or centered")
    values = np.asarray(x) - (q if encoding == "centered" else 0)
    return np.column_stack([values, np.ones(len(values))])


def forward(state, x):
    w, a, c = state
    pre = w @ np.swapaxes(x, -1, -2)
    return c[..., None] + np.mean(a[..., None] * np.maximum(pre, 0), axis=-2)


def velocity(state, x, residual, weights):
    """Negative gradient with M_w=M_a=m, M_c=1 and f=c+mean(a*ReLU)."""
    w, a, _ = state
    pre = w @ np.swapaxes(x, -1, -2)
    pe = residual * weights
    return (
        a[..., None] * (((pre > 0) * pe[..., None, :]) @ x),
        np.sum(np.maximum(pre, 0) * pe[..., None, :], axis=-1),
        np.sum(pe, axis=-1),
    )


def advance(state, update, rate):
    return tuple(s + rate * v for s, v in zip(state, update))


def derivative(state, x, weights):
    """Weighted Jacobian in optimizer coordinates, for one unbatched state."""
    w, a, _ = state
    pre = w @ x.T
    h = np.maximum(pre, 0)
    incoming = ((pre > 0)[:, :, None] * a[:, None, None] * x[None]).transpose(1, 0, 2)
    d = np.column_stack([h.T, incoming.reshape(len(x), -1)]) / np.sqrt(len(a))
    return np.sqrt(weights)[:, None] * np.column_stack([d, np.ones(len(x))])


def residual_pressures(residual, phi, weights, allocation):
    coefficients = (residual * weights) @ phi
    components = (phi @ (coefficients[:, None] * allocation)).T
    return coefficients, components


def pressure_ledger(state, x, y, weights, phi, allocation):
    """Exact instantaneous loss-rate and interference identities (not forecasts)."""
    residual = y - forward(state, x)
    coefficients = (weights * residual) @ phi
    d = derivative(state, x, weights)
    modes = (phi.T * np.sqrt(weights)) @ d
    implementations = coefficients[:, None] * modes
    total = implementations.sum(0)
    modal_self = np.sum(implementations**2)
    actual = total @ total
    sources = allocation.T @ implementations
    gram = sources @ sources.T
    source_self = np.trace(gram)
    return dict(
        loss=.5 * np.sum(coefficients**2),
        rate=actual,
        modal_self=modal_self,
        modal_cancellation=1 - actual / max(modal_self, 1e-300),
        factor_cancellation=1 - actual / max(source_self, 1e-300),
        coefficients=coefficients,
        modal_rate=coefficients * (modes @ total),
        source_gram=gram,
        derivative=d,
    )


def response_transport(state, x, train_weights, audit_weights, probes, pressures, tau):
    """Smooth directional response transport for declared pressure components.

    Derivatives are valid inside gate cells. Gate-jump and finite-step errors
    must be audited separately. The supplied pressures are never used to train.
    """
    w, a, _ = state
    d = derivative(state, x, audit_weights)
    kernel = d @ d.T
    lam = .5 / tau
    inverse_probes = np.linalg.solve(kernel + lam * np.eye(len(x)), probes)
    strength = np.sum(probes**2, axis=0) - lam * np.sum(probes * inverse_probes, axis=0)
    pre = w @ x.T
    gate = pre > 0
    h = np.maximum(pre, 0)
    weighted = np.sqrt(audit_weights)[:, None] * inverse_probes
    hv = h @ weighted
    incoming = np.einsum("jn,nr,nd->jrd", gate, weighted, x)
    grad_w = (2 * lam / len(a)) * hv[:, :, None] * incoming
    grad_a = (2 * lam / len(a)) * a[:, None] * np.sum(incoming**2, axis=-1)
    dw, da, _ = velocity(state, x, pressures, train_weights)
    transport = np.einsum("jrd,gjd->gr", grad_w, dw) + da @ grad_a
    return strength, transport


def response_strength(state, x, weights, probes, tau):
    d = derivative(state, x, weights)
    lam = .5 / tau
    inverse = np.linalg.solve(d @ d.T + lam * np.eye(len(x)), probes)
    return np.sum(probes**2, axis=0) - lam * np.sum(probes * inverse, axis=0)


def kernel_transport(state, x, weights, probes, pressures):
    """Signed instantaneous kernel transport, without a diagnostic horizon."""
    w, a, _ = state
    pre = w @ x.T
    gate = pre > 0
    weighted = np.sqrt(weights)[:, None] * probes
    hv = np.maximum(pre, 0) @ weighted
    incoming = np.einsum("jn,nr,nd->jrd", gate, weighted, x)
    grad_w = 2 / len(a) * hv[:, :, None] * incoming
    grad_a = 2 / len(a) * a[:, None] * np.sum(incoming**2, axis=-1)
    dw, da, _ = velocity(state, x, pressures, weights)
    signed = np.einsum("jrd,gjd->gr", grad_w, dw) + da @ grad_a
    d = derivative(state, x, weights)
    strength = np.sum((d.T @ probes)**2, axis=0)
    return strength, signed


def update_power(update):
    dw, da, dc = update
    return (np.sum(dw**2, axis=(-2, -1)) + np.sum(da**2, axis=-1)) / da.shape[-1] + dc**2


def rate_ledger(state, next_state, x, y, weights):
    """Exact endpoint split of effective loss rate into geometry and demand.

    This split includes every gate change and is valid between arbitrary actual
    checkpoints. It does not attribute the geometry change to pressure sources.
    """
    e0, e1 = y - forward(state, x), y - forward(next_state, x)
    norm0, norm1 = np.sum(weights * e0**2), np.sum(weights * e1**2)
    k00 = update_power(velocity(state, x, e0, weights)) / norm0
    k01 = update_power(velocity(next_state, x, e0, weights)) / norm0
    k10 = update_power(velocity(state, x, e1, weights)) / norm1
    k11 = update_power(velocity(next_state, x, e1, weights)) / norm1
    return dict(before=k00, after=k11,
                transport=.5 * (k01 - k00 + k11 - k10),
                rotation=.5 * (k10 - k00 + k11 - k01))


def step_ledger(state, next_state, x, y, weights, update, rate):
    """Exact finite-update loss ledger, including the nonlinear prediction term."""
    residual = y - forward(state, x)
    change = forward(next_state, x) - forward(state, x)
    w, a, _ = state
    dw, da, dc = update
    pre = w @ x.T
    tangent = dc + np.mean(
        da[:, None] * np.maximum(pre, 0) + a[:, None] * (pre > 0) * (dw @ x.T), axis=0
    )
    linear = rate * np.sum(weights * residual * tangent)
    nonlinear = np.sum(weights * residual * (change - rate * tangent))
    dz = rate * (dw @ x.T)
    gate_remainder = np.maximum(pre + dz, 0) - np.maximum(pre, 0) - (pre > 0) * dz
    bilinear_change = np.mean(rate * da[:, None] * (pre > 0) * dz, axis=0)
    gate_change = np.mean((a + rate * da)[:, None] * gate_remainder, axis=0)
    quadratic = -.5 * np.sum(weights * change**2)
    actual = np.sum(weights * (residual * change - .5 * change**2))
    return dict(actual=actual, linear=linear, nonlinear=nonlinear, quadratic=quadratic,
                bilinear=np.sum(weights * residual * bilinear_change),
                gates=np.sum(weights * residual * gate_change))


def kernel_change_ledger(state, next_state, x, weights, probes):
    """Exact finite kernel change: activation features, amplitudes, and gates."""
    w0, a0, _ = state
    w1, a1, _ = next_state
    z0, z1 = w0 @ x.T, w1 @ x.T
    weighted = np.sqrt(weights)[:, None] * probes
    h0, h1 = np.maximum(z0, 0) @ weighted, np.maximum(z1, 0) @ weighted
    b0 = np.einsum("jn,nr,nd->jrd", z0 > 0, weighted, x)
    b1 = np.einsum("jn,nr,nd->jrd", z1 > 0, weighted, x)
    n0, n1 = np.sum(b0**2, axis=-1), np.sum(b1**2, axis=-1)
    features = np.mean(h1**2 - h0**2, axis=0)
    amplitudes = np.mean(.5 * (a1**2 - a0**2)[:, None] * (n0 + n1), axis=0)
    gates = np.mean(.5 * (a1**2 + a0**2)[:, None] * (n1 - n0), axis=0)
    return dict(features=features, amplitudes=amplitudes, gates=gates,
                total=features + amplitudes + gates)


def finite_pressure_transport(state, next_state, x, weights, probes, displacements):
    """Exact line-path attribution of a finite kernel change to update sources.

    ``displacements`` sum to the actual parameter displacement. Smooth pieces
    are integrated exactly; each gate jump is allocated by the sources' signed
    preactivation displacements. Simultaneous jumps use a symmetric midpoint
    convention. This is an attribution of one actual update, not a unique causal
    mediation or a proposal to train on isolated pressures.
    """
    w, a, _ = state
    w1, a1, _ = next_state
    source_w, source_a, _ = displacements
    np.testing.assert_allclose(source_w.sum(0), w1 - w, atol=1e-12, rtol=1e-9)
    np.testing.assert_allclose(source_a.sum(0), a1 - a, atol=1e-12, rtol=1e-9)
    weighted = np.sqrt(weights)[:, None] * probes
    smooth = np.zeros((len(source_a), probes.shape[1]))
    jumps = np.zeros_like(smooth)
    for j in range(len(a)):
        z0 = x @ w[j]
        dz = x @ (w1[j] - w[j])
        da = a1[j] - a[j]
        directions = source_w[:, j] @ x.T
        changed = np.flatnonzero((z0 > 0) != (z0 + dz > 0))
        times = -z0[changed] / dz[changed]
        order = np.argsort(times)
        changed, times = changed[order], times[order]
        edges = np.unique(np.r_[0., np.clip(times, 0, 1), 1.])
        length = np.diff(edges)
        middle = .5 * (edges[:-1] + edges[1:])
        gates = z0[None] + middle[:, None] * dz > 0
        h = (gates * (z0[None] + middle[:, None] * dz)) @ weighted
        incoming = np.einsum("tn,nr,nd->trd", gates, weighted, x)
        norm = np.sum(incoming**2, axis=-1)
        source_h = np.einsum("tn,gn,nr->tgr", gates, directions, weighted)
        smooth += 2 / len(a) * (
            np.einsum("t,tr,tgr->gr", length, h, source_h)
            + source_a[:, j, None] * np.sum(
                length[:, None] * (a[j] + middle * da)[:, None] * norm, axis=0)
        )
        if len(changed):
            sign = np.where(dz[changed] > 0, 1., -1.)
            delta_b = sign[:, None, None] * weighted[changed, :, None] * x[changed, None]
            _, groups = np.unique(times, return_inverse=True)
            group_b = np.zeros((groups[-1] + 1, probes.shape[1], x.shape[1]))
            np.add.at(group_b, groups, delta_b)
            initial_b = np.einsum("n,nr,nd->rd", z0 > 0, weighted, x)
            midpoint = initial_b + np.cumsum(group_b, axis=0) - .5 * group_b
            norm_change = 2 * np.sum(midpoint[groups] * delta_b, axis=-1)
            event_change = (a[j] + times * da)[:, None]**2 / len(a) * norm_change
            shares = directions[:, changed] / dz[changed]
            jumps += shares @ event_change
    return dict(smooth=smooth, gates=jumps, total=smooth + jumps)


def gaussian_weight_factor(weights, batch):
    """Square root of the multinomial minibatch *weight covariance*."""
    root = np.sqrt(weights / batch)
    return np.diag(root) - weights[:, None] * root[None, :]


def gaussian_forecast_step(state, x, y, weights, rate, batch, rng):
    """Forecast only: moment-matched Gaussian SGD, refreshing every real gate.

    Gaussian weights are a gradient-noise approximation, not a training-data
    distribution. Actual experiments must use ordinary sampled minibatches.
    """
    residual = y - forward(state, x)
    noise = rng.normal(size=residual.shape) * np.sqrt(weights / batch)
    random_weights = weights + noise - weights * noise.sum(axis=-1, keepdims=True)
    return advance(state, velocity(state, x, residual, random_weights), rate)


def local_loss_forecasts(state, x, y, weights, rate, batch, horizons):
    """Second-order discrete-SGD forecasts, including optimizer-noise curvature.

    These are local Taylor approximations, not changes to training. The full
    curvature retains coefficient/feature coupling; the GN baseline omits it.
    Gate crossings and longer-horizon remainder must be measured empirically.
    """
    w, a, _ = state
    residual = y - forward(state, x)
    root_e = np.sqrt(weights) * residual
    d = derivative(state, x, weights)
    kernel = d @ d.T
    kv = kernel @ root_e
    gradient_power = root_e @ kv
    gn_curvature = kv @ kv
    pre = w @ x.T
    gate, h = pre > 0, np.maximum(pre, 0)
    b = (gate * (weights * residual)) @ x
    dw, da, _ = velocity(state, x, residual, weights)
    coupling = 2 / len(a) * np.sum(da * np.sum(b * dw, axis=1))
    full_curvature = gn_curvature - coupling
    example_gn = np.sum(kernel**2 * residual[None]**2)
    example_coupling = 2 / len(a) * np.sum(
        a * np.sum(((h * (weights * residual**2)) @ x) * b, axis=1)
    )
    noise_gn = example_gn - gn_curvature
    noise_full = example_gn - example_coupling - full_curvature
    loss = .5 * np.sum(root_e**2)
    rows = {}
    for horizon in horizons:
        coefficient = rate**2 * horizon * (2 * horizon - 1) / 2
        noise_coefficient = rate**2 * horizon / (2 * batch)
        rows[int(horizon)] = dict(
            persistence=loss,
            first_order=loss - rate * horizon * gradient_power,
            gauss_newton=loss - rate * horizon * gradient_power
            + coefficient * gn_curvature + noise_coefficient * noise_gn,
            transported=loss - rate * horizon * gradient_power
            + coefficient * full_curvature + noise_coefficient * noise_full,
        )
    return dict(loss=loss, rate=gradient_power, gn_curvature=gn_curvature,
                full_curvature=full_curvature, coupling=coupling,
                noise_gn=noise_gn, noise_full=noise_full, forecasts=rows)
