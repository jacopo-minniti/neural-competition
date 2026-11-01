"""Many associated refinements, exact capacity witnesses, and conditional forecasts."""

from __future__ import annotations

import itertools

import numpy as np


def specification(factors=24, alpha=1.15, richer=False, association=0.65):
    rank = np.arange(1, factors + 1)
    frequency = rank.astype(float) ** -alpha
    frequency *= (1.2 if richer else 0.4) / frequency.sum()
    angle = np.linspace(-0.8, 0.8, factors)
    attributes = np.column_stack([np.cos(angle), np.sin(angle)])
    return dict(factors=factors, alpha=alpha, richer=richer, association=association,
                q=frequency, attributes=attributes, coefficient=(-1.) ** rank,
                capacity=2 + 2 * factors, nuisance=4 if richer else 0)


def latent_draw(spec, count, rng):
    r = spec["factors"]
    if spec["richer"]:
        # Shared uniforms within groups preserve every marginal exactly.
        uniforms = rng.random((count, r))
        for group in range(4):
            ids = np.arange(group, r, 4)
            shared = rng.random((count, 1))
            use_shared = rng.random((count, 1)) < spec["association"]
            uniforms[:, ids] = np.where(use_shared, shared, uniforms[:, ids])
        indicator = uniforms < spec["q"]
        z = rng.uniform(-1, 1, (count, 2))
    else:
        context = rng.choice(r + 1, count, p=np.r_[1 - spec["q"].sum(), spec["q"]])
        indicator = context[:, None] == np.arange(1, r + 1)
        z = rng.choice([-1., 1.], size=(count, 2))
    return indicator.astype(float), z


def encode(spec, indicator, z, rng, rotation):
    count = len(z)
    context = 3 * (2 * indicator - 1)
    if spec["richer"]:
        context += rng.uniform(-.25, .25, context.shape)
    nuisance = rng.uniform(-1, 1, (count, spec["nuisance"]))
    raw = np.column_stack([context, z, nuisance, np.ones(count)])
    x = raw @ rotation
    y = 2 * z[:, 0] + np.sum(
        indicator * (z @ spec["attributes"].T) * spec["coefficient"], axis=1)
    return dict(x=x, y=y, indicator=indicator, z=z, p=np.full(count, 1 / count))


def dataset(spec, seed, train_count=1024, validation_count=4096):
    rng = np.random.default_rng(seed)
    dimension = spec["factors"] + 3 + spec["nuisance"]
    rotation = np.eye(dimension)
    if spec["richer"]:
        # Mix observed coordinates, but preserve the explicit affine intercept input.
        rotation[:-1, :-1] = np.linalg.qr(rng.normal(size=(dimension - 1, dimension - 1)))[0]
    if not spec["richer"]:
        context = np.repeat(np.arange(spec["factors"] + 1), 4)
        z = np.tile(np.array(list(itertools.product([-1., 1.], repeat=2))),
                    (spec["factors"] + 1, 1))
        indicator = (context[:, None] == np.arange(1, spec["factors"] + 1)).astype(float)
        train = encode(spec, indicator, z, rng, rotation)
        train["p"] = np.repeat(np.r_[1 - spec["q"].sum(), spec["q"]] / 4, 4)
        validation = train
    else:
        train = encode(spec, *latent_draw(spec, train_count, rng), rng, rotation)
        validation = encode(spec, *latent_draw(spec, validation_count, rng), rng, rotation)

    # A fixed diagnostic panel, balanced over interventions on all 24 factors.
    # Its uniform measure is intentionally not called population risk.
    indicators, attributes = [], []
    for j in range(spec["factors"]):
        base, _ = latent_draw(spec, 1, rng)
        for sign in [-1., 1.]:
            for value in [0., 1.]:
                row = base[0].copy()
                row[j] = value
                indicators.append(row)
                attributes.append(sign * spec["attributes"][j])
    indicator, z = np.array(indicators), np.array(attributes)
    audit = encode(spec, indicator, z, rng, rotation)
    columns = indicator * (z @ spec["attributes"].T)
    common = np.column_stack([np.ones(len(z)), z])
    columns -= common @ np.linalg.lstsq(common, columns, rcond=None)[0]
    columns /= np.linalg.norm(columns, axis=0, keepdims=True)
    audit["directions"] = columns
    audit["root"] = np.sqrt(audit["p"])

    capacity = spec["capacity"]
    w = np.zeros((capacity, dimension))
    a = np.zeros(capacity)
    r = spec["factors"]
    w[0, r], w[1, r] = 1, -1
    a[:2] = capacity * np.array([2., -2.])
    for j in range(r):
        w[2 + 2 * j:4 + 2 * j, j] = 1
        w[2 + 2 * j, r:r + 2] = spec["attributes"][j]
        w[3 + 2 * j, r:r + 2] = -spec["attributes"][j]
        a[2 + 2 * j:4 + 2 * j] = capacity * spec["coefficient"][j] / 2 * np.array([1, -1])
    witness = (w @ rotation, a, np.array(0.))
    for panel in [train, validation, audit]:
        np.testing.assert_allclose(forward(witness, panel["x"]), panel["y"], atol=1e-12)
    return train, validation, audit, witness


def initialize(width, dimension, kind, seed):
    rng = np.random.default_rng(seed)
    # Draw the same max-width bank for paired widths.
    shape = (200, dimension)
    if kind == "small_gaussian":
        w, a = .002 * rng.normal(size=shape), .002 * rng.normal(size=200)
    elif kind == "fanin_gaussian":
        w, a = rng.normal(size=shape) / np.sqrt(dimension), .5 * rng.normal(size=200)
    elif kind == "fanin_uniform":
        w = rng.uniform(-np.sqrt(3 / dimension), np.sqrt(3 / dimension), shape)
        a = rng.uniform(-np.sqrt(3) / 2, np.sqrt(3) / 2, 200)
    else:
        raise ValueError(kind)
    return w[:width], a[:width], np.array(0.)


def forward(state, x):
    w, a, c = state
    h = np.maximum(w @ np.swapaxes(x, -1, -2), 0)
    return c[..., None] + np.mean(a[..., None] * h, axis=-2)


def velocity(state, x, y, weights):
    w, a, _ = state
    pre = w @ np.swapaxes(x, -1, -2)
    h = np.maximum(pre, 0)
    e = y - forward(state, x)
    pe = weights * e
    return (a[..., None] * (((pre > 0) * pe[..., None, :]) @ x),
            np.sum(h * pe[..., None, :], axis=-1), np.sum(pe, axis=-1))


def response(state, audit, tau=5.):
    w, a, _ = state
    x, root = audit["x"], audit["root"]
    pre = w @ x.T
    h = np.maximum(pre, 0)
    ga = a[..., None] * (pre > 0)
    k = (np.swapaxes(h, -1, -2) @ h
         + (np.swapaxes(ga, -1, -2) @ ga) * (x @ x.T)) / a.shape[-1] + 1
    k *= root[:, None] * root[None, :]
    lam = .5 / tau
    inverse_d = np.linalg.solve(k + lam * np.eye(len(root)),
                                np.broadcast_to(audit["directions"],
                                                k.shape[:-1] + (audit["directions"].shape[1],)))
    strength = 1 - lam * np.sum(audit["directions"] * inverse_d, axis=-2)
    return np.maximum(strength, 0)


def frozen_forecast(state, train, steps, rate, batch, samples, rng):
    """Exact multinomial pushforward for the frozen-checkpoint force model."""
    count = rng.multinomial(steps * batch, train["p"], size=samples)
    v = velocity(state, train["x"], train["y"], count / (steps * batch))
    return tuple(s + rate * steps * dv for s, dv in zip(state, v))


def block_forecast(state, train, steps, rate, batch, samples, rng, block_size=4):
    """Refresh the checkpoint force after each declared exposure block."""
    result = tuple(np.broadcast_to(s, (samples,) + s.shape).copy() for s in state)
    for start in range(0, steps, block_size):
        block = min(block_size, steps - start)
        counts = rng.multinomial(block * batch, train["p"], size=samples)
        v = velocity(result, train["x"], train["y"], counts / (block * batch))
        result = tuple(s + rate * block * dv for s, dv in zip(result, v))
    return result


def influence(state, train, audit, tau=5.):
    """Smooth response susceptibility to each example and screened route diversity.

    Exact derivative within the current gate cell. Gate jumps are excluded here
    but retained by the physical response in the exposure pushforward forecast.
    """
    w, a, _ = state
    x, root, directions = audit["x"], audit["root"], audit["directions"]
    pre = w @ x.T
    h, gate = np.maximum(pre, 0), pre > 0
    ga = a[:, None] * gate
    k = ((h.T @ h + (ga.T @ ga) * (x @ x.T)) / len(a) + 1) * root[:, None] * root
    lam = .5 / tau
    v = np.linalg.solve(k + lam * np.eye(len(root)), directions) * root[:, None]
    hv = h @ v
    incoming = np.einsum("ji,ir,id->jrd", gate, v, x)
    norm = np.sum(incoming**2, axis=-1)
    route = (hv**2 + a[:, None] ** 2 * norm) / len(a)
    diversity = route.sum(0)**2 / np.maximum(np.sum(route**2, axis=0), 1e-300)
    train_pre = w @ train["x"].T
    train_h, train_g = np.maximum(train_pre, 0), train_pre > 0
    e = train["y"] - forward(state, train["x"])
    per_example = 2 * lam / len(a) * e[None] * (
        np.einsum("jr,jl,jrd,ld->rl", hv * a[:, None], train_g, incoming, train["x"])
        + (a[:, None] * norm).T @ train_h)
    mean = per_example @ train["p"]
    covariance = (per_example * train["p"]) @ per_example.T - np.outer(mean, mean)
    return dict(mean=mean, covariance=covariance, diversity=diversity,
                per_example=per_example)


def sgd_step(state, train, rate, batch, rng):
    ids = rng.choice(len(train["p"]), size=state[1].shape[:-1] + (batch,), p=train["p"])
    v = velocity(state, train["x"][ids], train["y"][ids], 1 / batch)
    return tuple(s + rate * dv for s, dv in zip(state, v))
