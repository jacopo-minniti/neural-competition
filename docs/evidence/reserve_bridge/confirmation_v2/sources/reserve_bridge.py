"""One controlled relaxation of the successful reserve task: factor co-occurrence."""

from __future__ import annotations

import itertools

import numpy as np


def task(factors=4, mass=.04, cooccurrence=1., spread=0., seed=0, gamma=1.5):
    """Keep factor marginals, observation geometry, and global teacher fixed.

    Mix the original mutually exclusive presence law with independent presence.
    Conditional on presence, factor signs are independent fair coins. The
    population is enumerable; continuous observations use antithetic views.
    """
    if not 0 < mass < 1 or not 0 <= cooccurrence <= 1 or not 0 <= spread < .5:
        raise ValueError("invalid bridge parameters")
    z = np.array(list(itertools.product([-1., 0., 1.], repeat=factors)))
    present = z != 0
    count = present.sum(1)
    q = np.arange(1, factors + 1, dtype=float)**-1.15
    q *= mass / q.sum()
    independent = np.prod(np.where(present, q / 2, 1 - q), axis=1)
    exclusive = np.zeros(len(z))
    exclusive[count == 0] = 1 - mass
    for i in range(factors):
        exclusive[(count == 1) & present[:, i]] = q[i] / 2
    p = (1 - cooccurrence) * exclusive + cooccurrence * independent
    nuisance = 4
    dimension = 1 + factors + 4 + nuisance
    direction = np.zeros((factors, dimension - 1))
    direction[np.arange(factors), np.arange(factors)] = np.sqrt(.5)
    direction[np.arange(factors), factors + np.arange(factors) % 4] = np.sqrt(.5)
    x = np.column_stack([1 - gamma * count, np.sin(np.pi / 3) * z @ direction])
    coefficient = (-1.)**np.arange(factors)
    teacher = np.zeros(dimension)
    teacher[1:1 + factors] = coefficient / (np.sin(np.pi / 3) * np.sqrt(.5))
    prototypes = np.zeros((1 + 2 * factors, dimension))
    prototypes[0, 0] = 1
    for i in range(factors):
        prototypes[1 + 2 * i, 0] = prototypes[2 + 2 * i, 0] = 1 - gamma
        prototypes[1 + 2 * i, 1:] = np.sin(np.pi / 3) * direction[i]
        prototypes[2 + 2 * i, 1:] = -np.sin(np.pi / 3) * direction[i]
    contrast = np.zeros((len(prototypes), factors))
    contrast[1 + 2 * np.arange(factors), np.arange(factors)] = 1 / np.sqrt(2)
    contrast[2 + 2 * np.arange(factors), np.arange(factors)] = -1 / np.sqrt(2)
    audit = dict(x=prototypes, p=np.full(len(prototypes), 1 / len(prototypes)),
                 directions=contrast)
    radius = np.zeros(dimension)
    radius[0] = spread
    radius[1:1 + factors] = spread * .1 / np.sqrt(factors)
    radius[-nuisance:] = spread
    if spread:
        rng = np.random.default_rng(seed)
        scale = rng.uniform(-spread, spread, (len(x), 2, 1))
        noise = np.zeros((len(x), 2, dimension))
        noise[:, :, 1:1 + factors] = rng.uniform(-1, 1, (len(x), 2, factors)) * radius[1:1 + factors]
        noise[:, :, -nuisance:] = rng.uniform(-spread, spread, (len(x), 2, nuisance))
        plus = x[:, None] * (1 + scale) + noise
        minus = x[:, None] * (1 - scale) - noise
        x = np.stack([plus, minus], axis=2).reshape(-1, dimension)
        z = np.repeat(z, 4, axis=0)
        p = np.repeat(p / 4, 4)
    y = 4 * np.maximum(x[:, 0], 0) + x @ teacher
    head = np.all(z == 0, axis=1)
    population = dict(x=x, y=y, p=p, head=head, head_center=prototypes[0], head_radius=radius)
    functional = dict(x=x, p=p, directions=np.sqrt(p)[:, None] * z / np.sqrt(q))
    witness_w = np.zeros((3, dimension))
    witness_w[0, 0], witness_w[1], witness_w[2] = 1, teacher, -teacher
    witness = (witness_w, np.array([12., 3., -3.]), np.array(0.))
    return population, audit, functional, dict(q=q, z=z, witness=witness,
        p_empty=float(p[head].sum()), cooccurrence_mass=float(p[(z != 0).sum(1) >= 2].sum()))


def forward(state, x):
    w, a, c = state
    pre = w @ np.swapaxes(x, -1, -2)
    return c[..., None] + np.mean(a[..., None] * np.maximum(pre, 0), axis=-2)


def velocity(state, x, y, p):
    w, a, _ = state
    pre = w @ np.swapaxes(x, -1, -2)
    pe = p * (y - forward(state, x))
    return (a[..., None] * (((pre > 0) * pe[..., None, :]) @ x),
            np.sum(np.maximum(pre, 0) * pe[..., None, :], axis=-1), pe.sum(-1))


def factor_response_power(state, data, functional):
    """Unfiltered factor response; exact zero implies a current factor loss floor."""
    w, a, _ = state
    x, p = data["x"], data["p"]
    pre = w @ x.T
    weighted = np.sqrt(p)[:, None] * functional["directions"]
    readout = np.maximum(pre, 0) @ weighted
    incoming = a[:, None, None] * np.einsum("jn,nr,nd->jrd", pre > 0, weighted, x)
    return (np.sum(readout**2, axis=0) + np.sum(incoming**2, axis=(0, 2))) / len(a)
