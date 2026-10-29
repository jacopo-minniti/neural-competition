"""Weighted functional response geometry for heterogeneous contextual task families."""

from __future__ import annotations

import itertools

import numpy as np

CASES = {
    "discrete": dict(q=[0.1], detail=[2.0], base=2.0, nodes=1, spread=0.0, attributes=1),
    "continuous": dict(q=[0.1], detail=[2.0], base=2.0, nodes=3, spread=0.15, attributes=1),
    "rarer": dict(q=[0.02], detail=[2.0], base=2.0, nodes=1, spread=0.0, attributes=1),
    "centered": dict(q=[0.1], detail=[2.0], base=-0.2, nodes=1, spread=0.0, attributes=1),
    "two_refinements": dict(
        q=[0.1, 0.03], detail=[2.0, -1.0], base=2.0, nodes=1, spread=0.0, attributes=1
    ),
    "separate_attribute": dict(q=[0.1], detail=[2.0], base=2.0, nodes=1, spread=0.0, attributes=2),
}


def task(name, dense=False):
    config = CASES[name]
    q, detail = np.array(config["q"]), np.array(config["detail"])
    r, attributes = len(q), config["attributes"]
    nodes = 9 if dense and config["spread"] else config["nodes"]
    grid, weights = np.polynomial.legendre.leggauss(nodes)
    x, p, y, refs, groups = [], [], [], [], []
    for context, probability in enumerate(np.r_[1 - q.sum(), q]):
        for signs in itertools.product([-1.0, 1.0], repeat=attributes):
            for indices in itertools.product(range(nodes), repeat=r + attributes):
                contexts = np.full(r, -2.0)
                if context:
                    contexts[context - 1] = 2.0
                contexts += config["spread"] * grid[list(indices[:r])]
                z = np.array(signs) + config["spread"] * grid[list(indices[r:])]
                refinements = np.full(r, z[-1])
                indicator = np.eye(r + 1)[context, 1:]
                target = config["base"] * z[0] + np.sum(detail * indicator * refinements)
                x.append(np.r_[contexts, z, 1.0])
                p.append(probability * np.prod(weights[list(indices)]) / 2 ** (r + 2 * attributes))
                y.append(target)
                refs.append((indicator - q) * refinements)
                groups.append((signs, indices[r:]))
    x, p, y, refs = np.array(x), np.array(p), np.array(y), np.array(refs)
    p /= p.sum()
    unique = {v: i for i, v in enumerate(dict.fromkeys(groups))}
    ids = np.array([unique[v] for v in groups])
    aggregation = np.eye(len(unique))[ids] * p[:, None]
    aggregation /= aggregation.sum(0, keepdims=True)
    root = np.sqrt(p)
    ref_y = refs @ detail
    columns = root[:, None] * refs
    uu, sv, _ = np.linalg.svd(columns, full_matrices=False)
    u = uu[:, sv > 1e-12]
    common = y - ref_y
    main = root * common
    if np.linalg.norm(main) < 1e-12:
        main = root * x[:, r]
    main /= np.linalg.norm(main)
    probes = np.column_stack([main, u])
    capacity = 2 + 2 * r
    # Explicit capacity witness on the whole bounded support.
    witness = config["base"] * x[:, r]
    for j in range(r):
        witness += (
            detail[j]
            / 2
            * (
                np.maximum(x[:, j] + x[:, r + attributes - 1], 0)
                - np.maximum(x[:, j] - x[:, r + attributes - 1], 0)
            )
        )
    assert np.allclose(witness, y, atol=1e-12)
    return dict(
        x=x,
        p=p,
        y=y,
        root=root,
        ref_y=ref_y,
        u=u,
        probes=probes,
        aggregation=aggregation,
        groups=ids,
        capacity=capacity,
        config=config,
    )


def common(e, data):
    return (e @ data["aggregation"])[:, data["groups"]]


def forward(state, x):
    w, a, c = state
    z = w @ x.T
    return c[:, None] + np.mean(a[..., None] * np.maximum(z, 0), axis=1)


def velocity(state, data, e=None):
    w, a, _ = state
    z = w @ data["x"].T
    if e is None:
        e = data["y"] - forward(state, data["x"])
    pe = e * data["p"]
    return (
        a[..., None] * (((z > 0) * pe[:, None]) @ data["x"]),
        np.sum(np.maximum(z, 0) * pe[:, None], axis=2),
        pe.sum(1),
    )


def norm2(v, m):
    return (np.sum(v[0] ** 2, axis=(1, 2)) + np.sum(v[1] ** 2, axis=1)) / m + v[2] ** 2


def advance(state, v, dt):
    return tuple(s + dt * d for s, d in zip(state, v))


def kernel(state, data):
    w, a, _ = state
    z = w @ data["x"].T
    h = np.maximum(z, 0)
    gate_a = (z > 0) * a[..., None]
    k = (
        np.einsum("smi,smj->sij", h, h)
        + np.einsum("smi,smj->sij", gate_a, gate_a) * (data["x"] @ data["x"].T)
    ) / a.shape[1] + 1
    return k * data["root"][None, :, None] * data["root"][None, None, :]


def response(state, data, tau=20.0):
    k = kernel(state, data)
    eig, vec = np.linalg.eigh(k)
    eig = np.maximum(eig, 0)
    filt = eig / (eig + 0.5 / tau)
    return (vec * filt[:, None, :]) @ np.swapaxes(vec, 1, 2)


def jacobian(state, data):
    w, a, _ = state
    z = w @ data["x"].T
    # Whitened parameter columns, with M_w=M_a=m and M_c=1.
    incoming = ((z > 0)[..., None] * data["x"][None, None] * a[:, :, None, None]).transpose(
        0, 2, 1, 3
    )
    d = np.concatenate(
        [np.maximum(z, 0).transpose(0, 2, 1), incoming.reshape(len(w), len(data["x"]), -1)], axis=2
    ) / np.sqrt(a.shape[1])
    d = np.concatenate([d, np.ones((len(w), len(data["x"]), 1))], axis=2)
    return d * data["root"][None, :, None]


def measure(state, data):
    w, a, _ = state
    prediction = forward(state, data["x"])
    e = data["y"] - prediction
    ec = common(e, data)
    er = e - ec
    k = kernel(state, data)
    eig, vec = np.linalg.eigh(k)
    eig = np.maximum(eig, 0)
    root_e = e * data["root"]
    load = np.einsum("sni,sn->si", vec, root_e) ** 2
    fixed = data["root"] * data["ref_y"]
    fixed /= np.linalg.norm(fixed)
    loading = np.einsum("sni,n->si", vec, fixed) ** 2
    # Column normalization measures structural range without calling tiny initial
    # singular values zero merely because the initial parameter norm is small.
    d = jacobian(state, data)
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    u, sv, _ = np.linalg.svd(dn, full_matrices=False)
    keep = sv > 1e-9 * np.maximum(sv[:, :1], 1)
    overlap = np.einsum("sni,n->si", u, fixed) ** 2
    coverage = np.sum(overlap * keep, axis=1)
    null_e = 0.5 * (
        np.sum(root_e**2, axis=1) - np.sum(np.einsum("sni,sn->si", u, root_e) ** 2 * keep, axis=1)
    )
    vc, vr = velocity(state, data, ec), velocity(state, data, er)
    cross = (np.sum(vc[0] * vr[0], axis=(1, 2)) + np.sum(vc[1] * vr[1], axis=1)) / a.shape[1] + vc[
        2
    ] * vr[2]
    z = w @ data["x"].T
    q = data["ref_y"] / np.linalg.norm(data["root"] * data["ref_y"])
    da = np.sum(np.maximum(z, 0) * (data["p"] * q), axis=2)
    dw = a[..., None] * (((z > 0) * (data["p"] * q)) @ data["x"])
    contributions = (da**2 + np.sum(dw**2, axis=2)) / a.shape[1]
    diversity = contributions.sum(1) ** 2 / np.maximum(np.sum(contributions**2, axis=1), 1e-300)
    # Terminal readout fit separates missing learned features from unfinished
    # readout optimization; this is an oracle audit, not a training intervention.
    feature = (
        np.concatenate(
            [np.maximum(z, 0).transpose(0, 2, 1), np.ones((len(w), len(data["x"]), 1))], axis=2
        )
        * data["root"][None, :, None]
    )
    coef = np.linalg.pinv(feature, rcond=1e-10) @ (data["root"] * data["y"])
    resid = np.einsum("sni,si->sn", feature, coef) - data["root"] * data["y"]
    result = dict(
        loss=0.5 * np.sum(data["p"] * e**2, axis=1),
        common_energy=0.5 * np.sum(data["p"] * ec**2, axis=1),
        refinement_energy=0.5 * np.sum(data["p"] * er**2, axis=1),
        coverage=coverage,
        null_energy=np.maximum(null_e, 0),
        rank=keep.sum(1),
        common_power=norm2(vc, a.shape[1]),
        refinement_power=norm2(vr, a.shape[1]),
        pressure_overlap=cross,
        implementation_diversity=diversity,
        readout_floor=0.5 * np.sum(resid**2, axis=1),
    )
    for tau in [20.0, 1000.0]:
        filt = eig / (eig + 0.5 / tau)
        b = (vec * filt[:, None, :]) @ np.swapaxes(vec, 1, 2)
        compressed = np.einsum("ni,snk,kj->sij", data["u"], b, data["u"])
        result[f"refine_strength_{int(tau)}"] = np.sum(loading * filt, axis=1)
        result[f"min_refine_strength_{int(tau)}"] = np.linalg.eigvalsh(compressed)[:, 0]
        result[f"breadth_{int(tau)}"] = filt.sum(1)
        result[f"loaded_gain_{int(tau)}"] = 0.5 * np.sum(load * filt, axis=1)
    return result


def transport_audit(state, data, dt):
    e = data["y"] - forward(state, data["x"])
    ec = common(e, data)
    vc, vr = velocity(state, data, ec), velocity(state, data, e - ec)
    vf = tuple(a + b for a, b in zip(vc, vr))
    b0 = response(state, data)
    bc = response(advance(state, vc, dt), data)
    br = response(advance(state, vr, dt), data)
    bf = response(advance(state, vf, dt), data)
    c = 0.5 * (bc - b0 + bf - br)
    r = 0.5 * (br - b0 + bf - bc)
    assert np.max(abs(c + r - (bf - b0))) < 1e-12
    probes = data["probes"]
    return {
        k: np.einsum("ni,snk,kj->sij", probes, v, probes)
        for k, v in [("common", c), ("refinement", r), ("total", bf - b0)]
    }
