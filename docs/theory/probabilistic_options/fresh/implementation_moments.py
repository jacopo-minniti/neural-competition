"""Functional implementation moments and controlled refinement hierarchies."""

from __future__ import annotations

import itertools

import numpy as np
from toy_option_resources import forward, jacobian, kernel


def hierarchy(n=2, ratio=0.25, overlap=1.0):
    """Categorical conditional refinements, all present; exact width 2+2N witness."""
    q = 0.2 * ratio ** np.arange(n)
    q /= q.sum() / 0.2
    probabilities = np.r_[1 - q.sum(), q]
    x, p, y, context, attributes = [], [], [], [], []
    coeff = 2 * (-1.0) ** np.arange(n)
    v = np.array([overlap, np.sqrt(1 - overlap**2)])
    for r, pr in enumerate(probabilities):
        for z in itertools.product([-1.0, 1.0], repeat=2):
            c = np.full(n, -2.0)
            if r:
                c[r - 1] = 2.0
            x.append(np.r_[c, z, 1.0])
            p.append(pr / 4)
            y.append(2 * z[0] + (coeff[r - 1] * np.dot(v, z) if r else 0))
            context.append(r)
            attributes.append(z)
    x, p, y, context, z = map(np.array, [x, p, y, context, attributes])
    refs = np.stack([(context == j + 1) - q[j] for j in range(n)], axis=1) * (z @ v)[:, None]
    columns = np.sqrt(p[:, None]) * refs
    directions = columns / np.linalg.norm(columns, axis=0)
    u, sv, _ = np.linalg.svd(columns, full_matrices=False)
    u = u[:, sv > 1e-12]
    m = 2 + 2 * n
    w = np.zeros((m, n + 3))
    a = np.zeros(m)
    w[0, n] = 1
    w[1, n] = -1
    a[:2] = m * np.array([2.0, -2.0])
    for j in range(n):
        w[2 + 2 * j, j] = w[3 + 2 * j, j] = 1
        w[2 + 2 * j, n : n + 2] = v
        w[3 + 2 * j, n : n + 2] = -v
        a[2 + 2 * j : 4 + 2 * j] = m * coeff[j] / 2 * np.array([1.0, -1.0])
    np.testing.assert_allclose(forward((w[None], a[None], np.zeros(1)), x)[0], y, atol=1e-12)
    return dict(
        x=x,
        p=p,
        y=y,
        root=np.sqrt(p),
        q=q,
        n=n,
        ratio=ratio,
        overlap=overlap,
        context=context,
        z=z,
        v=v,
        coeff=coeff,
        directions=directions,
        u=u,
        capacity=m,
        gram=np.sqrt(p[:, None]) * (x @ x.T) * np.sqrt(p[None, :]),
    )


def moments(state, data):
    """One model -> unordered gate-conditioned moments of (a, sqrt(P) Xw)."""
    w, a, c = (v[0] for v in state)
    z = w @ data["x"].T
    gates, inverse = np.unique(z > 0, axis=0, return_inverse=True)
    phi = z * data["root"]
    groups = []
    for j, g in enumerate(gates):
        chosen = inverse == j
        ff = phi[chosen]
        aa = a[chosen]
        groups.append(
            dict(
                g=g.astype(float),
                s=float(aa @ aa / len(a)),
                q=aa @ ff / len(a),
                R=ff.T @ ff / len(a),
                count=int(chosen.sum()),
            )
        )
    return dict(groups=groups, c=float(c), width=len(a))


def moment_output(resource, data):
    return sum(g["g"] * g["q"] for g in resource["groups"]) + resource["c"] * data["root"]


def moment_kernel(resource, data):
    return np.outer(data["root"], data["root"]) + sum(
        g["g"][:, None] * (g["R"] + g["s"] * data["gram"]) * g["g"][None, :]
        for g in resource["groups"]
    )


def moment_step(resource, data, dt):
    """Exact finite Euler update of moments if the original masks remain fixed."""
    e = data["root"] * data["y"] - moment_output(resource, data)
    out = []
    for g in resource["groups"]:
        pressure = g["g"] * e
        v = data["gram"] @ pressure
        q, R, s = g["q"], g["R"], g["s"]
        out.append(
            dict(
                g=g["g"],
                count=g["count"],
                s=s + 2 * dt * (q @ pressure) + dt**2 * (pressure @ R @ pressure),
                q=q + dt * (R @ pressure + s * v) + dt**2 * (q @ pressure) * v,
                R=R + dt * (np.outer(v, q) + np.outer(q, v)) + dt**2 * s * np.outer(v, v),
            )
        )
    return dict(groups=out, c=resource["c"] + dt * (data["root"] @ e), width=resource["width"])


def response_from_kernel(k, tau=20):
    eig, vec = np.linalg.eigh(k)
    eig = np.maximum(eig, 0)
    return (vec * (eig / (eig + 0.5 / tau))) @ vec.T


def observe(state, data):
    e = data["y"] - forward(state, data["x"])
    n = data["n"]
    p = data["p"]
    root = data["root"]
    d = data["directions"]
    k = kernel(state, data)
    eig, vec = np.linalg.eigh(k)
    eig = np.maximum(eig, 0)
    jj = jacobian(state, data)
    jj /= np.maximum(np.linalg.norm(jj, axis=1, keepdims=True), 1e-30)
    uu, ss, _ = np.linalg.svd(jj, full_matrices=False)
    keep = ss > 1e-9 * np.maximum(ss[:, :1], 1)
    projection = (uu * keep[:, None, :]) @ uu.transpose(0, 2, 1)
    coverage = np.einsum("ni,snk,ki->si", d, projection, d)
    amplitude = np.stack(
        [
            np.mean(
                e[:, data["context"] == j] * (data["z"][data["context"] == j] @ data["v"]), axis=1
            )
            for j in range(n + 1)
        ],
        axis=1,
    )
    gap = (amplitude[:, 1:] - amplitude[:, :1]) / data["coeff"]
    context_loss = np.stack(
        [0.5 * np.mean(e[:, data["context"] == j] ** 2, axis=1) for j in range(n + 1)], axis=1
    )
    result = dict(
        loss=0.5 * np.sum(p * e**2, axis=1),
        coverage=coverage,
        gap_error=gap,
        context_loss=context_loss,
        residual_loading=0.5 * ((e * root) @ d) ** 2,
    )
    for tau in [20, 1000]:
        filt = eig / (eig + 0.5 / tau)
        b = (vec * filt[:, None, :]) @ vec.transpose(0, 2, 1)
        result[f"strength_{tau}"] = np.einsum("ni,snk,ki->si", d, b, d)
        result[f"joint_min_{tau}"] = np.linalg.eigvalsh(
            np.einsum("ni,snk,kj->sij", data["u"], b, data["u"])
        )[:, 0]
        result[f"loaded_gain_{tau}"] = 0.5 * np.einsum("si,sij,sj->s", e * root, b, e * root)
    return result


def mix_within_gate(state, data, rng):
    """Orthogonally mix a random same-gate pair; preserve ALL joint moments.

    Choose the largest feasible angle from a predetermined grid, with random sign.
    No future loss or target alignment enters the choice.
    """
    w, a, c = state
    gates = w[0] @ data["x"].T > 0
    pairs = [
        (j, k)
        for j in range(len(a[0]))
        for k in range(j + 1, len(a[0]))
        if np.array_equal(gates[j], gates[k])
    ]
    rng.shuffle(pairs)
    for j, k in pairs:
        sign = rng.choice([-1.0, 1.0])
        for angle in [sign * t for t in [0.7, 0.4, 0.2, 0.1, 0.05]]:
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            ww = w.copy()
            aa = a.copy()
            ww[0, [j, k]] = rotation @ w[0, [j, k]]
            aa[0, [j, k]] = rotation @ a[0, [j, k]]
            if np.array_equal(ww[0] @ data["x"].T > 0, gates):
                return (ww, aa, c.copy()), dict(pair=[j, k], angle=angle)
    return None


def functional_particles(state, data):
    """Permutation-invariant empirical measure represented by weighted feature atoms.

    Atoms are (a, phi), phi=sqrt(P)Xw; each has weight 1/m. Positive rescaling
    moves into atom mass, giving the equivalent angular resource measure.
    """
    w, a, c = state
    return ((w @ data["x"].T) * data["root"], a.copy(), c.copy())


def particle_output(state, data):
    phi, a, c = state
    return c[:, None] * data["root"] + np.mean(a[..., None] * np.maximum(phi, 0), axis=1)


def particle_step(state, data, dt):
    phi, a, c = state
    residual = data["root"] * data["y"] - particle_output(state, data)
    da = np.sum(np.maximum(phi, 0) * residual[:, None], axis=2)
    dphi = a[..., None] * (((phi > 0) * residual[:, None]) @ data["gram"])
    return phi + dt * dphi, a + dt * da, c + dt * (residual @ data["root"])
