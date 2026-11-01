"""Tasks and spectral measurements for conditional-refinement hierarchies."""

from __future__ import annotations

import itertools

import numpy as np
from toy_option_resources import forward, jacobian, kernel


def hierarchy(n=2, ratio=0.25, overlap=1.0):
    """Build an N-refinement task with an exact width-(2+2N) witness."""
    q = 0.2 * ratio ** np.arange(n)
    q /= q.sum() / 0.2
    probabilities = np.r_[1 - q.sum(), q]
    x, p, y, context, attributes = [], [], [], [], []
    coeff = 2 * (-1.0) ** np.arange(n)
    v = np.array([overlap, np.sqrt(1 - overlap**2)])
    for r, probability in enumerate(probabilities):
        for z in itertools.product([-1.0, 1.0], repeat=2):
            c = np.full(n, -2.0)
            if r:
                c[r - 1] = 2.0
            x.append(np.r_[c, z, 1.0])
            p.append(probability / 4)
            y.append(2 * z[0] + (coeff[r - 1] * np.dot(v, z) if r else 0))
            context.append(r)
            attributes.append(z)
    x, p, y, context, z = map(np.array, [x, p, y, context, attributes])
    refs = np.stack([(context == j + 1) - q[j] for j in range(n)], axis=1) * (z @ v)[:, None]
    columns = np.sqrt(p[:, None]) * refs
    directions = columns / np.linalg.norm(columns, axis=0)
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    u = u[:, singular > 1e-12]
    capacity = 2 + 2 * n

    witness_w = np.zeros((capacity, n + 3))
    witness_a = np.zeros(capacity)
    witness_w[0, n] = 1
    witness_w[1, n] = -1
    witness_a[:2] = capacity * np.array([2.0, -2.0])
    for j in range(n):
        witness_w[2 + 2 * j, j] = witness_w[3 + 2 * j, j] = 1
        witness_w[2 + 2 * j, n : n + 2] = v
        witness_w[3 + 2 * j, n : n + 2] = -v
        witness_a[2 + 2 * j : 4 + 2 * j] = (
            capacity * coeff[j] / 2 * np.array([1.0, -1.0])
        )
    witness = (witness_w[None], witness_a[None], np.zeros(1))
    np.testing.assert_allclose(forward(witness, x)[0], y, atol=1e-12)

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
        capacity=capacity,
    )


def gate_summary(state, data):
    """Return descriptive gate-group counts without introducing a resource state."""
    gates = state[0][0] @ data["x"].T > 0
    _, counts = np.unique(gates, axis=0, return_counts=True)
    return len(counts), int(counts.max())


def observe(state, data):
    """Measure loss, residual demand, support, and finite-budget option strength."""
    residual = data["y"] - forward(state, data["x"])
    n = data["n"]
    p = data["p"]
    root = data["root"]
    directions = data["directions"]
    response_kernel = kernel(state, data)
    eigenvalues, eigenvectors = np.linalg.eigh(response_kernel)
    eigenvalues = np.maximum(eigenvalues, 0)

    derivative = jacobian(state, data)
    derivative /= np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 1e-30)
    left, singular, _ = np.linalg.svd(derivative, full_matrices=False)
    keep = singular > 1e-9 * np.maximum(singular[:, :1], 1)
    projection = (left * keep[:, None, :]) @ left.transpose(0, 2, 1)
    coverage = np.einsum("ni,snk,ki->si", directions, projection, directions)

    amplitude = np.stack(
        [
            np.mean(
                residual[:, data["context"] == j]
                * (data["z"][data["context"] == j] @ data["v"]),
                axis=1,
            )
            for j in range(n + 1)
        ],
        axis=1,
    )
    gap = (amplitude[:, 1:] - amplitude[:, :1]) / data["coeff"]
    context_loss = np.stack(
        [
            0.5 * np.mean(residual[:, data["context"] == j] ** 2, axis=1)
            for j in range(n + 1)
        ],
        axis=1,
    )
    result = dict(
        loss=0.5 * np.sum(p * residual**2, axis=1),
        coverage=coverage,
        gap_error=gap,
        context_loss=context_loss,
        residual_loading=0.5 * ((residual * root) @ directions) ** 2,
    )
    for tau in [20, 1000]:
        filtered = eigenvalues / (eigenvalues + 0.5 / tau)
        response = (eigenvectors * filtered[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
        result[f"strength_{tau}"] = np.einsum(
            "ni,snk,ki->si", directions, response, directions
        )
        result[f"joint_min_{tau}"] = np.linalg.eigvalsh(
            np.einsum("ni,snk,kj->sij", data["u"], response, data["u"])
        )[:, 0]
        result[f"loaded_gain_{tau}"] = 0.5 * np.einsum(
            "si,sij,sj->s", residual * root, response, residual * root
        )
    return result
