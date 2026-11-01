"""Checks of the new mathematical claims, independent of endpoint forecasts."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from intervene_delayed_options import expand, gram, pred, step  # noqa: E402
from probe_delayed_option_recovery import measure  # noqa: E402
from probe_option_transport_resilience import audit  # noqa: E402


def test_response_matching_and_smooth_transport():
    assert max(audit().values()) < 1e-11


def test_batched_intervention_metric_matches_autodiff():
    rng = np.random.default_rng(991)
    x = np.array([[-2, -1, 1], [-2, 1, 1], [2, -1, 1], [2, 1, 1.0]])
    p = np.array([0.45, 0.45, 0.05, 0.05])
    y = np.array([-2, 2, -4, 4.0])
    s = (rng.normal(size=(3, 4, 3)), rng.normal(size=(3, 4)), np.zeros(3), 4, 4, 4)
    reserve = expand(s)
    uniform = expand(s, uniform=True)
    np.testing.assert_allclose(gram(reserve, x, p), gram(s, x, p), atol=1e-14)
    np.testing.assert_allclose(pred(reserve, x), pred(s, x), atol=1e-14)
    w, a, c = [torch.tensor(v[0], dtype=torch.float64, requires_grad=True) for v in reserve[:3]]
    xt = torch.tensor(x)
    f = torch.relu(xt @ w.T) @ a / 4 + c
    loss = (torch.tensor(p) * (f - torch.tensor(y)) ** 2).sum() / 2
    loss.backward()
    before = [v.copy() for v in reserve[:3]]
    step(reserve, x, p, y, 0.001)
    for i, (parameter, metric) in enumerate(zip([w, a, c], [4, 1, 1])):
        np.testing.assert_allclose(
            reserve[i][0], before[i][0] - 0.001 * metric * parameter.grad.numpy(), atol=1e-14
        )
    for _ in range(30):
        step(s, x, p, y, 0.01)
        step(uniform, x, p, y, 0.01)
    np.testing.assert_allclose(pred(s, x), pred(uniform, x), atol=1e-13)


def test_blind_floor_is_residual_option_energy_and_geometry_distance():
    x = np.array([[-2, -1, 1], [-2, 1, 1], [2, -1, 1], [2, 1, 1.0]])
    p = np.array([0.45, 0.45, 0.05, 0.05])
    y = np.array([-2, 2, -4, 4.0])
    # One inactive neuron: all four examples have the same output, no output edits
    # distinguish them. This is a structural witness, not a training initializer.
    state = (np.array([[[0.0, 0.0, -2.0]]]), np.array([[1.0]]), np.array([0.0]))
    r = measure(state, x, p, y)
    np.testing.assert_allclose(r["blind_floor"], [2.6], atol=1e-14)
    np.testing.assert_allclose(r["null_energy"], [2.6], atol=1e-14)
    np.testing.assert_allclose(r["blind_distance2"], [4 / 6], atol=1e-14)
    assert r["blind_recovery_clock"][0] > 1e8


def test_time_order_cannot_be_replaced_by_integrated_spectrum():
    # Same two PSD operators, same integrated spectrum, different demand response.
    p = np.array([[1.0, 0.0], [0.0, 0.0]])
    v = np.array([1.0, 1.0]) / np.sqrt(2)
    q = np.outer(v, v)
    a = np.eye(2) + (np.exp(-2) - 1) * p
    b = np.eye(2) + (np.exp(-2) - 1) * q
    d = np.array([1.0, 0.0])
    u1, u2 = b @ a, a @ b
    assert abs(np.linalg.norm(u1 @ d) ** 2 - np.linalg.norm(u2 @ d) ** 2) > 0.1
    for u in [u1, u2]:
        benefit = np.eye(2) - u.T @ u
        assert np.linalg.eigvalsh(benefit).min() > -1e-12
        np.testing.assert_allclose(0.5 * d @ benefit @ d, 0.5 * (d @ d - (u @ d) @ (u @ d)))
