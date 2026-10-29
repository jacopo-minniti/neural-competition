"""Causal controls: match all initial response directions and keep anchors fixed."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from probe_causal_response import evolve, physical, signed_expand  # noqa: E402
from probe_response_feedback import cache, freeze_gates, power, velocity  # noqa: E402


def setup():
    torch.manual_seed(721)
    s = dict(
        w=torch.randn(3, 4, dtype=torch.float64),
        b=torch.randn(3, dtype=torch.float64),
        a=torch.randn(10, 3, dtype=torch.float64),
        c=torch.randn(10, dtype=torch.float64),
        norm=np.sqrt(3),
        mw=1.0,
        ma=1.0,
    )
    return s, torch.randn(6, 4, dtype=torch.float64)


def kernel(s, x):
    params = tuple(s[k] for k in ["w", "b", "a", "c"])

    def f(w, b, a, c):
        return (x @ w.T + b).relu() @ a.T / s["norm"] + c

    jac = torch.autograd.functional.jacobian(f, params)
    return sum(
        metric * v.reshape(60, -1) @ v.reshape(60, -1).T
        for metric, v in zip([s["mw"], s["mw"], s["ma"], 1.0], jac)
    ) / len(x)


def test_signed_copy_matches_entire_kernel_and_reduces_feature_velocity():
    s, x = setup()
    signed = signed_expand(s)
    torch.testing.assert_close(cache(s, x)[0], cache(signed, x)[0])
    torch.testing.assert_close(kernel(s, x), kernel(signed, x))
    d = torch.randn(6, 10, dtype=torch.float64)
    v, vs = velocity(s, x, d), velocity(signed, x, d)
    torch.testing.assert_close(vs["w"][:3], 0.4 * v["w"])
    torch.testing.assert_close(vs["w"][3:], -0.2 * v["w"])
    torch.testing.assert_close(vs["a"][:, :3], 0.5 * v["a"])


def test_common_output_anchor_preserves_jacobian_and_gradient():
    s, x = setup()
    common = torch.randn(6, 10, dtype=torch.float64)
    target = torch.softmax(torch.randn_like(common), -1)
    offset = (common - cache(s, x)[0]).detach()
    torch.testing.assert_close(cache(s, x)[0] + offset, common)
    d = target - common.softmax(-1)
    expected = velocity(s, x, d)
    params = {k: s[k].clone().requires_grad_() for k in ["w", "b", "a", "c"]}
    score = torch.nn.functional.cross_entropy(cache(dict(s, **params), x)[0] + offset, target)
    gradients = torch.autograd.grad(score, tuple(params.values()))
    for k, g in zip(params, gradients):
        torch.testing.assert_close(expected[k], -g)
    rate = 1e-6
    after = evolve(s, x, target, 1, rate, offset)
    actual = (
        float(score.detach())
        - float(torch.nn.functional.cross_entropy(cache(after, x)[0] + offset, target))
    ) / rate
    np.testing.assert_allclose(actual, power(s, x, d), rtol=1e-4)


def test_refresh_uses_only_predicted_parameters_and_one_step_is_exact():
    s, x = setup()
    target = torch.softmax(torch.randn(6, 10, dtype=torch.float64), -1)
    real = evolve(s, x, target, 4, 0.2)
    refreshed = physical(evolve(freeze_gates(s), x, target, 4, 0.2, refresh=1))
    for k in ["w", "b", "a", "c"]:
        torch.testing.assert_close(refreshed[k], real[k])


def test_finite_budget_response_matches_dense_spectral_filter():
    from probe_causal_response import quadratic_gains

    s, x = setup()
    s = signed_expand(s)
    params = tuple(s[k] for k in ["w", "b", "a", "c"])

    def f(w, b, a, c):
        return (x @ w.T + b).relu() @ a.T / s["norm"] + c

    derivatives = torch.autograd.functional.jacobian(f, params)
    j = torch.cat(
        [
            v.reshape(60, -1) * np.sqrt(metric)
            for v, metric in zip(derivatives, [s["mw"], s["mw"], s["ma"], 1.0])
        ],
        dim=1,
    )
    p = cache(s, x)[0].softmax(-1)
    d = torch.randn_like(p)
    h = torch.block_diag(*(torch.diag(pi) - pi[:, None] * pi[None, :] for pi in p))
    a = j.T @ h @ j / len(x)
    g = j.T @ d.flatten() / len(x)
    eigen, vectors = torch.linalg.eigh(a)
    eigen = eigen.clamp_min(0)
    loaded = vectors.T @ g
    rate = 0.02
    observed = quadratic_gains(s, x, d, p, 20, rate)
    for steps in [1, 5, 20]:
        filt = torch.where(
            eigen > 1e-12, (1 - (1 - rate * eigen) ** (2 * steps)) / eigen, 2 * rate * steps
        )
        expected = float(0.5 * (loaded.square() * filt).sum())
        np.testing.assert_allclose(observed[str(steps)], expected, rtol=1e-9, atol=1e-10)


def test_lower_discounted_value_does_not_imply_lower_fixed_budget_gain():
    before = np.array([1.0, 0.05])
    after = np.array([0.5, 0.15])

    def discounted(k):
        return 0.5 * np.sum(k / (k + 0.5))

    def gain(k, t):
        return 0.5 * np.sum(1 - np.exp(-2 * t * k))

    assert discounted(after) < discounted(before)
    assert gain(after, 0.1) < gain(before, 0.1)
    assert gain(after, 5.0) > gain(before, 5.0)
