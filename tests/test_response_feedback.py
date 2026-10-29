"""Independent checks of response matching and the simplified feedback model."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from probe_response_feedback import (  # noqa: E402
    cache,
    expand,
    freeze_gates,
    linearize,
    option_value,
    power,
    update,
    velocity,
)


def sample():
    torch.manual_seed(42)
    s = dict(
        w=torch.randn(3, 4, dtype=torch.float64),
        b=torch.randn(3, dtype=torch.float64),
        a=torch.randn(10, 3, dtype=torch.float64),
        c=torch.randn(10, dtype=torch.float64),
        norm=np.sqrt(3),
        mw=1.0,
        ma=1.0,
    )
    x = torch.randn(7, 4, dtype=torch.float64)
    return s, x


def gram(s, x):
    w, b, a, c = [s[k].clone().requires_grad_() for k in ["w", "b", "a", "c"]]

    def f(w, b, a, c):
        return (x @ w.T + b).relu() @ a.T / s["norm"] + c

    j = torch.autograd.functional.jacobian(f, (w, b, a, c))
    return sum(
        metric * v.reshape(70, -1) @ v.reshape(70, -1).T
        for metric, v in zip([s["mw"], s["mw"], s["ma"], 1.0], j)
    ) / len(x)


def test_full_kernel_matching_and_exact_one_step_feedback_even_across_gates():
    s, x = sample()
    reserve = expand(s)
    torch.testing.assert_close(gram(s, x), gram(reserve, x))
    torch.testing.assert_close(cache(s, x)[0], cache(reserve, x)[0])
    e = torch.randn(7, 10, dtype=torch.float64)
    v = velocity(s, x, e)
    vr = velocity(reserve, x, e)
    rate = 1.3
    after = update(s, v, rate)
    ar = update(reserve, vr, rate)
    dh = cache(after, x)[2] - cache(s, x)[2]
    expected = -(1 - 1 / 4) * rate * (dh @ v["a"].T) / s["norm"]
    torch.testing.assert_close(cache(ar, x)[0] - cache(after, x)[0], expected)
    assert ((cache(after, x)[1] > 0) != (cache(s, x)[1] > 0)).any()


def test_smooth_transport_is_four_times_bilinear_work():
    s, x = sample()
    e = torch.randn(7, 10, dtype=torch.float64)
    v = velocity(s, x, e)
    hdot = (x @ v["w"].T + v["b"]) * (cache(s, x)[1] > 0)
    q = float((e * (hdot @ v["a"].T) / s["norm"]).sum(1).mean())
    eps = 1e-6
    derivative = (power(update(s, v, eps), x, e) - power(update(s, v, -eps), x, e)) / (2 * eps)
    np.testing.assert_allclose(derivative, 4 * q, rtol=1e-7, atol=1e-8)
    r = expand(s)
    rv = velocity(r, x, e)
    dr = (power(update(r, rv, eps), x, e) - power(update(r, rv, -eps), x, e)) / (2 * eps)
    np.testing.assert_allclose(dr, derivative / 4, rtol=1e-7, atol=1e-8)


def test_fixed_gate_model_gradient_matches_autograd():
    s, x = sample()
    s = freeze_gates(s)
    s["w"] += 0.8
    e = torch.randn(7, 10, dtype=torch.float64)
    actual = velocity(s, x, e)
    for key in ["w", "b", "a", "c"]:
        s[key].requires_grad_()
    score = (cache(s, x)[0] * e).sum() / len(x)
    grads = torch.autograd.grad(score, [s[k] for k in ["w", "b", "a", "c"]])
    for key, g in zip(["w", "b", "a", "c"], grads):
        torch.testing.assert_close(actual[key], g)


def test_linear_response_has_fixed_kernel_and_correct_gradient():
    s, x = sample()
    e = torch.randn(7, 10, dtype=torch.float64)
    s = linearize(s)
    before = power(s, x, e)
    s["w"] += 0.8
    s["a"] -= 0.4
    np.testing.assert_allclose(power(s, x, e), before, atol=1e-13)
    actual = velocity(s, x, e)
    for key in ["w", "b", "a", "c"]:
        s[key].requires_grad_()
    score = (cache(s, x)[0] * e).sum() / len(x)
    gradients = torch.autograd.grad(score, [s[k] for k in ["w", "b", "a", "c"]])
    for key, g in zip(["w", "b", "a", "c"], gradients):
        torch.testing.assert_close(actual[key], g)


def test_full_measure_resolvent_matches_dense_autograd():
    s, x = sample()
    d = torch.randn(7, 10, dtype=torch.float64)
    p = cache(s, x)[0].softmax(-1)
    w, b, a, c = [s[k].clone().requires_grad_() for k in ["w", "b", "a", "c"]]

    def f(w, b, a, c):
        return (x @ w.T + b).relu() @ a.T / s["norm"] + c

    derivatives = torch.autograd.functional.jacobian(f, (w, b, a, c))
    j = torch.cat([v.reshape(70, -1) for v in derivatives], dim=1)
    h = torch.block_diag(*(torch.diag(pi) - pi[:, None] * pi[None, :] for pi in p))
    curvature = j.T @ h @ j / len(x)
    rhs = j.T @ d.flatten() / len(x)
    expected = 0.5 * rhs @ torch.linalg.solve(curvature + 0.5 * torch.eye(len(rhs)), rhs)
    actual = option_value(s, x, d, p)
    np.testing.assert_allclose(actual["value"], float(expected), rtol=1e-8, atol=1e-10)
    matched = option_value(expand(s), x, d, p)
    np.testing.assert_allclose(actual["value"], matched["value"], rtol=1e-10)
