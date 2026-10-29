"""Checks of probability fitting, event chronology, and frequency transfer tasks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from probabilistic_options import (
    clock,
    episodes,
    fit,
    loss_gradient_hessian,
    probability,
    weighted_task,
)  # noqa: E402


def test_cloglog_derivatives_match_finite_differences():
    eta = np.array([-6.0, -2.0, 0.0, 2.0])
    eps = 1e-5
    for event in [0.0, 1.0]:
        y = np.full(len(eta), event)
        loss, grad, hess = loss_gradient_hessian(eta, y)
        lp, gp, _ = loss_gradient_hessian(eta + eps, y)
        lm, gm, _ = loss_gradient_hessian(eta - eps, y)
        np.testing.assert_allclose(grad, (lp - lm) / (2 * eps), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(hess, (gp - gm) / (2 * eps), rtol=1e-5, atol=1e-8)
        assert np.all(np.isfinite(loss))


def test_rate_fit_recovers_known_probabilities():
    rng = np.random.default_rng(297001)
    x = rng.normal(size=6000)
    p = -np.expm1(-np.exp(-0.5 - 0.8 * x))
    y = rng.random(len(x)) < p
    rows = [dict(features=dict(x=float(a)), event=int(b), weight=1.0) for a, b in zip(x, y)]
    model = fit(rows, ["x"], penalty=0)
    predicted = probability(model, [dict(x=float(a)) for a in x])
    assert np.sqrt(np.mean((predicted - p) ** 2)) < 0.03
    assert model["coef"][1] < 0


def test_log_clock_has_a_heterogeneous_rate_interpretation():
    rng = np.random.default_rng(297002)
    shape = 1.7
    rates = rng.gamma(shape, 1 / 20.0, size=200000)
    for horizon in [30.0, 50.0, 100.0]:
        expected = np.exp(-shape * clock(horizon, "log"))
        assert abs(np.mean(np.exp(-rates * (horizon - 20))) - expected) < 0.002


def test_zipf_weights_preserve_target_and_centered_contrasts():
    for n in [3, 4]:
        d = weighted_task(n, "zipf", 1.3, 0.5)
        q = d["q"]
        np.testing.assert_allclose(q / q[0], np.arange(1, n + 1) ** -1.3)
        np.testing.assert_allclose(d["p"].sum(), 1)
        np.testing.assert_allclose(d["directions"].T @ d["root"], 0, atol=1e-14)
        uniform = weighted_task(n, "geometric", 1.0, 0.5)
        np.testing.assert_array_equal(d["y"], uniform["y"])


def test_landmark_features_precede_events_and_first_opening_is_retained():
    def row(t, cov, gap, strength):
        return dict(
            bank=1,
            seed=0,
            case="test",
            width=4,
            n=1,
            ratio=1.0,
            overlap=0.0,
            time=t,
            coverage=[cov],
            gap_error=[gap],
            strength_1000=[strength],
            gate_groups=2,
        )

    path = [
        row(0, 1, 1, 0.001),
        row(10, 0.5, 1, 0.2),
        row(20, 0.5, 1, 0.3),
        row(30, 1, 0.9, 0.8),
        row(50, 0.5, 0.8, 0.1),
    ]
    close, opening, query = episodes({0: path})
    assert close[0]["event"] == 1 and close[0]["features"]["log_strength"] == np.log(0.001)
    assert len(opening) == 1 and opening[0]["event"] == 1
    assert opening[0]["features"]["log_strength"] == np.log(0.3)
    assert query[0]["event"] == 1  # First reopening occurred, even though it later closed.
    model = dict(features=[], center=[], scale=[], coef=[0.0], clock="log")
    probabilities = probability(model, [{}, {}], [clock(50, "log"), clock(100, "log")])
    assert 0 < probabilities[0] < probabilities[1] < 1
