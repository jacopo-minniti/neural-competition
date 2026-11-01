"""Independent checks for the exploratory natural-training pressure test."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/studies'))
from audit_probabilistic_option_transport import advance  # noqa: E402
from probabilistic_option_transport import transport  # noqa: E402
from probe_probabilistic_natural_training import fields, forecast, measure, transported  # noqa: E402


def test_batched_full_task_update_matches_original_audit():
    rng = np.random.default_rng(744)
    w, a, c = rng.normal(size=(9, 4, 2)), rng.normal(size=(9, 4)), rng.normal(size=9)
    angle, q, step = np.pi / 3, .1, .002
    x = np.array([[1., 0.], [-np.cos(angle), np.sin(angle)],
                  [-np.cos(angle), -np.sin(angle)]])
    dw, da, dc, _, _ = fields(w, a, c, angle, q)
    for index in range(len(w)):
        reference = advance(w[index], a[index], c[index], x,
                            np.array([1-q, q/2, q/2]), np.array([4., 1., -1.]), step)
        for actual, expected in zip((w[index]+step*dw[index], a[index]+step*da[index],
                                     c[index]+step*dc[index]), reference):
            np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_response_matches_full_weighted_jacobian():
    rng = np.random.default_rng(745)
    w, a = rng.normal(size=(12, 4, 2)), rng.normal(size=(12, 4))
    angle, q = np.pi / 12, .01
    x = np.array([[1., 0.], [-np.cos(angle), np.sin(angle)],
                  [-np.cos(angle), -np.sin(angle)]])
    direction = np.array([0., 1., -1.])/np.sqrt(2)
    observed = measure(w, a, np.zeros(len(w)), angle, q)
    for index in range(len(w)):
        pre = x @ w[index].T
        incoming = ((pre > 0)*a[index])[..., None]*x[:, None, :]
        jac = np.concatenate([np.maximum(pre, 0), incoming.reshape(3, -1)], axis=1)/2
        jac = np.column_stack([jac, np.ones(3)])*np.sqrt([1-q, q/2, q/2])[:, None]
        kernel = jac @ jac.T
        value = direction @ kernel @ np.linalg.solve(kernel+.025*np.eye(3), direction)
        np.testing.assert_allclose(value, observed['response'][index], atol=2e-13)


def test_individual_clocks_reproduce_exact_pressure_flow_and_intercept_case():
    rng = np.random.default_rng(746)
    w, a = rng.normal(size=(7, 4, 2)), rng.normal(size=(7, 4))
    clocks = np.linspace(0, 2, len(w))
    actual_w, actual_a = transported(w, a, clocks)
    for index, clock in enumerate(clocks):
        expected_w, expected_a = transport(w[index], a[index], clock)
        np.testing.assert_allclose(actual_w[index], expected_w)
        np.testing.assert_allclose(actual_a[index], expected_a)
    # With all common gates inactive, only the intercept learns: u=4(1-exp(-(1-q)t)).
    w = np.full((3, 4, 2), -1.)
    a = np.ones((3, 4))
    result = forecast(w, a, .1, [0., 1.], .01)
    np.testing.assert_allclose(result[1.], 4*(1-np.exp(-.9)), atol=1e-10)
