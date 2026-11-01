"""Independent mathematical checks of pressure transport and option events."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from probabilistic_option_transport import (  # noqa: E402
    blind_joint_probability,
    blind_probability,
    contrast_kernel,
    transition_probabilities,
    transport,
)


def test_transport_matches_gradient_field_and_handles_common_gate_death():
    w = np.array([[0.5, 0.3], [0.5, -0.8], [-0.2, 0.6]])
    a = np.array([0.8, -1.0, 0.9])
    eps = 1e-7
    wp, ap = transport(w, a, eps)
    np.testing.assert_allclose((wp - w) / eps, [[0.8, 0], [-1, 0], [0, 0]], atol=1e-7)
    np.testing.assert_allclose((ap - a) / eps, [0.5, 0.5, 0], atol=1e-7)
    wt, at = transport(w, a, 2)
    assert wt[1, 0] == 0
    np.testing.assert_allclose(at[1], -np.sqrt(0.75))
    np.testing.assert_allclose(at**2 - np.sum(wt**2, axis=1), a**2 - np.sum(w**2, axis=1))
    ws, ass = transport(*transport(w, a, 0.4), 1.6)
    np.testing.assert_allclose(ws, wt, atol=1e-14)
    np.testing.assert_allclose(ass, at, atol=1e-14)


def test_gaussian_orthant_formula_against_independent_polar_integral():
    grid, weight = np.polynomial.legendre.leggauss(256)
    angle = (grid + 1) * np.pi / 4
    for theta in [np.pi / 12, np.pi / 6, np.pi / 3]:
        np.testing.assert_allclose(blind_probability(theta, 0), 0.5 - theta / np.pi)
        for exposure in [0.2, 0.8, 2.0, 5.0]:
            numerator = np.cosh(exposure) * np.cos(angle) - np.tan(theta) * np.sin(angle)
            integral = weight @ (1 + numerator / np.hypot(np.sinh(exposure), numerator)) / 8
            np.testing.assert_allclose(blind_probability(theta, exposure), integral, atol=2e-13)
            joint = blind_joint_probability(theta, exposure)
            limit = np.pi / 2 - theta
            joint_angle = (grid + 1) * limit / 2
            joint_numerator = (np.cosh(exposure) * np.cos(joint_angle)
                               - np.tan(theta) * np.sin(joint_angle))
            joint_integral = limit / (4 * np.pi) * (weight @ (
                1 + joint_numerator / np.hypot(np.sinh(exposure), joint_numerator)))
            np.testing.assert_allclose(joint, joint_integral, atol=2e-13)
            assert max(0, blind_probability(theta, 0) + integral - 1) <= joint
            assert joint <= min(blind_probability(theta, 0), integral) + 1e-14
        np.testing.assert_allclose(blind_probability(theta, 20), 3 / 8, atol=3e-9)


def test_null_contrast_is_exactly_joint_gate_blindness():
    rng = np.random.default_rng(9127)
    w, a = rng.normal(size=(300, 2)), rng.normal(size=300)
    theta = np.pi / 3
    probes = np.array([[1, 0], [-np.cos(theta), np.sin(theta)],
                       [-np.cos(theta), -np.sin(theta)]])
    q = 0.1
    root = np.sqrt([1 - q, q / 2, q / 2])
    d = np.array([0, 1, -1]) / np.sqrt(2)
    for exposure in [0, 0.7, 2]:
        wt, at = transport(w, a, exposure)
        pre = wt @ probes.T
        h, g = np.maximum(pre, 0), pre > 0
        # Compute the entire weighted block independently from its Jacobian.
        incoming = at[:, None, None] * g[..., None] * probes[None]
        jac = np.concatenate([h[..., None], incoming], axis=2) * root[None, :, None]
        direct = np.sum(np.einsum("i,nij->nj", d, jac) ** 2, axis=1)
        np.testing.assert_allclose(contrast_kernel(wt, at, theta, q), direct, atol=1e-14)
        blind = np.all(pre[:, 1:] <= 0, axis=1)
        np.testing.assert_array_equal(direct == 0, blind)


def test_transitions_normalize_and_both_creation_and_destruction_occur():
    for theta in [np.pi / 12, np.pi / 3]:
        for width in [1, 4, 8]:
            initial = blind_probability(theta, 0) ** width
            result = transition_probabilities(theta, 2, width)
            np.testing.assert_allclose(
                initial + result["newly_closed"] - result["reopened"], result["closed"]
            )
            assert result["newly_closed"] > 0
            assert result["reopened"] > 0
            assert 0 <= result["closed_given_initially_open"] <= 1
            zero = transition_probabilities(theta, 0, width)
            assert abs(zero["newly_closed"]) < 1e-15
            assert abs(zero["reopened"]) < 1e-15
    assert blind_probability(np.pi / 12, 2) < blind_probability(np.pi / 12, 0)
    assert blind_probability(np.pi / 3, 2) > blind_probability(np.pi / 3, 0)


def test_joint_probability_resolves_crossings_near_zero_exposure():
    # The endpoint integral has a narrow boundary layer; fixed quadrature can miss it.
    theta = np.pi / 3
    b0 = blind_probability(theta, 0)
    for exposure in [1e-5, 1e-7, 1e-9]:
        joint = blind_joint_probability(theta, exposure)
        assert 0 < b0 - joint < exposure
        assert 0 < blind_probability(theta, exposure) - joint < exposure
