"""Independent checks of pressure identities, transport, and SGD curvature."""

import numpy as np
import torch

from neural_competition.frequency_pressure import (
    advance,
    derivative,
    factor_allocation,
    finite_pressure_transport,
    forward,
    gaussian_weight_factor,
    inputs,
    kernel_change_ledger,
    kernel_transport,
    local_loss_forecasts,
    parity_coefficients,
    pressure_ledger,
    product_basis,
    rate_ledger,
    residual_pressures,
    response_transport,
    step_ledger,
    velocity,
)
from neural_competition.parity import panels


def fixture():
    q = np.array([.4, .2, .07])
    pop, _ = panels(q)
    x = inputs(pop["x"], q)
    rng = np.random.default_rng(101)
    state = (rng.normal(size=(5, 4)), rng.normal(size=5), np.array(.2))
    phi, incidence = product_basis(pop["x"], q)
    return q, pop, x, state, phi, factor_allocation(incidence)


def test_product_basis_and_closed_form_target_including_unbiased_bit():
    for q in [np.array([.4, .2, .07]), np.array([.5, .5, .1])]:
        pop, _ = panels(q)
        phi, _ = product_basis(pop["x"], q)
        np.testing.assert_allclose(phi.T @ (pop["p"][:, None] * phi), np.eye(8), atol=1e-14)
        np.testing.assert_allclose(phi @ parity_coefficients(q), pop["y"], atol=1e-14)


def test_current_residual_pressures_add_and_loss_rate_matches_finite_update():
    _, pop, x, state, phi, allocation = fixture()
    residual = pop["y"] - forward(state, x)
    coefficients, components = residual_pressures(residual, phi, pop["p"], allocation)
    np.testing.assert_allclose(components.sum(0), residual, atol=1e-14)
    np.testing.assert_allclose(np.sum(coefficients**2), np.sum(pop["p"] * residual**2))
    updates = velocity(state, x, components, pop["p"])
    full = velocity(state, x, residual, pop["p"])
    for u, v in zip(updates, full):
        np.testing.assert_allclose(u.sum(0), v, atol=1e-14)
    ledger = pressure_ledger(state, x, pop["y"], pop["p"], phi, allocation)
    np.testing.assert_allclose(ledger["source_gram"].sum(), ledger["rate"], atol=1e-13)
    np.testing.assert_allclose(ledger["modal_rate"].sum(), ledger["rate"], atol=1e-13)
    eps = 1e-7
    next_state = advance(state, full, eps)
    loss = .5 * np.sum(pop["p"] * (pop["y"] - forward(next_state, x))**2)
    np.testing.assert_allclose((ledger["loss"] - loss) / eps, ledger["rate"], rtol=1e-5)


def test_response_transport_and_finite_ledger_retain_nonlinearity():
    _, pop, x, state, phi, allocation = fixture()
    e = pop["y"] - forward(state, x)
    _, pressures = residual_pressures(e, phi, pop["p"], allocation)
    probes = np.sqrt(pop["p"])[:, None] * phi[:, :4]
    strength, transport = response_transport(state, x, pop["p"], pop["p"], probes, pressures, 3)
    v = velocity(state, x, e, pop["p"])
    eps = 1e-7
    plus = response_transport(advance(state, v, eps), x, pop["p"], pop["p"],
                              probes, pressures, 3)[0]
    minus = response_transport(advance(state, v, -eps), x, pop["p"], pop["p"],
                               probes, pressures, 3)[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), transport.sum(0),
                               atol=1e-8, rtol=1e-5)
    assert np.all(strength >= 0) and np.all(strength <= 1)
    next_state = advance(state, v, .3)
    ledger = step_ledger(state, next_state, x, pop["y"], pop["p"], v, .3)
    np.testing.assert_allclose(ledger["actual"], sum(ledger[k] for k in
                               ["linear", "nonlinear", "quadratic"]), atol=1e-14)
    np.testing.assert_allclose(ledger["nonlinear"], ledger["bilinear"] + ledger["gates"],
                               atol=1e-14)
    changes = kernel_change_ledger(state, next_state, x, pop["p"], probes)
    d0, d1 = derivative(state, x, pop["p"]), derivative(next_state, x, pop["p"])
    expected = np.sum((d1.T @ probes)**2, axis=0) - np.sum((d0.T @ probes)**2, axis=0)
    np.testing.assert_allclose(changes["total"], expected, atol=1e-14)
    assert abs(ledger["nonlinear"]) > 1e-6


def test_optimizer_jacobian_and_noise_curvature_against_torch():
    _, pop, x, state, _, _ = fixture()
    w, a, c = state
    m, dimension = w.shape
    metric = np.r_[np.full(m + m * dimension, m), 1]
    theta = torch.tensor(np.r_[a, w.ravel(), c], requires_grad=True)
    xt, p, y = [torch.tensor(v) for v in [x, pop["p"], pop["y"]]]

    def prediction(t):
        return torch.relu(xt @ t[m:-1].reshape(m, dimension).T) @ t[:m] / m + t[-1]

    def loss(t):
        return .5 * (p * (prediction(t) - y)**2).sum()

    j = torch.autograd.functional.jacobian(prediction, theta).detach().numpy()
    h = torch.autograd.functional.hessian(loss, theta).detach().numpy()
    g = torch.autograd.functional.jacobian(loss, theta).detach().numpy()
    np.testing.assert_allclose(derivative(state, x, pop["p"]),
                               np.sqrt(pop["p"])[:, None] * j * np.sqrt(metric), atol=1e-14)
    v = -metric * g
    f = local_loss_forecasts(state, x, pop["y"], pop["p"], .01, 16, [1, 10])
    np.testing.assert_allclose(f["full_curvature"], v @ h @ v, atol=1e-12)
    individual = (pop["y"] - forward(state, x))[:, None] * j * metric
    covariance = (individual.T * pop["p"]) @ individual - np.outer(v, v)
    np.testing.assert_allclose(f["noise_full"], np.sum(h * covariance), atol=1e-12)


def test_loss_rate_transport_sorting_and_exact_endpoint_identity():
    _, pop, x, state, phi, allocation = fixture()
    e = pop["y"] - forward(state, x)
    root_e = np.sqrt(pop["p"]) * e
    s = root_e / np.linalg.norm(root_e)
    _, pressures = residual_pressures(e, phi, pop["p"], allocation)
    strength, transport = kernel_transport(state, x, pop["p"], s[:, None], pressures)
    d = derivative(state, x, pop["p"])
    kernel = d @ d.T
    sorting = 2 * (np.sum((kernel @ s)**2) - strength[0]**2)
    v = velocity(state, x, e, pop["p"])
    eps = 1e-7
    ledger = rate_ledger(state, advance(state, v, eps), x, pop["y"], pop["p"])
    np.testing.assert_allclose((ledger["after"] - ledger["before"]) / eps,
                               transport.sum() - sorting, atol=1e-5, rtol=1e-5)
    assert sorting >= -1e-14
    finite = rate_ledger(state, advance(state, v, .4), x, pop["y"], pop["p"])
    np.testing.assert_allclose(finite["after"] - finite["before"],
                               finite["transport"] + finite["rotation"], atol=1e-14)


def test_gaussian_pressure_preserves_full_shared_minibatch_covariance():
    _, pop, x, state, _, _ = fixture()
    p = pop["p"]
    factor = gaussian_weight_factor(p, 17)
    np.testing.assert_allclose(factor @ factor.T, (np.diag(p) - np.outer(p, p)) / 17,
                               atol=1e-17)
    d = derivative(state, x, p)
    e = pop["y"] - forward(state, x)
    individual = e[:, None] * d / np.sqrt(p)[:, None]
    mean = p @ individual
    expected = ((individual.T * p) @ individual - np.outer(mean, mean)) / 17
    propagated = individual.T @ factor
    np.testing.assert_allclose(propagated @ propagated.T, expected, atol=1e-14)


def test_finite_pressure_transport_includes_all_gate_jumps_and_adds_exactly():
    _, pop, x, state, phi, allocation = fixture()
    e = pop["y"] - forward(state, x)
    _, pressures = residual_pressures(e, phi, pop["p"], allocation)
    components = velocity(state, x, pressures, pop["p"])
    displacement = tuple(3 * v for v in components)
    next_state = tuple(v + dv.sum(0) for v, dv in zip(state, displacement))
    probes = np.sqrt(pop["p"])[:, None] * phi
    result = finite_pressure_transport(state, next_state, x, pop["p"], probes, displacement)
    exact = kernel_change_ledger(state, next_state, x, pop["p"], probes)
    np.testing.assert_allclose(result["total"].sum(0), exact["total"], atol=1e-13)
    assert np.linalg.norm(result["gates"]) > 1e-8
    # Endpoint and path allocations differ when amplitudes move across a jump.
    constant_a = (displacement[0], np.zeros_like(displacement[1]), displacement[2])
    end = (next_state[0], state[1], next_state[2])
    path = finite_pressure_transport(state, end, x, pop["p"], probes, constant_a)
    endpoint = kernel_change_ledger(state, end, x, pop["p"], probes)
    np.testing.assert_allclose(path["gates"].sum(0), endpoint["gates"], atol=1e-13)
