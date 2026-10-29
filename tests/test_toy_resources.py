"""Independent checks of the toy resource geometry and intervention invariants."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from audit_toy_option_events import coverage  # noqa: E402
from explore_stable_distinctions import task as previous_task  # noqa: E402
from recycle_toy_resources import recycle  # noqa: E402
from toy_option_resources import (  # noqa: E402
    CASES,
    advance,
    common,
    forward,
    jacobian,
    kernel,
    response,
    task,
    transport_audit,
    velocity,
)


def test_task_extension_preserves_old_toys_and_orthogonal_pressure_split():
    rng = np.random.default_rng(813)
    for name in CASES:
        data = task(name)
        e = rng.normal(size=(3, len(data["x"])))
        c = common(e, data)
        np.testing.assert_allclose(common(c, data), c, atol=1e-14)
        np.testing.assert_allclose(np.sum(data["p"] * c * (e - c), axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(common(data["ref_y"][None], data), 0, atol=1e-14)
        if name in ["discrete", "continuous", "rarer"]:
            config = data["config"]
            x, p, y = previous_task(config["nodes"], config["spread"], config["q"][0])
            for actual, expected in zip([data["x"], data["p"], data["y"]], [x, p, y]):
                np.testing.assert_allclose(actual, expected, atol=1e-14)
        task(name, dense=True)  # Includes the exact bounded-support capacity witness.


def test_optimizer_metric_kernel_and_velocity_against_autodiff():
    rng = np.random.default_rng(814)
    data = task("separate_attribute")
    m, d = 5, data["x"].shape[1]
    state = (rng.normal(size=(1, m, d)), rng.normal(size=(1, m)), np.array([0.3]))
    theta = torch.tensor(np.r_[state[1].ravel(), state[0].ravel(), state[2]], requires_grad=True)
    xt = torch.tensor(data["x"])

    def f(t):
        return torch.relu(xt @ t[m:-1].reshape(m, d).T) @ t[:m] / m + t[-1]

    j = torch.autograd.functional.jacobian(f, theta).detach().numpy()
    metric = np.r_[np.full(m * (d + 1), m), 1]
    expected_d = data["root"][:, None] * j * np.sqrt(metric)
    np.testing.assert_allclose(jacobian(state, data)[0], expected_d, atol=1e-14)
    np.testing.assert_allclose(kernel(state, data)[0], expected_d @ expected_d.T, atol=1e-14)
    loss = 0.5 * (torch.tensor(data["p"]) * (f(theta) - torch.tensor(data["y"])) ** 2).sum()
    loss.backward()
    v = velocity(state, data)
    np.testing.assert_allclose(
        np.r_[v[1].ravel(), v[0].ravel(), v[2]], -metric * theta.grad.numpy(), atol=1e-14
    )


def test_intrinsic_susceptibility_and_filtered_response_derivative():
    rng = np.random.default_rng(815)
    data = task("discrete")
    m, d = 4, 3
    state = (rng.normal(size=(1, m, d)), rng.normal(size=(1, m)), np.array([0.1]))
    metric_root = np.sqrt(np.r_[np.full(m * (d + 1), m), 1])
    beta = torch.tensor(np.r_[state[1].ravel(), state[0].ravel(), state[2]] / metric_root)
    xt, root, mr = map(torch.tensor, [data["x"], data["root"], metric_root])

    def f(b):
        t = b * mr
        return root * (torch.relu(xt @ t[m:-1].reshape(m, d).T) @ t[:m] / m + t[-1])

    dd = torch.autograd.functional.jacobian(f, beta)
    u, v, p = [torch.tensor(rng.normal(size=len(root))) for _ in range(3)]
    hu = torch.autograd.functional.hessian(lambda b: u @ f(b), beta)
    hv = torch.autograd.functional.hessian(lambda b: v @ f(b), beta)
    ru, rv, rp = [dd.T @ q for q in [u, v, p]]
    predicted = (rv @ hu @ rp + ru @ hv @ rp).item()
    eps = 1e-6
    vel = velocity(state, data, p.numpy()[None] / data["root"])
    plus, minus = advance(state, vel, eps), advance(state, vel, -eps)
    dk = (kernel(plus, data)[0] - kernel(minus, data)[0]) / (2 * eps)
    np.testing.assert_allclose(u.numpy() @ dk @ v.numpy(), predicted, rtol=1e-7, atol=1e-8)
    lam = 0.5 / 20
    q = np.linalg.inv(kernel(state, data)[0] + lam * np.eye(len(root)))
    db = (response(plus, data)[0] - response(minus, data)[0]) / (2 * eps)
    np.testing.assert_allclose(db, lam * q @ dk @ q, rtol=1e-5, atol=1e-8)
    audit = transport_audit(state, data, 0.05)
    np.testing.assert_allclose(audit["common"] + audit["refinement"], audit["total"], atol=1e-14)


def test_recycling_preserves_predictions_and_trace_but_not_full_response():
    data = task("discrete")
    # Duplicate features provide an exact, nontrivial readout redistribution.
    state = (
        np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]]),
        np.array([[1.0, 2.0, -1.0, 1.0]]),
        np.array([0.2]),
    )
    original = tuple(v.copy() for v in state)
    result = recycle(state, data, data, np.random.default_rng(816))
    assert result is not None
    control, treatment, meta = result
    for s in [control, treatment]:
        np.testing.assert_allclose(forward(s, data["x"]), forward(state, data["x"]), atol=1e-13)
        assert s[0].shape == state[0].shape
        assert s[1][0, meta["unit"]] == 0
    kc, kt = kernel(control, data)[0], kernel(treatment, data)[0]
    np.testing.assert_allclose(np.trace(kc), np.trace(kt), atol=1e-13)
    assert np.linalg.norm(kc - kt) > 1e-3
    for before, after in zip(original, state):
        np.testing.assert_array_equal(before, after)


def test_latent_coverage_is_not_finite_budget_strength():
    data = task("discrete")
    w = np.array([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    state = (w, np.ones((1, 4)), np.zeros(1))
    tiny = (w * 1e-6, state[1] * 1e-6, state[2])
    np.testing.assert_allclose(coverage(state, data), coverage(tiny, data), atol=1e-12)
    fixed = data["root"] * data["ref_y"]
    fixed /= np.linalg.norm(fixed)
    assert fixed @ response(tiny, data)[0] @ fixed < 1e-8
    assert fixed @ response(state, data)[0] @ fixed > 0.1


def test_loaded_finite_response_bounds_instantaneous_progress():
    rng = np.random.default_rng(817)
    data = task("continuous")
    state = (rng.normal(size=(4, 5, 3)), rng.normal(size=(4, 5)), np.zeros(4))
    e = (data["y"] - forward(state, data["x"])) * data["root"]
    norm = np.sum(e**2, axis=1)
    actual = np.einsum("si,sij,sj->s", e, kernel(state, data), e) / norm
    for tau in [1.0, 20.0, 1000.0]:
        b = np.einsum("si,sij,sj->s", e, response(state, data, tau), e) / norm
        lower = (0.5 / tau) * b / (1 - b)
        assert np.all(actual >= lower - 1e-12)
