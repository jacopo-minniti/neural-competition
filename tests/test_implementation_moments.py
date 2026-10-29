"""Independent invariants for the functional resource state."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from implementation_moments import (
    hierarchy,
    mix_within_gate,
    moment_kernel,
    moment_output,
    moment_step,
    moments,
)  # noqa: E402
from toy_option_resources import advance, forward, kernel, velocity  # noqa: E402


def test_hierarchy_targets_and_frequency_weights():
    for n in [1, 2, 4, 8]:
        for ratio in [1.0, 0.25, 0.1]:
            for overlap in [0.0, 0.5, 1.0]:
                d = hierarchy(n, ratio, overlap)
                np.testing.assert_allclose(d["q"][1:] / d["q"][:-1], ratio)
                np.testing.assert_allclose(d["q"].sum(), 0.2)
                np.testing.assert_allclose(
                    [d["p"][d["context"] == j + 1].sum() for j in range(n)], d["q"]
                )
                assert d["capacity"] == 2 + 2 * n


def test_moments_reconstruct_and_evolve_exactly_without_crossings():
    rng = np.random.default_rng(9911)
    data = hierarchy(2, 0.25, 0.0)
    state = (rng.normal(size=(1, 8, 5)), rng.normal(size=(1, 8)), np.array([0.3]))
    resource = moments(state, data)
    np.testing.assert_allclose(
        moment_output(resource, data), data["root"] * forward(state, data["x"])[0], atol=1e-13
    )
    np.testing.assert_allclose(moment_kernel(resource, data), kernel(state, data)[0], atol=1e-13)
    dt = 1e-5
    after = advance(state, velocity(state, data), dt)
    assert np.array_equal(state[0] @ data["x"].T > 0, after[0] @ data["x"].T > 0)
    evolved = moment_step(resource, data, dt)
    np.testing.assert_allclose(
        moment_output(evolved, data), data["root"] * forward(after, data["x"])[0], atol=1e-13
    )
    np.testing.assert_allclose(moment_kernel(evolved, data), kernel(after, data)[0], atol=1e-13)


def test_functional_state_ignores_input_rotations_and_unit_order():
    rng = np.random.default_rng(9912)
    data = hierarchy(2, 0.25, 1.0)
    state = (rng.normal(size=(1, 12, 5)), rng.normal(size=(1, 12)), np.zeros(1))
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    rotated = dict(data, x=data["x"] @ rotation)
    transformed = (state[0] @ rotation, state[1], state[2])
    for a, b in zip(moments(state, data)["groups"], moments(transformed, rotated)["groups"]):
        for k in ["s", "q", "R"]:
            np.testing.assert_allclose(a[k], b[k], atol=1e-13)
    perm = rng.permutation(12)
    shuffled = (state[0][:, perm], state[1][:, perm], state[2])
    for a, b in zip(moments(state, data)["groups"], moments(shuffled, data)["groups"]):
        for k in ["s", "q", "R"]:
            np.testing.assert_allclose(a[k], b[k], atol=1e-13)


def test_gate_mixing_preserves_full_moments_function_and_options():
    rng = np.random.default_rng(9913)
    data = hierarchy(1, 1.0, 1.0)
    w = rng.normal(size=(1, 32, 4))
    state = (w, rng.normal(size=(1, 32)), np.zeros(1))
    result = mix_within_gate(state, data, rng)
    assert result is not None
    mixed, _ = result
    for a, b in zip(moments(state, data)["groups"], moments(mixed, data)["groups"]):
        for k in ["s", "q", "R"]:
            np.testing.assert_allclose(a[k], b[k], atol=1e-13)
    np.testing.assert_allclose(forward(state, data["x"]), forward(mixed, data["x"]), atol=1e-13)
    np.testing.assert_allclose(kernel(state, data), kernel(mixed, data), atol=1e-13)


def test_equal_moments_do_not_determine_boundary_flux():
    # One gate group, identical full moments/function/K, different next gates.
    from implementation_moments import functional_particles, particle_output, particle_step

    data = dict(
        x=np.eye(2),
        p=np.ones(2) / 2,
        root=np.ones(2) / np.sqrt(2),
        gram=np.eye(2) / 2,
        y=np.array([1.5, -1.5]),
    )
    state = (np.array([[[1.0, 2.0], [2.0, 1.0]]]), np.array([[-1.0, 1.0]]), np.zeros(1))
    t = 0.4
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    mixed = ((rotation @ state[0][0])[None], (rotation @ state[1][0])[None], np.zeros(1))
    r1, r2 = moments(state, data), moments(mixed, data)
    for k in ["s", "q", "R"]:
        np.testing.assert_allclose(r1["groups"][0][k], r2["groups"][0][k], atol=1e-14)
    np.testing.assert_allclose(forward(state, data["x"]), forward(mixed, data["x"]), atol=1e-14)
    np.testing.assert_allclose(kernel(state, data), kernel(mixed, data), atol=1e-14)
    future = [advance(s, velocity(s, data), 0.5) for s in [state, mixed]]
    assert np.all(future[0][0] > 0) and np.any(future[1][0] < 0)
    assert np.linalg.norm(forward(future[0], data["x"]) - forward(future[1], data["x"])) > 0.01
    for s, f in zip([state, mixed], future):
        actual = particle_step(functional_particles(s, data), data, 0.5)
        np.testing.assert_allclose(
            particle_output(actual, data), data["root"] * forward(f, data["x"]), atol=1e-14
        )


def test_identical_resource_measure_makes_mean_field_duplication_inert():
    rng = np.random.default_rng(9914)
    data = hierarchy(2, 0.25, 0.0)
    original = (0.2 * rng.normal(size=(1, 6, 5)), 0.2 * rng.normal(size=(1, 6)), np.zeros(1))
    duplicate = (
        np.repeat(original[0], 2, axis=1),
        np.repeat(original[1], 2, axis=1),
        original[2].copy(),
    )
    for _ in range(100):
        original = advance(original, velocity(original, data), 0.05)
        duplicate = advance(duplicate, velocity(duplicate, data), 0.05)
    np.testing.assert_allclose(
        forward(original, data["x"]), forward(duplicate, data["x"]), atol=1e-12
    )
    np.testing.assert_allclose(kernel(original, data), kernel(duplicate, data), atol=1e-12)
