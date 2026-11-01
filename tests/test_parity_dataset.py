import numpy as np
import pytest

from neural_competition.parity import frequencies, panels, sample


def test_exact_panel_weights_labels_and_audit():
    q = frequencies()
    population, audit = panels(q)
    assert population["x"].shape == (256, 8)
    np.testing.assert_allclose(population["p"].sum(), 1)
    np.testing.assert_allclose(population["p"] @ population["x"], q)
    np.testing.assert_array_equal(population["y"], population["x"].sum(1) % 2)
    np.testing.assert_allclose(population["p"] @ population["y"], (1 - np.prod(1 - 2 * q)) / 2)
    np.testing.assert_allclose(audit["p"], 1 / 256)
    _, other_audit = panels(frequencies(alpha=0))
    for key in audit:
        np.testing.assert_array_equal(audit[key], other_audit[key])


def test_sample_reproducible_and_matches_marginals():
    q = frequencies()
    data = sample(q, 100000, np.random.default_rng(42))
    again = sample(q, 100000, np.random.default_rng(42))
    np.testing.assert_array_equal(data["x"], again["x"])
    np.testing.assert_array_equal(data["y"], data["x"].sum(1) % 2)
    assert np.all(np.abs(data["x"].mean(0) - q) < .01)


def test_invalid_probabilities_rejected():
    for q in ([0, .5], [1], [float("nan")], []):
        with pytest.raises(ValueError):
            panels(q)
