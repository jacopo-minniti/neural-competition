"""Capacity, association, exposure moments, and response susceptibility checks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from power_law_tasks import (  # noqa: E402
    dataset,
    forward,
    frozen_forecast,
    influence,
    initialize,
    latent_draw,
    local_forecast,
    response,
    sgd_step,
    shared_exposure_closure,
    specification,
    velocity,
)


def test_bounded_continuous_and_cooccurring_tasks_have_exact_capacity_witness():
    for richer in [False, True]:
        spec = specification(richer=richer)
        train, val, audit, witness = dataset(spec, 8192, train_count=256)
        for panel in [train, val, audit]:
            np.testing.assert_allclose(forward(witness, panel["x"]), panel["y"], atol=1e-12)
        assert spec["capacity"] == 50 and audit["directions"].shape == (96, 24)
        np.testing.assert_allclose(np.linalg.norm(audit["directions"], axis=0), 1)
        np.testing.assert_allclose(audit["directions"].sum(0), 0, atol=1e-12)
        if richer:
            assert np.max(train["indicator"].sum(1)) > 1


def test_association_preserves_power_law_marginals():
    spec = specification(richer=True)
    indicator, _ = latent_draw(spec, 200000, np.random.default_rng(8328))
    error = np.abs(indicator.mean(0) - spec["q"])
    se = np.sqrt(spec["q"] * (1 - spec["q"]) / len(indicator))
    assert np.all(error < 6 * se)
    # Same-group dependence from the mixture construction has a known covariance.
    i, j = 0, 4
    expected = spec["association"] * (min(spec["q"][i], spec["q"][j])
                                      - spec["q"][i] * spec["q"][j])
    observed = np.mean(indicator[:, i] * indicator[:, j]) - indicator[:, i].mean() * indicator[:, j].mean()
    np.testing.assert_allclose(observed, expected, atol=.002)


def test_smooth_susceptibility_matches_independent_finite_difference():
    spec = specification()
    train, _, audit, _ = dataset(spec, 8193)
    state = initialize(50, train["x"].shape[1], "fanin_uniform", 8171)
    inf = influence(state, train, audit)
    v = velocity(state, train["x"], train["y"], train["p"])
    eps = 1e-7
    plus = tuple(s + eps * dv for s, dv in zip(state, v))
    minus = tuple(s - eps * dv for s, dv in zip(state, v))
    numeric = (response(plus, audit) - response(minus, audit)) / (2 * eps)
    np.testing.assert_allclose(numeric, inf["mean"], atol=1e-7, rtol=2e-5)
    assert np.linalg.eigvalsh(inf["covariance"]).min() > -1e-12
    assert np.all((inf["diversity"] >= 1 - 1e-12) & (inf["diversity"] <= 50 + 1e-12))


def test_frozen_aggregate_is_exact_sum_of_checkpoint_sample_gradients():
    spec = specification()
    train, _, _, _ = dataset(spec, 883)
    state = initialize(50, train["x"].shape[1], "fanin_gaussian", 883)
    n, rate = 12, .01
    seed = 1993
    count = np.random.default_rng(seed).multinomial(n, train["p"], size=1)[0]
    ids = np.repeat(np.arange(len(count)), count)
    individual = [velocity(state, train["x"][i:i+1], train["y"][i:i+1], 1.) for i in ids]
    expected = tuple(s + rate / 3 * np.sum([v[k] for v in individual], axis=0)
                     for k, s in enumerate(state))
    actual = frozen_forecast(state, train, 4, rate, 3, 1, np.random.default_rng(seed))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a[0], b, atol=1e-14)


def test_local_models_equal_actual_first_step_for_identical_exposure():
    train, _, _, _ = dataset(specification(richer=True), 7721, train_count=256)
    state = initialize(50, train["x"].shape[1], "fanin_gaussian", 887)
    replicas = tuple(np.broadcast_to(s, (4,) + s.shape).copy() for s in state)
    actual = sgd_step(replicas, train, .01, 16, np.random.default_rng(1231))
    for bilinear in [False, True]:
        predicted = local_forecast(state, train, 1, .01, 16, 4,
                                   np.random.default_rng(1231), bilinear=bilinear)
        for a, b in zip(actual, predicted):
            np.testing.assert_allclose(a, b, atol=1e-14)


def test_shared_count_closure_is_not_independent_feature_survival():
    # Enumerate all eight exposure sequences independently of the binomial formula.
    import itertools

    q, rate = .2, .1
    margins = np.array([.05, .08])
    opening, closing = np.array([2., 1.]), np.array([1., 1.])
    expected = 0.
    for sequence in itertools.product([0, 1], repeat=3):
        margin = margins.copy()
        probability = 1.
        for rare in sequence:
            margin += rate * (opening if rare else -closing)
            probability *= q if rare else 1 - q
        if np.all(margin <= 0):
            expected += probability
    actual = shared_exposure_closure(margins, opening, closing, q, 3, 1, rate)
    np.testing.assert_allclose(actual, expected)
    cloned = shared_exposure_closure(np.tile(margins, 4), np.tile(opening, 4),
                                     np.tile(closing, 4), q, 3, 1, rate)
    np.testing.assert_allclose(actual, cloned)
    protected = shared_exposure_closure(np.r_[margins, 1.], np.r_[opening, 1.],
                                        np.r_[closing, .1], q, 3, 1, rate)
    assert protected == 0 < actual
