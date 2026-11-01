"""The bridge changes one data assumption while retaining the reserve anchor."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from competition_reserve import task as original_task  # noqa: E402
from reserve_bridge import factor_response_power, forward, task  # noqa: E402


def test_exclusive_endpoint_matches_original_population_and_fixed_audit():
    original, _, audit, _ = original_task(0, factors=4, total_rare=.04)
    bridge, fixed, _, _ = task(cooccurrence=0)
    for key in ["x", "p", "directions"]:
        np.testing.assert_allclose(fixed[key], audit[key], atol=1e-14)
    for x, y, p in zip(original["x"], original["y"], original["p"]):
        matches = np.all(np.isclose(bridge["x"], x), axis=1)
        assert matches.sum() == 1
        np.testing.assert_allclose(bridge["y"][matches], y, atol=1e-14)
        np.testing.assert_allclose(bridge["p"][matches], p, atol=1e-14)


def test_marginals_orthogonal_factor_demands_and_capacity_survive_cooccurrence():
    for mix in [0, .5, 1]:
        for spread in [0, .1]:
            data, _, functional, meta = task(cooccurrence=mix, spread=spread, seed=22)
            p, z, q = data["p"], meta["z"], meta["q"]
            np.testing.assert_allclose(p.sum(), 1)
            np.testing.assert_allclose(p @ (z != 0), q)
            phi = z / np.sqrt(q)
            np.testing.assert_allclose(phi.T @ (p[:, None] * phi), np.eye(4), atol=1e-14)
            np.testing.assert_allclose((p * data["y"]) @ phi,
                                       (-1.)**np.arange(4) * np.sqrt(q), atol=1e-14)
            np.testing.assert_allclose(forward(meta["witness"], data["x"]), data["y"], atol=1e-14)
            np.testing.assert_allclose(np.sum(functional["directions"]**2, axis=0), 1)


def test_global_closure_gives_the_factor_loss_floor():
    data, _, functional, meta = task(cooccurrence=1)
    w = np.zeros((4, data["x"].shape[1]))
    w[:, 0] = 1
    state = (w, np.ones(4), np.array(0.))
    power = factor_response_power(state, data, functional)
    np.testing.assert_array_equal(power, np.zeros(4))
    loss = .5 * np.sum(data["p"] * (data["y"] - forward(state, data["x"]))**2)
    assert loss >= .5 * meta["q"].sum()


def test_association_changes_overlap_without_changing_marginals_or_target_demands():
    base, _, _, initial = task(association=0)
    associated, _, functional, meta = task(association=1)
    p, z, q = associated["p"], meta["z"], meta["q"]
    np.testing.assert_allclose(p @ (z != 0), q, atol=1e-14)
    np.testing.assert_allclose((z / np.sqrt(q)).T @ (p[:, None] * z / np.sqrt(q)),
                               np.eye(4), atol=1e-14)
    assert meta["cooccurrence_mass"] > 10 * initial["cooccurrence_mass"]
    np.testing.assert_allclose(associated["x"], base["x"])
    np.testing.assert_allclose(associated["y"], base["y"])
    np.testing.assert_allclose(np.sum(functional["directions"]**2, axis=0), 1)
