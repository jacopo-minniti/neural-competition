"""Structural protection, progress clocks, and critical-pressure identities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
from competition_reserve import (  # noqa: E402
    capability,
    common_flow,
    directional_lifetimes,
    exposure_for_gain,
    forward,
    head_prediction,
    protected_reserve,
    task,
    velocity,
)


def test_protected_blocks_do_not_move_under_common_pressure_and_give_operator_floor():
    train, _, audit, _ = task(351, spread=.15)
    rng = np.random.default_rng(352)
    state = (rng.normal(size=(8, train["x"].shape[1])) * .2,
             rng.normal(size=8) * .2, np.array(0.))
    floor, mask = protected_reserve(state, train, audit)
    assert mask.any()
    for _ in range(100):
        v = velocity(state, train, common_only=True)
        np.testing.assert_array_equal(v[0][mask], 0)
        np.testing.assert_array_equal(v[1][mask], 0)
        state = tuple(s + .01 * dv for s, dv in zip(state, v))
        assert np.min(capability(state, audit) - floor) > -1e-12


def test_pressure_completion_inverts_functional_gain_without_using_frequency():
    train, _, audit, _ = task(312)
    rng = np.random.default_rng(313)
    state = (rng.normal(size=(4, train["x"].shape[1])) * .2,
             rng.normal(size=4) * .2, np.array(0.))
    for gain in [.2, 1., 3.]:
        s = exposure_for_gain(state, gain)
        assert 0 < s <= gain
        np.testing.assert_allclose(head_prediction(common_flow(state, s)) - head_prediction(state),
                                   gain, atol=1e-12)
    floor, _ = protected_reserve(state, train, audit)
    for s in np.linspace(0, 4, 15):
        assert np.min(capability(common_flow(state, s), audit) - floor) > -1e-12


def test_lifetime_formula_matches_physical_rare_gate_crossings():
    _, _, _, meta = task(414)
    rng = np.random.default_rng(415)
    w = .2 * rng.normal(size=(4, meta["directions"].shape[1] + 1))
    w[:, 0] = [.1, .2, .3, -.2]
    state = (w, np.full(4, .3), np.array(0.))
    lifetimes = directional_lifetimes(state, meta["directions"], meta["theta"])
    assert np.all(np.isinf(lifetimes[-1]))
    for j in range(3):
        for r in range(24):
            s = lifetimes[j, r]
            if s > 1e-6:
                at = common_flow(state, s)[0][j]
                np.testing.assert_allclose(at[0], np.tan(meta["theta"]) *
                                           abs(at[1:] @ meta["directions"][r]), atol=1e-12)


def test_three_unit_capacity_witness_includes_continuous_correlated_contexts():
    for factors in [24, 32]:
        for spread in [0, .15]:
            train, _, _, meta = task(419, factors=factors, spread=spread)
            assert meta["capacity"] == 3
            assert np.isclose(meta["q"].sum(), .04)
            # The generator independently checks its witness on two panels.
            assert np.all(np.isfinite(train["y"]))
            assert forward((np.zeros((3, train["x"].shape[1])), np.ones(3), 0.), train["x"]).shape == train["y"].shape


def test_last_alternative_law_matches_joint_functional_null_events():
    train, _, audit, meta = task(537)
    rng = np.random.default_rng(538)
    w = .2 * rng.normal(size=(4, train["x"].shape[1]))
    w[:, 0] = rng.uniform(.03, .08, 4)
    state = (w, rng.uniform(.15, .3, 4), np.array(0.))
    closing = directional_lifetimes(state, meta["directions"], meta["theta"]).max(axis=0)
    critical_gain = np.array([head_prediction(common_flow(state, s)) - head_prediction(state)
                              for s in closing])
    assert np.all(np.isfinite(critical_gain))
    # Many correlated contrasts close under the SAME scalar demand. Compare the
    # joint event to actual resolvent capability, not another gate-only formula.
    for gain in np.linspace(.01, critical_gain.max() + .1, 25):
        s = exposure_for_gain(state, gain)
        beta = capability(common_flow(state, s), audit)
        np.testing.assert_array_equal(beta < 1e-11, critical_gain < gain)
    # A genuinely different, head-inactive alternative prevents complete loss.
    w[-1, 0] = -.1
    protected_state = (w, state[1], state[2])
    assert np.all(np.isinf(directional_lifetimes(
        protected_state, meta["directions"], meta["theta"]).max(axis=0)))
    assert np.all(capability(common_flow(protected_state, 3.), audit) > 1e-8)
