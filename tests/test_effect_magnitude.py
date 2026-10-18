import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def magnitude(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/studies"))
    return importlib.import_module("analyze_effect_magnitude")


def test_total_step_budget_survives_splitting_a_useful_correction(magnitude):
    # Residual -1, total correction .2, represented by either two or four copies.
    # The true gain is .18 in both cases, while C+ grows from .01 to .015.
    frame = pd.DataFrame({
        "gain": [.19, .195], "harm": [0., 0.], "penalty": [.01, .015],
        "credit": [0., 0.], "net": [.01, .015], "penalty_per_energy": [1., 3.],
        "initial_loss": [.5, .5], "final_loss": [.32, .32],
    })
    result = magnitude.objective_budget(frame)
    np.testing.assert_allclose(result.linear_work, [.2, .2], atol=1e-15)
    np.testing.assert_allclose(result.quadratic_cost, [.02, .02], atol=1e-15)
    np.testing.assert_allclose(result.progress, [.18, .18], atol=1e-15)
    assert result.penalty.iloc[1] > result.penalty.iloc[0]


def test_forecast_never_fits_held_out_outcomes(magnitude):
    groups = np.repeat(np.eye(2), 5, axis=0)
    x = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4.])[:, None]
    y = 2*x[:, 0] + groups @ np.array([3., -2.])
    test = np.array([False]*4 + [True] + [False]*4 + [True])
    train = ~test
    original = magnitude.fit_predict(groups, x, y, train, test)
    changed = y.copy()
    changed[test] += 1e9
    repeated = magnitude.fit_predict(groups, x, changed, train, test)
    np.testing.assert_allclose(original, [11., 6.], atol=1e-12)
    np.testing.assert_array_equal(original, repeated)


def test_anti_residual_work_bound_includes_an_explicit_opposing_neuron(magnitude):
    # Residual -1; corrections .3, -.1, .1. The second neuron supplies exactly
    # .1 units of anti-residual linear work. Joint correction .3 gains .255.
    frame = pd.DataFrame({
        "gain": [.35], "harm": [.105], "penalty": [.03], "credit": [.04],
        "net": [-.01], "penalty_per_energy": [.03/.055],
        "initial_loss": [.5], "final_loss": [.245],
    })
    result = magnitude.objective_budget(frame).iloc[0]
    assert result.antiwork_lower <= .1 <= result.antiwork_upper
    assert result.anti_fraction_lower <= .1/.4 <= result.anti_fraction_upper
    assert result.linear_work == pytest.approx(.3)
    assert result.quadratic_cost == pytest.approx(.045)
