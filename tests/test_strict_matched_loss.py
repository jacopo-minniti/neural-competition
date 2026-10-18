import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/studies"))
    return importlib.import_module("analyze_strict_matched_loss")


def scalar_case(analysis, changes):
    after = torch.tensor([[changes]], dtype=torch.float64)
    before = torch.zeros_like(after)
    upper = torch.triu(torch.ones(1, len(changes), len(changes), dtype=torch.bool), diagonal=1)
    values = analysis.detailed_metrics(before, after, torch.ones(1, 1), upper)[0].numpy()
    return analysis.measured(values, visits=1)


def test_strict_event_is_harmful_marginal_even_when_joint_update_helps(analysis):
    result = scalar_case(analysis, [0.9, 0.3])
    # Direct half-MSE values: L0=.5, Lr=.005, Ls=.245, Lrs=.02.
    # Only adding s after r harms: .02-.005=.015.
    assert result["loss_after"] < result["loss_before"]
    assert result["strong_events"] == 1
    assert result["strict_frequency"] == 0.5
    assert result["strong_harm"] == pytest.approx(0.015)
    assert result["eligible_benefit"] == pytest.approx(0.495 + 0.255)
    assert result["event_benefit"] == pytest.approx(0.255)
    assert result["aggregate_severity"] == pytest.approx(0.015 / 0.75)
    assert result["event_severity"] == pytest.approx(0.015 / 0.255)


def test_useful_overlap_has_zero_frequency_and_undefined_conditional_severity(analysis):
    result = scalar_case(analysis, [0.1, 0.1])
    assert result["penalty"] == pytest.approx(0.01)
    assert result["strict_frequency"] == result["aggregate_severity"] == 0
    assert np.isnan(result["event_severity"])
    assert np.isnan(result["harm_per_event"])


def test_benefit_denominator_excludes_neurons_below_sign_tolerance(analysis):
    result = scalar_case(analysis, [0.9, 0.3, 1e-12])
    assert result["helpful_neurons"] == 2
    assert result["eligible_benefit"] == pytest.approx(0.75, abs=1e-15, rel=0)
    assert result["solo_gain"] > result["eligible_benefit"]


def test_no_eligible_pair_is_undefined_rather_than_zero_frequency(analysis):
    result = scalar_case(analysis, [0.1, -0.1])
    assert result["eligible_ordered"] == 0
    assert np.isnan(result["strict_frequency"])
    assert np.isnan(result["aggregate_severity"])
