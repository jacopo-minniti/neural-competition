"""Independent mathematical checks for finite-update and sampling identities."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_competition.learning_ledger import cross_entropy_ledger, dot, panel_signal


def test_finite_population_correction_is_unbiased_including_cancellation():
    gradients = torch.tensor(
        [[1.0, 2.0], [-1.0, -2.0], [3.0, -1.0], [-3.0, 1.0]], dtype=torch.float64
    )
    for shift in [0.0, 0.7]:
        g = gradients + shift
        truth = g.mean(0).square().sum()
        for size in [2, 3, 4]:
            estimates, raws = [], []
            for ids in itertools.combinations(range(4), size):
                sample = g[list(ids)]
                raw, corrected, _ = panel_signal(
                    [sample.mean(0)], sample.square().sum(1).mean(), size, 4
                )
                estimates.append(float(corrected))
                raws.append(float(raw))
            assert np.mean(estimates) == pytest.approx(float(truth), abs=1e-14)
            expected = truth + (1 / size - 1 / 4) * g.var(0, unbiased=True).sum()
            assert np.mean(raws) == pytest.approx(float(expected), abs=1e-14)
    with pytest.raises(ValueError):
        panel_signal([g[0]], torch.tensor(1.0), 1, 4)


def test_loss_ledger_handles_large_relu_crossings_and_logit_gauge():
    torch.manual_seed(49)
    x = torch.randn(9, 5, dtype=torch.float64)
    y = torch.arange(9) % 3
    w = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    dw, da = torch.randn_like(w) * 2, torch.randn_like(a)

    def function(w, a):
        return (x @ w.T).relu() @ a.T

    old, tangent = torch.autograd.functional.jvp(function, (w, a), (dw, da))
    new = function(w + dw, a + da)
    row = cross_entropy_ledger(old, new, y, tangent)
    assert abs(row["closure"]) < 1e-14
    assert row["cost"] > 0
    assert abs(row["nonlinear"]) > 1e-3
    loss = torch.nn.functional.cross_entropy(function(w, a), y)
    grad = torch.autograd.grad(loss, (w, a))
    assert row["tangent"] == pytest.approx(float(-dot(grad, (dw, da))), abs=1e-14)
    shifted = cross_entropy_ledger(old + 13, new - 21, y, tangent - 34)
    for field in ["gain", "tangent", "nonlinear", "cost"]:
        assert row[field] == pytest.approx(shifted[field], abs=1e-13)


def test_study_analytic_gradients_and_jvp_against_autograd():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/studies"))
    from trace_observed_learning import forward, gradients, tangent

    torch.manual_seed(51)
    parameters = tuple(
        t.requires_grad_()
        for t in [
            torch.randn(4, 7, dtype=torch.float64),
            torch.randn(4, dtype=torch.float64),
            torch.randn(10, 4, dtype=torch.float64),
            torch.randn(10, dtype=torch.float64),
        ]
    )
    x = torch.randn(13, 7, dtype=torch.float64)
    y = torch.arange(13) % 10
    cached = forward(parameters, x)
    g, norms = gradients(parameters, x, y, cached)
    individual = []
    for i in range(len(x)):
        loss = torch.nn.functional.cross_entropy(cached[0][i : i + 1], y[i : i + 1])
        individual.append(torch.autograd.grad(loss, parameters, retain_graph=True))
    for k, actual in enumerate(g):
        torch.testing.assert_close(actual, torch.stack([v[k] for v in individual]).mean(0))
    torch.testing.assert_close(norms, torch.stack([dot(v, v) for v in individual]))
    delta = tuple(torch.randn_like(v) for v in parameters)
    _, ref = torch.autograd.functional.jvp(lambda *s: forward(s, x)[0], parameters, delta)
    torch.testing.assert_close(tangent(parameters, x, cached, delta), ref)
