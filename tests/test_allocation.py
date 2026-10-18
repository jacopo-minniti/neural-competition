import importlib
from pathlib import Path

import torch

from neural_competition.allocation import (
    box_minimum,
    linearized_changes,
    objective,
    quadratic_terms,
    uniform_minimum,
    work_statistics,
)
from neural_competition.trajectory import components


def test_identical_useful_copies_do_not_create_allocation_regret():
    e = torch.full((1, 5), -1., dtype=torch.float64)
    for width in [1, 8, 32]:
        ell = torch.full((1, 5, width), .2/width, dtype=torch.float64)
        b, g = quadratic_terms(e, ell)
        uniform = uniform_minimum(b, g)
        alpha, gap, _ = box_minimum(b, g, uniform)
        torch.testing.assert_close(objective(b, g, alpha), torch.tensor([-.18], dtype=e.dtype))
        torch.testing.assert_close(objective(b, g, alpha), objective(b, g, uniform))
        assert gap.max() < 1e-12
        stats = work_statistics(e, ell)
        assert abs(float(stats['positive_work'])-.2) < 1e-12
        assert abs(float(stats['progress'])-.18) < 1e-12


def test_box_solver_known_interior_optimum_and_duality_bound():
    g = torch.tensor([[[2., .5], [.5, 1.]]], dtype=torch.float64)
    optimum = torch.tensor([[.2, .7]], dtype=torch.float64)
    b = -(g @ optimum[:, :, None]).squeeze(-1)
    alpha, gap, _ = box_minimum(b, g, uniform_minimum(b, g))
    error = objective(b, g, alpha)-objective(b, g, optimum)
    assert torch.all(error <= gap + 1e-14)
    torch.testing.assert_close(alpha, optimum, atol=1e-8, rtol=1e-8)


def test_general_subdivision_invariance_and_anti_work():
    rng = torch.Generator().manual_seed(47)
    e = torch.randn(1, 17, generator=rng, dtype=torch.float64)
    ell = torch.randn(1, 17, 3, generator=rng, dtype=torch.float64)
    values = []
    for copies in [1, 3]:
        split = ell.repeat_interleave(copies, dim=-1)/copies
        b, g = quadratic_terms(e, split)
        u = uniform_minimum(b, g)
        a, gap, _ = box_minimum(b, g, u)
        values.append((objective(b, g, u)-objective(b, g, a), work_statistics(e, split)))
        assert gap.max() < 1e-8
    torch.testing.assert_close(values[0][0], values[1][0])
    for key in values[0][1]:
        torch.testing.assert_close(values[0][1][key], values[1][1][key])


def test_linearized_neuron_changes_against_autograd_jvp():
    rng = torch.Generator().manual_seed(12)
    p = torch.randn(2, 4, 5, generator=rng, dtype=torch.float64)
    d = torch.randn(p.shape, generator=rng, dtype=torch.float64)
    x = torch.randn(2, 21, 3, generator=rng, dtype=torch.float64)
    norm = torch.full((2, 1), 2., dtype=torch.float64)
    _, expected = torch.autograd.functional.jvp(lambda q: components(q, x, norm), p, d)
    torch.testing.assert_close(linearized_changes(p, d, x, norm), expected)


def test_work_budget_and_orthogonal_decomposition():
    rng = torch.Generator().manual_seed(78)
    e = torch.randn(2, 37, generator=rng, dtype=torch.float64)
    d = torch.randn(2, 37, 9, generator=rng, dtype=torch.float64)*.02
    s = work_statistics(e, d)
    exact = .5*(e.square() - (e+d.sum(-1)).square()).mean(1)
    torch.testing.assert_close(s['progress'], exact)
    torch.testing.assert_close(s['curvature'], s['parallel_cost']+s['perpendicular_cost'])
    assert torch.all(s['perpendicular_cost'] >= -1e-14)


def test_multiclass_fisher_quadratic_matches_autograd(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/studies'))
    study = importlib.import_module('run_allocation_mnist')
    rng = torch.Generator().manual_seed(718)
    z = torch.randn(13, 3, generator=rng, dtype=torch.float64)
    y = torch.nn.functional.one_hot(torch.arange(13) % 3, 3).double()
    ell = torch.randn(13, 5, 3, generator=rng, dtype=torch.float64)*.1
    zero = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    for ce in [False, True]:
        def loss(alpha):
            logits = z + torch.einsum('nrc,r->nc', ell, alpha)
            return study.loss_rows(logits, y, ce).mean()
        b, gram = study.terms(z, y, ell, ce)
        gradient = torch.autograd.functional.jacobian(loss, zero)
        hessian = torch.autograd.functional.hessian(loss, zero)
        torch.testing.assert_close(b[0], gradient)
        torch.testing.assert_close(gram[0], hessian)


def test_multiclass_linearization_matches_jvp(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/studies'))
    study = importlib.import_module('run_allocation_mnist')
    rng = torch.Generator().manual_seed(819)
    p = [torch.randn(5, 4, generator=rng, dtype=torch.float64),
         torch.randn(5, generator=rng, dtype=torch.float64),
         torch.randn(5, 3, generator=rng, dtype=torch.float64)]
    d = [torch.randn(a.shape, generator=rng, dtype=torch.float64) for a in p]
    x = torch.randn(17, 4, generator=rng, dtype=torch.float64)
    _, expected = torch.autograd.functional.jvp(lambda *a: study.scores(list(a), x),
                                               tuple(p), tuple(d))
    torch.testing.assert_close(study.linearized(p, d, x).sum(1), expected)
