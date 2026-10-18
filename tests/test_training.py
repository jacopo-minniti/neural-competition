import math

import pytest
import torch

from neural_competition.training import advance, batch_gradient, components


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("train_bias", [False, True])
def test_shared_scalar_training_matches_pytorch_and_chunked_continuation(kind, train_bias):
    """Check optimizer state, layer rates, padding, and resumed update counts."""
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(812)
    initial = torch.randn(2, 4, 5, generator=rng, dtype=torch.float64) * .2
    initial[0, 2:] = 0
    x = torch.randn(31, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(31, generator=rng, dtype=torch.float64)
    ids = torch.randint(31, (7, 2, 9), generator=rng)
    seeds = torch.tensor([0, 1])
    norm = torch.tensor([[math.sqrt(2)], [2.]], dtype=torch.float64)
    rates = torch.ones(2, 1, 5, dtype=torch.float64)
    rates[:, :, -1] = .4
    first, second = torch.zeros_like(initial), torch.zeros_like(initial)
    actual = advance(initial, first, second, x, y, ids, seeds, norm, rates,
                     .003, kind, train_bias, 0)
    prefix = advance(initial, first, second, x, y, ids[:3], seeds, norm, rates,
                     .003, kind, train_bias, 0)
    resumed = advance(*prefix, x, y, ids[3:], seeds, norm, rates,
                      .003, kind, train_bias, 3)
    for whole, chunks in zip(actual, resumed):
        torch.testing.assert_close(whole, chunks, atol=0, rtol=0)
    assert torch.count_nonzero(actual[0][0, 2:]) == 0
    for model, width in enumerate([2, 4]):
        w = torch.nn.Parameter(initial[model, :width, :3].clone())
        b = torch.nn.Parameter(initial[model, :width, 3].clone(), requires_grad=train_bias)
        a = torch.nn.Parameter(initial[model, :width, -1].clone())
        groups = [{"params": [w, b], "lr": .003}, {"params": [a], "lr": .0012}]
        optimizer = (torch.optim.Adam(groups) if kind == 2 else
                     torch.optim.SGD(groups, momentum=.9 if kind == 1 else 0))
        for step in range(len(ids)):
            batch = ids[step, model]
            prediction = (x[batch] @ w.T + b).relu() @ a / math.sqrt(width)
            objective = .5 * (prediction - y[batch]).square().mean()
            objective.backward()
            optimizer.step()
            optimizer.zero_grad()
        expected = torch.cat([w, b[:, None], a[:, None]], dim=-1)
        torch.testing.assert_close(actual[0][model, :width], expected, atol=1e-14, rtol=1e-12)


def test_analytic_scalar_gradient_matches_autograd():
    rng = torch.Generator().manual_seed(813)
    p = torch.randn(2, 4, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 17, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 17, generator=rng, dtype=torch.float64)
    norm = torch.tensor([[2.], [2.]], dtype=torch.float64)
    objective = .5 * (components(p, x, norm).sum(-1) - y).square().mean(1).sum()
    expected, = torch.autograd.grad(objective, p)
    torch.testing.assert_close(batch_gradient(p.detach(), x, y, norm), expected)
