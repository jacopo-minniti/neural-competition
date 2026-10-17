import math

import torch

from neural_competition.model import ShallowReLU, loss
from neural_competition.trajectory import components, interaction_metrics, momentum_step


def test_batched_momentum_matches_autograd_including_padding():
    torch.set_num_threads(1)
    models = [ShallowReLU(3, width, 12, zero_output=False) for width in [2, 4]]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
    p = torch.zeros(2, 4, 5)
    for i, model in enumerate(models):
        p[i, : model.width] = model.parameter_blocks()
    v = torch.zeros_like(p)
    norm = torch.tensor([[math.sqrt(2)], [2.0]])
    rng = torch.Generator().manual_seed(91)
    for _ in range(8):
        x = torch.randn(2, 7, 3, generator=rng)
        y = torch.randn(2, 7, generator=rng)
        p, v = momentum_step(p, v, x, y, norm, 0.1, 0.9)
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            loss(model, x[i], y[i]).backward()
            optimizer.step()
            torch.testing.assert_close(
                p[i, : model.width], model.parameter_blocks(), atol=2e-7, rtol=2e-6
            )
    assert torch.count_nonzero(p[0, 2:]) == 0


def test_all_pair_accounting_telescopes_through_different_updates():
    rng = torch.Generator().manual_seed(88)
    x = torch.randn(2, 19, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 19, generator=rng, dtype=torch.float64)
    p = torch.randn(2, 4, 5, generator=rng, dtype=torch.float64)
    norm = torch.full((2, 1), 2.0, dtype=torch.float64)
    upper = torch.triu(torch.ones(2, 4, 4, dtype=torch.bool), diagonal=1)
    initial = 0.5 * (components(p, x, norm).sum(-1) - y).square().mean(1)
    total = torch.zeros(2, dtype=torch.float64)
    for _ in range(6):
        q = p + 0.3 * torch.randn(2, 4, 5, generator=rng, dtype=torch.float64)
        stats = interaction_metrics(components(p, x, norm), components(q, x, norm), y, upper)
        reconstructed = stats[:, 3] - stats[:, 2] + stats[:, 4] - stats[:, 5]
        torch.testing.assert_close(reconstructed, stats[:, 1] - stats[:, 0])
        total += reconstructed
        p = q
    final = 0.5 * (components(p, x, norm).sum(-1) - y).square().mean(1)
    torch.testing.assert_close(total, final - initial)


def test_broad_penalty_counts_diminishing_returns_without_reversal():
    before = torch.zeros(1, 1, 2, dtype=torch.float64)
    after = torch.full_like(before, 0.4)
    y = torch.ones(1, 1, dtype=torch.float64)
    upper = torch.tensor([[[False, True], [False, False]]])
    result = interaction_metrics(before, after, y, upper)[0]
    assert result[6] == 1 and result[10] == 0
    torch.testing.assert_close(result[4] / result[2], torch.tensor(0.25, dtype=torch.float64))
