import math

import torch

from neural_competition.model import ShallowReLU, loss
from neural_competition.trajectory import adam_step, run_adam_block


def test_batched_adam_matches_torch_parameters_and_moments():
    torch.set_num_threads(1)
    models = [ShallowReLU(3, width, 18) for width in [2, 4]]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.003, foreach=False) for m in models]
    p = torch.zeros(2, 4, 5)
    for i, model in enumerate(models):
        p[i, : model.width] = model.parameter_blocks()
    first, second = torch.zeros_like(p), torch.zeros_like(p)
    norm = torch.tensor([[math.sqrt(2)], [2.0]])
    rng = torch.Generator().manual_seed(12)
    for step in range(1, 101):
        x = torch.randn(2, 11, 3, generator=rng)
        y = torch.randn(2, 11, generator=rng)
        p, first, second = adam_step(p, first, second, x, y, norm, 0.003, step)
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            loss(model, x[i], y[i]).backward()
            optimizer.step()
            torch.testing.assert_close(
                p[i, : model.width], model.parameter_blocks(), atol=5e-7, rtol=1e-5
            )
            for key, expected in [("exp_avg", first), ("exp_avg_sq", second)]:
                blocks = model.blocks(tuple(optimizer.state[v][key] for v in model.parameters()))
                torch.testing.assert_close(expected[i, : model.width], blocks, atol=2e-7, rtol=1e-5)
    assert torch.count_nonzero(p[0, 2:]) == 0


def test_adam_block_continuation_preserves_bias_correction_and_metrics():
    rng = torch.Generator().manual_seed(19)
    p = torch.randn(2, 4, 5, generator=rng) * 0.2
    x = torch.randn(40, 3, generator=rng)
    y = torch.randn(40, generator=rng)
    indices = torch.randint(40, (30, 2, 7), generator=rng)
    seed_map = torch.arange(2)
    widths = torch.full((2, 1), 4)
    xr = x[:9].double()[None].expand(2, -1, -1)
    yr = y[:9].double()[None].expand(2, -1)
    upper = torch.triu(torch.ones(2, 4, 4, dtype=torch.bool), 1)
    tail = (seed_map, widths, xr, yr, upper, 0.003)
    all_p, all_m, all_v, all_stats = run_adam_block(
        p, torch.zeros_like(p), torch.zeros_like(p), x, y, indices, *tail, 0
    )
    mid_p, mid_m, mid_v, early = run_adam_block(
        p, torch.zeros_like(p), torch.zeros_like(p), x, y, indices[:11], *tail, 0
    )
    end_p, end_m, end_v, late = run_adam_block(mid_p, mid_m, mid_v, x, y, indices[11:], *tail, 11)
    for actual, expected in [
        (end_p, all_p),
        (end_m, all_m),
        (end_v, all_v),
        (torch.cat([early, late]), all_stats),
    ]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
