import math

import pytest
import torch

from neural_competition.mnist import (
    advance,
    batch_transfer,
    components,
    gradients,
    initialize,
    interaction_statistics,
    loss_rows,
    propose,
    scores,
)


@pytest.mark.parametrize("ce", [False, True])
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mnist_manual_gradients_and_optimizer_match_pytorch(ce, kind, dtype):
    torch.set_num_threads(1)
    p = initialize(4, 3, input_dim=5, classes=3, dtype=dtype)
    actual = [torch.nn.Parameter(v.clone()) for v in p]
    optimizer = (torch.optim.Adam(actual, lr=0.003) if kind == 2
                 else torch.optim.SGD(actual, lr=0.003, momentum=0.9 if kind == 1 else 0))
    first, second = [torch.zeros_like(v) for v in p], [torch.zeros_like(v) for v in p]
    rng = torch.Generator().manual_seed(8)
    for step in range(5):
        x = torch.randn(11, 5, generator=rng, dtype=dtype)
        y = torch.nn.functional.one_hot(torch.randint(3, (11,), generator=rng), 3).to(dtype)
        loss_rows(scores(actual, x), y, ce).mean().backward()
        g = gradients(p, x, y, ce)
        for measured, reference in zip(g, actual):
            torch.testing.assert_close(measured, reference.grad,
                                       rtol=2e-5 if dtype == torch.float32 else 1e-12,
                                       atol=1e-7 if dtype == torch.float32 else 1e-14)
        p, first, second = propose(p, g, first, second, kind, 0.003, step)
        optimizer.step()
        optimizer.zero_grad()
        for measured, reference in zip(p, actual):
            torch.testing.assert_close(measured, reference,
                                       rtol=2e-5 if dtype == torch.float32 else 1e-12,
                                       atol=1e-7 if dtype == torch.float32 else 1e-14)


def test_vector_mse_exact_budget_and_four_loss_pairs_with_gate_crossings():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(67)
    p = initialize(5, 9, input_dim=4, classes=3, dtype=torch.float64)
    q = [v + torch.randn(v.shape, generator=rng, dtype=v.dtype) for v in p]
    x = torch.randn(13, 4, generator=rng, dtype=torch.float64)
    y = torch.randn(13, 3, generator=rng, dtype=torch.float64)
    stats, _, (i, j, pairs) = interaction_statistics(p, q, x, y, False, rng)
    assert stats["pair_identity_error"] < 1e-12
    assert stats["logit_additivity_error"] < 1e-12
    baseline = loss_rows(scores(p, x), y, False).mean()
    for a, b, pair in zip(i, j, pairs):
        losses = []
        for neurons in [[a], [b], [a, b]]:
            moved = [v.clone() for v in p]
            for target, source in zip(moved, q):
                ids = torch.tensor(neurons)
                target[ids] = source[ids]
            losses.append(loss_rows(scores(moved, x), y, False).mean())
        torch.testing.assert_close(losses[2] - losses[0] - losses[1] + baseline, pair)
    torch.testing.assert_close(components(q, x).sum(1), scores(q, x))


def test_ce_pairs_exact_but_do_not_close_whole_update_budget():
    rng = torch.Generator().manual_seed(9)
    p = initialize(4, 4, input_dim=3, classes=3, dtype=torch.float64)
    q = [v + torch.randn(v.shape, generator=rng, dtype=v.dtype) for v in p]
    x = torch.randn(8, 3, generator=rng, dtype=torch.float64)
    y = torch.nn.functional.one_hot(torch.arange(8) % 3, 3).double()
    stats, alone, (i, j, pairs) = interaction_statistics(p, q, x, y, True, rng,
                                                       pair_samples=100)
    assert stats["budget_error"] < 1e-12
    assert abs(stats["net_all"] - stats["pair_net"]) > 1e-4
    baseline = loss_rows(scores(p, x), y, True).mean()
    for a, b, pair in zip(i, j, pairs):
        moved = [v.clone() for v in p]
        for target, source in zip(moved, q):
            ids = torch.tensor([a, b])
            target[ids] = source[ids]
        joint = loss_rows(scores(moved, x), y, True).mean()
        torch.testing.assert_close(joint - baseline - alone[a] - alone[b], pair)


def test_pure_advance_is_repeatable_and_does_not_mutate_starting_state():
    torch.set_num_threads(1)
    p = initialize(4, 1, input_dim=3, classes=2)
    originals = [v.clone() for v in p]
    first, second = [torch.zeros_like(v) for v in p], [torch.zeros_like(v) for v in p]
    rng = torch.Generator().manual_seed(23)
    x = torch.randn(15, 3, generator=rng)
    y = torch.nn.functional.one_hot(torch.arange(15) % 2, 2).float()
    ids = torch.randint(15, (7, 5), generator=rng)
    result = advance(p, first, second, x, y, ids, True, 2, 0.001, 0)
    repeated = advance(p, first, second, x, y, ids, True, 2, 0.001, 0)
    for a, b in zip(result, repeated):
        for c, d in zip(a, b):
            assert torch.equal(c, d)
    for a, b in zip(p, originals):
        assert torch.equal(a, b)
    assert all(torch.count_nonzero(v) == 0 for v in first + second)
    assert math.isfinite(float(loss_rows(scores(result[0], x), y, True).mean()))


def test_adam_transfer_keeps_common_prior_moments_and_recovers_global_scores():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(33)
    p = initialize(4, 7, input_dim=5, classes=3)
    first = [torch.randn(value.shape, generator=rng) * 0.01 for value in p]
    second = [torch.rand(value.shape, generator=rng) * 0.01 for value in p]
    originals = [value.clone() for value in p + first + second]
    x = torch.randn(20, 5, generator=rng)
    y = torch.nn.functional.one_hot(torch.arange(20) % 3, 3).float()
    kwargs = dict(ce=True, kind=2, lr=0.003, completed=9, seed=33, probes=8, batch_size=4)
    measured = batch_transfer(p, first, second, x, y, **kwargs)
    repeated = batch_transfer(p, first, second, x, y, **kwargs)
    assert measured == repeated
    assert measured["attribution_error"] < 1e-12
    assert abs(measured["net_first_order_gain"] - measured["helpful_gain"]
               + measured["harmful_gain"]) < 1e-12
    for value, original in zip(p + first + second, originals):
        assert torch.equal(value, original)
