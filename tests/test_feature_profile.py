import torch

from neural_competition.feature_profile import diagonal_correction, profile_readout
from neural_competition.training import batch_gradient


def test_profile_decomposition_gradient_and_rank_deficiency():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(841)
    x = torch.randn(45, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(45, generator=rng, dtype=torch.float64)
    p = torch.randn(1, 4, 5, generator=rng, dtype=torch.float64)
    phi = (x @ p[0, :, :3].T + p[0, :, 3]).relu() / 2
    a, loss = profile_readout(phi, y)
    residual = y - phi @ a
    torch.testing.assert_close(phi.T @ residual, torch.zeros(4, dtype=x.dtype), atol=1e-12, rtol=0)
    current = 0.5 * (phi @ p[0, :, -1] - y).square().mean()
    torch.testing.assert_close(current, loss + 0.5 * (phi @ (p[0, :, -1] - a)).square().mean())
    q = p.clone()
    q[0, :, -1] = a
    grad = batch_gradient(q, x[None], y[None], torch.tensor([[2.0]], dtype=x.dtype))
    direction = torch.randn(4, 3, generator=rng, dtype=x.dtype)
    eps = 1e-6

    def profiled(delta):
        features = (x @ (p[0, :, :3] + delta * direction).T + p[0, :, 3]).relu() / 2
        return profile_readout(features, y)[1]

    finite = (profiled(eps) - profiled(-eps)) / (2 * eps)
    torch.testing.assert_close(finite, (grad[0, :, :3] * direction).sum(), atol=1e-8, rtol=1e-6)
    _, duplicated = profile_readout(torch.cat([phi, phi[:, :1]], 1), y)
    torch.testing.assert_close(loss, duplicated)


def test_diagonal_control_decreases_loss_but_need_not_equal_coupled_solve():
    x = torch.tensor([[1.0, 1.0], [1.0, 0.9], [0.0, 0.1], [0.0, 0.0]], dtype=torch.float64)
    y = torch.tensor([1.0, -0.5, 0.2, 0.0], dtype=torch.float64)
    a = torch.zeros(2, dtype=torch.float64)
    changed = diagonal_correction(x, y, a)
    _, optimum = profile_readout(x, y)
    current = 0.5 * (x @ changed - y).square().mean()
    assert optimum < current < 0.5 * y.square().mean()
