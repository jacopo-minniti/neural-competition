import torch

from neural_competition.analysis import neuron_prediction_changes


def test_relu_prediction_expansion_and_activation_crossings():
    rng = torch.Generator().manual_seed(41)
    x = torch.randn(12, 2, generator=rng, dtype=torch.float64) * 0.1
    before = torch.randn(3, 4, generator=rng, dtype=torch.float64)
    before[:, 2] = 2
    after = before + 0.01 * torch.randn(3, 4, generator=rng, dtype=torch.float64)
    s = neuron_prediction_changes({"parameters_before": before, "parameters_after": after}, x)
    assert not s["gate_changed"].any()
    torch.testing.assert_close(
        s["finite"], s["linear"] + s["quadratic_same_region"], atol=1e-15, rtol=1e-12
    )

    # A crossed ReLU gate contributes a finite change not captured by the
    # same-region bilinear term; keep it explicitly marked, not mislabelled.
    before = torch.tensor([[0.0, -0.1, 1.0]], dtype=torch.float64)
    after = torch.tensor([[0.0, 0.1, 1.0]], dtype=torch.float64)
    s = neuron_prediction_changes(
        {"parameters_before": before, "parameters_after": after}, torch.ones(1, 1)
    )
    assert s["gate_changed"].item()
    torch.testing.assert_close(s["finite"], torch.tensor([[0.1]], dtype=torch.float64))
    assert s["linear"].item() == 0 and s["quadratic_same_region"].item() == 0


def test_finite_step_quadratic_matches_autograd():
    from neural_competition.distributions import finite_step_scales
    from neural_competition.model import ShallowReLU, loss

    model = ShallowReLU(3, 4, 7, zero_output=False).double()
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(3, 5, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    direction = tuple(
        torch.randn(p.shape, generator=generator, dtype=p.dtype) * 0.02 for p in model.parameters()
    )
    gradient = torch.autograd.grad(
        loss(model, x.flatten(0, 1), y.flatten()), tuple(model.parameters()), create_graph=True
    )
    linear = sum((g * v).sum() for g, v in zip(gradient, direction))
    hvp = torch.autograd.grad(linear, tuple(model.parameters()))
    quadratic = sum((h * v).sum() for h, v in zip(hvp, direction)) / 2
    before = model.parameter_blocks()
    delta = model.blocks(direction)
    frame = finite_step_scales(
        {"parameters_before": before, "actual_step": delta}, x, y, scales=[1.0]
    )
    assert abs(frame.first_order_mean.iloc[0] - linear.item()) < 1e-12
    assert abs(frame.quadratic_mean.iloc[0] - linear.item() - quadratic.item()) < 1e-12
    with torch.no_grad():
        old = loss(model, x.flatten(0, 1), y.flatten()).item()
        for p, v in zip(model.parameters(), direction):
            p.add_(v)
        exact = loss(model, x.flatten(0, 1), y.flatten()).item() - old
    assert abs(frame.mean_loss_change.iloc[0] - exact) < 1e-12


def test_seed_interval_uses_paired_width_differences():
    import pandas as pd

    from neural_competition.distributions import paired_width_effect

    frame = pd.DataFrame(
        {"seed": [0, 0, 1, 1, 2, 2], "width": [8, 16] * 3, "score": [0, 1, 100, 101, -70, -69]}
    )
    effect = paired_width_effect(frame, "score")
    assert effect["n"] == 3
    assert effect["mean"] == effect["low"] == effect["high"] == 1
