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
