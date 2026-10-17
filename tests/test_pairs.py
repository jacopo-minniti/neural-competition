import torch

from neural_competition.pairs import (
    pair_loss_effects,
    pair_statistics,
    source_neuron_changes,
    summarize_batches,
)


def evaluate(values):
    changes = torch.tensor([[values]], dtype=torch.float64)
    residual = -torch.ones(1, 1, dtype=torch.float64)
    alone, interaction = pair_loss_effects(residual, changes)
    stats, _ = pair_statistics(alone, interaction, torch.tensor([0.5]))
    return summarize_batches(stats)


def test_both_helpful_updates_can_make_each_other_harmful():
    result = evaluate([0.8, 0.8])
    assert result["frequency"] == 1
    assert abs(result["severity"] - 1 / 3) < 1e-12


def test_positive_interaction_does_not_imply_harmful_interference():
    result = evaluate([0.4, 0.4])
    assert result["positive_interaction_fraction"] == 1
    assert result["frequency"] == 0 and result["severity"] == 0


def test_the_two_orders_can_have_different_outcomes():
    result = evaluate([1.5, 0.1])
    assert result["event_count"] == 1
    assert result["frequency"] == 0.5
    assert result["unordered_frequency"] == 1
    assert abs(result["severity"] - 0.055 / 0.47) < 1e-12


def test_no_helpful_pair_is_undefined_rather_than_zero_conflict():
    result = evaluate([0.0, 0.0])
    assert result["eligible_count"] == 0
    assert __import__("math").isnan(result["frequency"])


def test_pair_effects_match_direct_network_losses_with_gate_crossings():
    from neural_competition.model import ShallowReLU, loss

    model = ShallowReLU(2, 3, 11, zero_output=False).double()
    parameters = model.parameter_blocks()
    rng = torch.Generator().manual_seed(12)
    x = torch.randn(2, 12, 2, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 12, generator=rng, dtype=torch.float64)
    delta = torch.randn(2, 3, 4, generator=rng, dtype=torch.float64) * 0.7
    prediction, changes = source_neuron_changes(parameters, delta, x)
    alone, interaction = pair_loss_effects(prediction - y, changes)

    def assign(blocks):
        with torch.no_grad():
            model.weight.copy_(blocks[:, :2])
            model.bias.copy_(blocks[:, 2])
            model.output.copy_(blocks[:, -1])

    for b in range(2):
        assign(parameters)
        before = loss(model, x[b], y[b])
        for r in range(3):
            isolated = parameters.clone()
            isolated[r] += delta[b, r]
            assign(isolated)
            torch.testing.assert_close(loss(model, x[b], y[b]) - before, alone[b, r])
            for s in range(r):
                both = isolated.clone()
                both[s] += delta[b, s]
                assign(both)
                expected = alone[b, r] + alone[b, s] + interaction[b, r, s]
                torch.testing.assert_close(loss(model, x[b], y[b]) - before, expected)
