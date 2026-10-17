import torch

from neural_competition.neurons import gradient_masses, movement_statistics, prediction_moves


def test_neuron_negative_mass_is_hidden_by_other_neurons():
    # Two batch demands: neuron 0 conflicts, neuron 1 more than offsets it.
    gradients = torch.tensor([[[1.0], [1.0]], [[-2.0], [3.0]]])
    result, neurons = gradient_masses(gradients)
    assert result["N"] > 0 and result["global_N"] == 0
    assert neurons[0]["N"] > 0 and neurons[1]["P"] > 0
    assert abs(result["P"] - result["N"] - result["pair_mean"]) < 1e-12


def test_cancellation_can_protect_a_correct_prediction():
    # Both neurons alone harm a correct prediction, together they leave it intact.
    result, gram = movement_statistics(torch.zeros(2), torch.tensor([[[1.0, -1.0], [2.0, -2.0]]]))
    assert result["solo_harm"].item() > 0
    assert result["joint_harm"].item() == 0
    assert result["R"].item() == 0
    assert result["solo_harm_offset_fraction"].item() == 1
    assert gram[0, 0, 1] < 0


def test_reinforcement_can_help_or_harm_depending_on_residual():
    moves = torch.tensor([[[-0.2, -0.3]]])
    helpful, _ = movement_statistics(torch.ones(1), moves)
    harmful, _ = movement_statistics(torch.zeros(1), moves)
    assert helpful["R"].item() > 1
    assert helpful["loss_change"].item() < 0
    assert harmful["loss_change"].item() > 0


def test_finite_neuron_changes_equal_isolated_block_changes():
    from neural_competition.model import ShallowReLU

    model = ShallowReLU(2, 3, 17, zero_output=False).double()
    before = model.parameter_blocks()
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(20, 2, generator=generator, dtype=torch.float64)
    delta = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64) * 0.2
    prediction, changes, _ = prediction_moves(before, delta, x)
    torch.testing.assert_close(prediction, model(x))
    for b in range(2):
        for r in range(3):
            isolated = before.clone()
            isolated[r] += delta[b, r]
            with torch.no_grad():
                model.weight.copy_(isolated[:, :2])
                model.bias.copy_(isolated[:, 2])
                model.output.copy_(isolated[:, -1])
            torch.testing.assert_close(model(x) - prediction, changes[b, :, r])
