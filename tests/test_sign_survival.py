import torch

from neural_competition.sign_survival import advance_protected
from neural_competition.training import advance


def test_inactive_projection_exactly_matches_existing_optimizers():
    rng = torch.Generator().manual_seed(123)
    p = torch.randn(2, 4, 5, generator=rng, dtype=torch.float64)*.1
    p[:, :, -2] = 0
    x = torch.randn(30, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(30, generator=rng, dtype=torch.float64)
    ids = torch.randint(30, (7, 2, 8), generator=rng)
    seeds = torch.arange(2)
    norm = torch.full((2, 1), 2., dtype=torch.float64)
    for kind in range(3):
        args = (p.clone(), torch.zeros_like(p), torch.zeros_like(p), x, y, ids,
                seeds, norm, torch.ones_like(p), .03, kind)
        expected = advance(*args, False, 0)
        actual = advance_protected(*args, 0, p[:, :, -1].sign(), torch.zeros(2, dtype=torch.bool),
                                   torch.full((2,), -1))
        for left, right in zip(expected, actual[:3]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert actual[3].sum() == 0


def test_projection_only_changes_crossing_readout_and_keeps_optimizer_state():
    p = torch.tensor([[[1., 0., .01]]], dtype=torch.float64)
    x = torch.ones(1, 1, dtype=torch.float64)
    y = -torch.ones(1, dtype=torch.float64)
    args = (p, torch.zeros_like(p), torch.zeros_like(p), x, y,
            torch.zeros(1, 1, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.float64), torch.ones_like(p), .1, 0)
    ordinary = advance(*args, False, 0)
    actual = advance_protected(*args, 0, torch.ones(1, 1), torch.ones(1, dtype=torch.bool),
                               torch.full((1,), -1))
    assert ordinary[0][0, 0, -1] < 0
    assert actual[0][0, 0, -1] == 0
    torch.testing.assert_close(actual[0][:, :, :-1], ordinary[0][:, :, :-1])
    torch.testing.assert_close(actual[1], ordinary[1])
    assert actual[3].item() == 1
    torch.testing.assert_close(actual[4][0], -ordinary[0][0, 0, -1])


def test_damping_matches_reference_correction_without_projecting_its_sign():
    p = torch.tensor([[[1., 0., .01]], [[1., 0., -.1]]], dtype=torch.float64)
    x = torch.ones(1, 1, dtype=torch.float64)
    y = -torch.ones(1, dtype=torch.float64)
    args = (p, torch.zeros_like(p), torch.zeros_like(p), x, y,
            torch.zeros(1, 2, 1, dtype=torch.long), torch.arange(2),
            torch.ones(2, 1, dtype=torch.float64), torch.ones_like(p), .1, 0)
    ordinary = advance(*args, False, 0)
    actual = advance_protected(*args, 0, torch.ones(2, 1),
                               torch.tensor([True, False]), torch.tensor([-1, 0]))
    budget = -ordinary[0][0, 0, -1]
    length = (ordinary[0][1, 0, -1]-p[1, 0, -1]).abs()
    expected = torch.minimum(budget, length)
    torch.testing.assert_close((actual[0][1, 0, -1]-ordinary[0][1, 0, -1]).abs(), expected)
    assert actual[0][1, 0, -1] < 0  # No half-line projection in the damping arm.
    torch.testing.assert_close(actual[1], ordinary[1])
    torch.testing.assert_close(actual[5][1], expected)
