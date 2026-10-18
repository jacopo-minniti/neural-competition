import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/studies"))
    torch.set_num_threads(1)
    return importlib.import_module("run_temporal_branches")


def test_harm_split_and_exact_objective_budget(study):
    rng = torch.Generator().manual_seed(92)
    before = torch.randn(2, 13, 4, generator=rng, dtype=torch.float64)
    after = before + torch.randn(2, 13, 4, generator=rng, dtype=torch.float64)*0.1
    b0 = before + torch.randn(2, 13, 4, generator=rng, dtype=torch.float64)*0.1
    a0 = b0 + torch.randn(2, 13, 4, generator=rng, dtype=torch.float64)*0.1
    y = torch.randn(2, 13, generator=rng, dtype=torch.float64)
    mask = torch.tensor([[0., 1., 1., 1.], [1., 0., 1., 1.]])
    actual, _ = study.receiver_statistics(before, after, y, b0, a0, mask)
    ordinary, _ = study.receiver_statistics(b0, a0, y, b0, a0, mask)
    torch.testing.assert_close(actual[:, 1]-actual[:, 0],
                               actual[:, 9]-actual[:, 8]+actual[:, 10])
    torch.testing.assert_close(actual[:, 3]-ordinary[:, 3], actual[:, 11]+actual[:, 12])


def test_intervention_preserves_other_neurons_and_matches_prediction_dose(study):
    rng = torch.Generator().manual_seed(1)
    p = torch.randn(2, 4, 5, generator=rng)*0.3
    following = p + torch.randn(2, 4, 5, generator=rng)*0.01
    widths = torch.tensor([[4], [4]])
    x = torch.randn(2, 31, 3, generator=rng, dtype=torch.float64)
    senders = torch.tensor([[1, 2], [3, 0]])
    variants, info = study.make_interventions(p, following, senders, widths, x, 0.2)
    assert all(d["relative_match_error"] < 1e-3 for d in info)
    torch.testing.assert_close(variants[:2], following, rtol=0, atol=0)
    for s in range(2):
        for i in range(2):
            mask = torch.arange(4) != senders[s, i]
            torch.testing.assert_close(variants[(1+2*s)*2+i, mask], following[i, mask], rtol=0, atol=0)


def test_cross_width_dose_matches_removed_function_movement(study):
    rng = torch.Generator().manual_seed(1)
    p = torch.randn(2, 4, 5, generator=rng)*0.3
    following = p + torch.randn(2, 4, 5, generator=rng)*0.01
    p[0, 2:] = 0
    following[0, 2:] = 0
    widths = torch.tensor([[2], [4]])
    x = torch.randn(1, 31, 3, generator=rng, dtype=torch.float64).expand(2, -1, -1)
    senders = torch.tensor([[1, 2], [0, 3]])
    _, info = study.make_interventions(p, following, senders, widths, x, .2, torch.tensor([0, 0]))
    for rank in [0, 1]:
        group = [row for row in info if row["probe_sender"] == rank]
        assert group[0]["removed_prediction_rms"] == pytest.approx(group[1]["removed_prediction_rms"], rel=.001)
        assert max(row["relative_cross_width_match_error"] for row in group) < .001


def test_zero_feasible_common_dose_does_not_select_active_neurons(study):
    rng = torch.Generator().manual_seed(31)
    p = torch.randn(2, 4, 5, generator=rng)*0.3
    following = p + torch.randn(2, 4, 5, generator=rng)*0.01
    following[0, 1] = p[0, 1]
    widths = torch.tensor([[4], [4]])
    x = torch.randn(2, 31, 3, generator=rng, dtype=torch.float64)
    variants, info = study.make_interventions(p, following, torch.tensor([[1, 2]]), widths,
                                             x, .2, torch.tensor([0, 0]))
    for variant in [0, 1, 2]:
        torch.testing.assert_close(variants[variant*2:(variant+1)*2], following, rtol=0, atol=0)
    assert all(row["requested_common_rms"] == 0 for row in info)


@pytest.mark.parametrize("adam", [False, True])
def test_zero_dose_branch_is_exactly_normal_with_optimizer_history(study, adam):
    rng = torch.Generator().manual_seed(23)
    p = torch.randn(2, 4, 5, generator=rng)*0.3
    first = torch.randn(2, 4, 5, generator=rng)*0.001
    second = torch.full_like(p, 0.01)
    x = torch.randn(20, 3, generator=rng)
    y = torch.randn(20, generator=rng)
    indices = torch.randint(20, (5, 2, 6), generator=rng)
    widths, seeds = torch.tensor([[4], [4]]), torch.tensor([0, 1])
    xref = x[:10].double()[None].expand(2, -1, -1)
    yref = y[:10].double()[None].expand(2, -1)
    senders = torch.tensor([[1, 2], [3, 0]])
    variants, _ = study.make_interventions(p, p, senders, widths, xref, 0.)
    fp, ff, fv, records, _ = study.branch_rollout(
        variants, first.repeat(5, 1, 1), second.repeat(5, 1, 1), x, y, indices,
        seeds, widths, xref, yref, senders, 5, .001, .9, adam, 100,
    )
    expected = study.advance(p, first, second, x, y, indices, seeds, widths.double().sqrt().float(),
                             .001, .9, adam, 100)
    for a, b in zip((fp, ff, fv), expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for arm in [1, 2]:
        torch.testing.assert_close(records[:, :, arm], records[:, :, 0], rtol=0, atol=0)


def test_lookahead_preserves_original_minibatch_streams(study):
    seeds = [12026, 12027]
    generators = [torch.Generator().manual_seed(s) for s in seeds]
    first = torch.stack([torch.randint(8192, (1000, 64), generator=g) for g in generators], 1)
    states = [g.get_state().clone() for g in generators]
    future = study.future_indices(first, 990, generators, 129, 8192)
    for g, state in zip(generators, states):
        assert torch.equal(g.get_state(), state)
    expected = torch.stack([
        torch.randint(8192, (2000, 64), generator=torch.Generator().manual_seed(s)) for s in seeds
    ], 1)[990:1119]
    assert torch.equal(future, expected)


def test_probe_schedule_covers_training_without_looking_past_end(study):
    for steps in [100_000, 200_000]:
        times = study.probe_times(steps)
        assert len(times) == 25 and times[0] == 0
        assert np.all(np.diff(times) > 0)
        assert times[-1] + 129 < steps


def test_analysis_signs_and_common_reference_phase(study):
    import pandas as pd

    analysis = importlib.import_module("analyze_temporal_branches")
    rows = []
    for view in ["reference", "heldout"]:
        for arm, h, q, residual, loss in [
            ("ordinary", 2., 0., 0., .04),
            ("sender_damped", 3., .4, .6, .045),
            ("uniform_matched", 4., .7, 1.3, .05),
        ]:
            rows.append(dict(
                condition="toy", width=4, seed=0, probe_id=1, probe_sender=0,
                completed=13, sender=2, view=view, lag=32, arm=arm,
                ordinary_origin_loss=.05 if view == "reference" else .9,
                removed_prediction_rms=.1, relative_match_error=0.,
                cumulative_gain=10., cumulative_harm=h, cumulative_anti_work=h/2,
                origin_loss=.05, loss_after=loss, gain=1., harm=h,
                cumulative_quality_harm=q, cumulative_residual_harm=residual,
            ))
    out = analysis.contrast(pd.DataFrame(rows))
    np.testing.assert_allclose(out.peer_harm_benefit, 1.)
    np.testing.assert_allclose(out.peer_loss_benefit, .005)
    np.testing.assert_allclose(out.selective_harm_saving, 1.)
    np.testing.assert_allclose(out.selective_loss_saving, .005)
    np.testing.assert_allclose(out.reference_mse, .1)


def test_first_shared_scope_keeps_first_sample_per_model_across_views(study):
    import pandas as pd

    analysis = importlib.import_module("analyze_temporal_branches")
    frame = pd.DataFrame([
        dict(condition="toy", width=4, seed=seed, view=view, lag=32, completed=t,
             probe_id=i, reference_mse=mse)
        for seed in [0, 1] for view in ["reference", "heldout"]
        for i, (t, mse) in enumerate([(0, 1.), (5, .15), (9, .1), (20, .01)])
    ])
    first = analysis.scopes(frame)["first_shared_loss"]
    assert len(first) == 4
    assert first.completed.eq(5).all()


def test_calibration_filter_preserves_paired_width_support(study):
    import pandas as pd

    analysis = importlib.import_module("analyze_temporal_branches")
    frame = pd.DataFrame([
        dict(condition="toy", width=width, seed=0, probe_id=probe, probe_sender=0,
             completed=probe, reference_mse=.1, relative_match_error=0.,
             relative_cross_width_match_error=.1 if (width == 8 and probe == 1) else .001)
        for probe in [0, 1] for width in [4, 8]
    ])
    kept = analysis.scopes(frame)["well_matched"]
    assert len(kept) == 2
    assert kept.probe_id.eq(0).all()
    assert set(kept.width) == {4, 8}
