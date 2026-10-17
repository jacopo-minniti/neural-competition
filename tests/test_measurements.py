"""Numerical checks of the scientific quantities, rather than training accuracy."""

import math

import torch
from torch.nn import functional as F

from neural_competition.config import ExperimentConfig, OptimizerConfig
from neural_competition.diagnostics import batch_sharpness, gradient_blocks
from neural_competition.experiment import run_experiments
from neural_competition.model import ShallowReLU, loss


def test_neuron_attribution_and_first_order_cross_loss():
    model = ShallowReLU(3, 5, 9, zero_output=False).double()
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(12, generator=generator, dtype=torch.float64)
    value_b = loss(model, x[:6], y[:6])
    value_c = loss(model, x[6:], y[6:])
    parameters = tuple(model.parameters())
    gradients_b = torch.autograd.grad(value_b, parameters)
    gradients_c = torch.autograd.grad(value_c, parameters)
    direct_dot = sum((b * c).sum() for b, c in zip(gradients_b, gradients_c))
    neuron_dot = (model.blocks(gradients_b) * model.blocks(gradients_c)).sum(dim=1)
    torch.testing.assert_close(neuron_dot.sum(), direct_dot)
    eta = 1e-6
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients_b):
            parameter.sub_(eta * gradient)
    observed_derivative = (loss(model, x[6:], y[6:]) - value_c.detach()) / eta
    torch.testing.assert_close(observed_derivative, -direct_dot, atol=2e-6, rtol=2e-5)


def test_hvp_sharpness_matches_explicit_tiny_hessian():
    model = ShallowReLU(2, 3, 4, zero_output=False).double()
    with torch.no_grad():
        model.bias.fill_(2.0)  # Keep this finite-difference-free check away from ReLU boundaries.
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(7, 2, generator=generator, dtype=torch.float64) * 0.1
    y = torch.randn(7, generator=generator, dtype=torch.float64)
    vector = torch.cat([p.detach().flatten() for p in model.parameters()]).requires_grad_()

    def objective(v):
        prediction = F.relu(F.linear(x, v[:6].reshape(3, 2), v[6:9])) @ v[9:] / math.sqrt(3)
        return 0.5 * (prediction - y).square().mean()

    gradient = torch.autograd.grad(objective(vector), vector)[0]
    hessian = torch.autograd.functional.hessian(objective, vector)
    expected = gradient @ hessian @ gradient / gradient.square().sum()
    observed = batch_sharpness(model, x[None], y[None])[0]
    torch.testing.assert_close(observed, expected)


def test_initial_error_is_identical_and_larger_student_can_represent_teacher():
    x = torch.randn(12, 3, generator=torch.Generator().manual_seed(2))
    for width in [4, 16, 64]:
        model = ShallowReLU(3, width, 1)
        torch.testing.assert_close(model(x), torch.zeros(12))
    teacher = ShallowReLU(3, 4, 1, zero_output=False)
    student = ShallowReLU(3, 16, 2)
    with torch.no_grad():
        student.weight[:4].copy_(teacher.weight)
        student.bias[:4].copy_(teacher.bias)
        student.output[:4].copy_(teacher.output * math.sqrt(16 / 4))
    torch.testing.assert_close(student(x), teacher(x))


def test_saved_steps_and_gradients_are_consistent(tmp_path):
    config = ExperimentConfig(
        input_dim=3,
        teacher_width=2,
        n_train=48,
        n_test=24,
        widths=[4],
        seeds=[0],
        steps=3,
        batch_size=8,
        probe_batches=4,
        checkpoint_steps=[0, 1, 2],
        sharpness_batches=1,
        cpu_threads=1,
        optimizers=[
            OptimizerConfig("sgd", 0.03, "sgd"),
            OptimizerConfig("momentum", 0.003, "momentum"),
            OptimizerConfig("adam", 0.001, "adam"),
        ],
    )
    output = run_experiments(config, tmp_path / "experiment", "cpu")
    for spec in config.optimizers:
        path = output / f"w0004_seed0_{spec.label}"
        snapshot = torch.load(path / "checkpoints/step_000001.pt", weights_only=True)
        torch.testing.assert_close(
            snapshot["parameters_after"] - snapshot["parameters_before"], snapshot["actual_step"]
        )
        if spec.name == "sgd":
            torch.testing.assert_close(
                snapshot["actual_step"],
                -spec.lr * snapshot["sampled_gradient"],
                atol=6e-8,
                rtol=5e-4,
            )
        model = ShallowReLU(3, 4)
        blocks = snapshot["parameters_before"]
        with torch.no_grad():
            model.weight.copy_(blocks[:, :3])
            model.bias.copy_(blocks[:, 3])
            model.output.copy_(blocks[:, 4])
        data = torch.load(output / "data.pt", weights_only=True)
        indices = data["probe_indices"][0]
        expected_loss, expected_gradient = gradient_blocks(
            model, data["x_train"][indices], data["y_train"][indices]
        )
        torch.testing.assert_close(expected_gradient, snapshot["probe_gradients"][0])
        torch.testing.assert_close(expected_loss, snapshot["probe_losses_before"][0])
        assert torch.isfinite(snapshot["batch_sharpness_samples"]).all()
    # A completed sweep can be reopened without overwriting its checkpoint files.
    checkpoint = output / "w0004_seed0_sgd/checkpoints/step_000001.pt"
    modified = checkpoint.stat().st_mtime_ns
    run_experiments(config, output, "cpu", resume=True)
    assert checkpoint.stat().st_mtime_ns == modified


def test_batch_sharpness_is_ratio_of_expectations_not_mean_of_ratios():
    from neural_competition.diagnostics import batch_curvature_samples

    model = ShallowReLU(2, 3, 4, zero_output=False).double()
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(2, 8, 2, generator=generator, dtype=torch.float64)
    y = torch.randn(2, 8, generator=generator, dtype=torch.float64)
    y[1].mul_(10)
    numerator, denominator = batch_curvature_samples(model, x, y)
    individual = batch_sharpness(model, x, y)
    torch.testing.assert_close(numerator / denominator, individual)
    weighted = (individual * denominator).sum() / denominator.sum()
    torch.testing.assert_close(numerator.sum() / denominator.sum(), weighted)
    assert abs(weighted - individual.mean()) > 1e-3


def test_loss_crossings_and_muon_hybrid_are_saved(tmp_path):
    config = ExperimentConfig(
        input_dim=3,
        teacher_width=2,
        n_train=48,
        n_test=24,
        widths=[4],
        seeds=[0],
        steps=3,
        batch_size=8,
        probe_batches=4,
        checkpoint_steps=[0, 2],
        sharpness_batches=1,
        cpu_threads=1,
        loss_checkpoints=[100.0],
        loss_check_every=1,
        optimizers=[OptimizerConfig("muon", 0.02, "muon", 0.95, 0.001)],
    )
    output = run_experiments(config, tmp_path / "muon", "cpu")
    folder = output / "w0004_seed0_muon"
    first = torch.load(folder / "checkpoints/step_000000.pt", weights_only=True)
    last = torch.load(folder / "checkpoints/step_000002.pt", weights_only=True)
    assert first["loss_targets_crossed"] == [100.0]
    assert last["loss_targets_crossed"] == []
    assert last["optimizer_after"]["momentum_buffer"][:, :3].norm() > 0
    assert last["optimizer_after"]["exp_avg"][:, -2:].norm() > 0
    assert last["actual_step"][:, :3].norm() > 0
    assert last["actual_step"][:, -2:].norm() > 0
    numerator = last["batch_curvature_numerators"]
    denominator = last["batch_curvature_denominators"]
    from neural_competition.diagnostics import summarize

    assert math.isclose(
        summarize(last)["batch_sharpness_ratio"], (numerator.sum() / denominator.sum()).item()
    )


def test_saved_state_sgd_continuation_preserves_parameters_and_batch_order(tmp_path):
    from neural_competition.experiment import run_one
    from neural_competition.model import make_data

    config = ExperimentConfig(
        input_dim=3,
        teacher_width=2,
        n_train=48,
        n_test=24,
        widths=[4],
        seeds=[0],
        steps=2,
        batch_size=8,
        probe_batches=4,
        checkpoint_steps=[0, 1],
        sharpness_batches=1,
        cpu_threads=1,
    )
    data = make_data(config)
    initial_model = ShallowReLU(3, 4, 7, zero_output=False)
    initial = initial_model.parameter_blocks()
    spec = OptimizerConfig("sgd", 0.03, "sgd")
    run_one(
        config,
        spec,
        4,
        0,
        data,
        tmp_path / "continued",
        torch.device("cpu"),
        initial_parameters=initial,
        start_update=5,
    )
    first = torch.load(tmp_path / "continued/checkpoints/step_000000.pt", weights_only=True)
    torch.testing.assert_close(first["parameters_before"], initial)
    generator = torch.Generator().manual_seed(config.data_seed + 10000)
    indices = torch.randint(config.n_train, (7, config.batch_size), generator=generator)
    _, gradient = gradient_blocks(
        initial_model, data["x_train"][indices[5]], data["y_train"][indices[5]]
    )
    torch.testing.assert_close(first["sampled_gradient"], gradient)
