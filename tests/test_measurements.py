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
