"""Diagnostics at one frozen parameter state, followed by one real update."""

import copy
import math

import torch

from .model import loss


def gradient_blocks(model, x, y):
    value = loss(model, x, y)
    gradients = torch.autograd.grad(value, tuple(model.parameters()))
    return value.detach(), model.blocks(gradients).detach()


@torch.no_grad()
def probe_losses(model, x_probe, y_probe):
    shape = y_probe.shape
    predictions = model(x_probe.reshape(-1, x_probe.shape[-1])).reshape(shape)
    return 0.5 * (predictions - y_probe).square().mean(dim=1)


def optimizer_buffers(model, optimizer):
    result = {}
    for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
        values = [
            optimizer.state.get(p, {}).get(key, torch.zeros_like(p)).detach()
            for p in model.parameters()
        ]
        result[key] = model.blocks(values).cpu()
    return result


def batch_curvature_samples(model, x_probe, y_probe):
    """Return g_B^T H_B g_B and ||g_B||^2 separately in CPU float64.

    Their ratio of sums estimates Definition 1 of Andreyev & Beneventano
    (arXiv:2412.20553v3). A mean of individual ratios is a different statistic.
    ReLU derivatives describe the local activation region, not boundary crossings.
    """
    if len(x_probe) == 0:
        empty = torch.empty(0, dtype=torch.float64)
        return empty, empty
    reference = copy.deepcopy(model).cpu().double()
    numerators, denominators = [], []
    parameters = tuple(reference.parameters())
    for x, y in zip(x_probe.cpu().double(), y_probe.cpu().double()):
        gradients = torch.autograd.grad(loss(reference, x, y), parameters, create_graph=True)
        direction = tuple(g.detach() for g in gradients)
        directional_derivative = sum((g * v).sum() for g, v in zip(gradients, direction))
        hvp = torch.autograd.grad(directional_derivative, parameters)
        numerator = sum((v * h).sum() for v, h in zip(direction, hvp))
        denominator = sum(v.square().sum() for v in direction)
        numerators.append(numerator.detach())
        denominators.append(denominator.detach())
    return torch.stack(numerators), torch.stack(denominators)


def batch_sharpness(model, x_probe, y_probe):
    """Individual Rayleigh quotients (legacy statistic; not the paper's ratio)."""
    numerator, denominator = batch_curvature_samples(model, x_probe, y_probe)
    return torch.where(denominator > 0, numerator / denominator, float("nan"))


def collect_before(model, optimizer, data, x_probe, y_probe, sharpness_batches):
    train_loss, mean_gradient = gradient_blocks(model, data["x_train"], data["y_train"])
    gradients, values = [], []
    for x, y in zip(x_probe, y_probe):
        value, gradient = gradient_blocks(model, x, y)
        gradients.append(gradient.cpu())
        values.append(value.cpu())
    with torch.no_grad():
        test_predictions = model(data["x_test"])
        test_loss = 0.5 * (test_predictions - data["y_test"]).square().mean()
    curvature_numerator, curvature_denominator = batch_curvature_samples(
        model, x_probe[:sharpness_batches], y_probe[:sharpness_batches]
    )
    return {
        "parameters_before": model.parameter_blocks().cpu(),
        "optimizer_before": optimizer_buffers(model, optimizer),
        "probe_gradients": torch.stack(gradients),
        "mean_gradient": mean_gradient.cpu(),
        "probe_losses_before": torch.stack(values),
        "test_predictions_before": test_predictions.detach().cpu(),
        "train_loss_before": float(train_loss.cpu()),
        "test_loss_before": float(test_loss.cpu()),
        "batch_sharpness_samples": curvature_numerator / curvature_denominator,
        "batch_curvature_numerators": curvature_numerator,
        "batch_curvature_denominators": curvature_denominator,
    }


def alignment_arrays(gradients):
    flat = gradients.flatten(1)
    gram = flat @ flat.T
    norms = flat.norm(dim=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = torch.where(denominator > 0, gram / denominator.clamp_min(1e-30), float("nan"))
    pair_indices = torch.triu_indices(len(flat), len(flat), offset=1)
    return gram[pair_indices[0], pair_indices[1]], cosine[pair_indices[0], pair_indices[1]]


def distribution_summary(prefix, values):
    values = values.flatten().double()
    values = values[torch.isfinite(values)]
    if not len(values):
        return {
            f"{prefix}_{key}": math.nan
            for key in (
                "mean",
                "std",
                "skewness",
                "excess_kurtosis",
                "negative_fraction",
                "q01",
                "q05",
                "q50",
                "q95",
                "q99",
            )
        }
    centered = values - values.mean()
    variance = centered.square().mean()
    skewness = (centered.pow(3).mean() / variance.pow(1.5)).item() if variance > 0 else math.nan
    kurtosis = (centered.pow(4).mean() / variance.square() - 3).item() if variance > 0 else math.nan
    quantiles = torch.quantile(
        values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64)
    )
    result = {
        f"{prefix}_skewness": skewness,
        f"{prefix}_excess_kurtosis": kurtosis,
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_std": values.std(unbiased=False).item(),
        f"{prefix}_negative_fraction": (values < 0).double().mean().item(),
    }
    result.update(
        {
            f"{prefix}_{key}": q.item()
            for key, q in zip(("q01", "q05", "q50", "q95", "q99"), quantiles)
        }
    )
    return result


def summarize(snapshot):
    gradients = snapshot["probe_gradients"]
    flat = gradients.flatten(1)
    pair_dot, pair_cosine = alignment_arrays(gradients)
    mean_gradient = snapshot["mean_gradient"].flatten()
    delta = snapshot["actual_step"].flatten()
    linear = flat @ delta
    changes = snapshot["probe_losses_after"] - snapshot["probe_losses_before"]
    neuron_scores = torch.einsum("ird,jrd->ijr", gradients, gradients)
    indices = torch.triu_indices(len(flat), len(flat), offset=1)
    negative_mass = (-neuron_scores[indices[0], indices[1]]).clamp_min(0).sum(dim=-1)
    previous = snapshot["previous_step"].flatten()
    denominator = delta.norm() * previous.norm()
    result = {
        "train_loss": snapshot["train_loss_before"],
        "test_loss": snapshot["test_loss_before"],
        "train_loss_after": snapshot["train_loss_after"],
        "test_loss_after": snapshot["test_loss_after"],
        "gradient_norm_mean": flat.norm(dim=1).mean().item(),
        "mean_gradient_norm": mean_gradient.norm().item(),
        "step_norm": delta.norm().item(),
        "step_previous_cosine": (torch.dot(delta, previous) / denominator).item()
        if denominator > 0
        else math.nan,
        "neuron_negative_mass_mean": negative_mass.mean().item(),
        "probe_loss_increase_fraction": (changes > 0).float().mean().item(),
        "probe_linear_harm_fraction": (linear > 0).float().mean().item(),
        "probe_loss_change_mean": changes.mean().item(),
        "finite_step_remainder_mean": (changes - linear).mean().item(),
        "batch_sharpness": snapshot["batch_sharpness_samples"].mean().item()
        if len(snapshot["batch_sharpness_samples"])
        else math.nan,
    }
    if "batch_curvature_numerators" in snapshot:
        numerator = snapshot["batch_curvature_numerators"].sum()
        denominator = snapshot["batch_curvature_denominators"].sum()
        result["batch_sharpness_ratio"] = (
            (numerator / denominator).item() if denominator > 0 else math.nan
        )
    for prefix, values in (
        ("pair_dot", pair_dot),
        ("pair_cosine", pair_cosine),
        ("gradient_mean_dot", flat @ mean_gradient),
        ("gradient_step_dot", linear),
    ):
        result.update(distribution_summary(prefix, values))
    return result
