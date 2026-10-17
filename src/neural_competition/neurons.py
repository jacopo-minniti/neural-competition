"""Separate neuron gradient conflict from cancellation of prediction changes."""

import math

import torch


def candidate_descent(snapshot, gradients=None, *, include_history=True):
    """SGD descent direction before multiplying by lr; freeze pre-step history.

    PyTorch's momentum here has zero dampening, no Nesterov, no decay.
    A candidate minibatch changes g_B but does not replay the past trajectory.
    """
    gradients = snapshot["probe_gradients"] if gradients is None else gradients
    gradients = gradients.double()
    spec = snapshot["optimizer"]
    if spec["name"] == "sgd" or not include_history:
        return gradients
    if spec["name"] != "momentum":
        raise ValueError("Only plain SGD and classical momentum are supported")
    previous = snapshot["optimizer_before"]["momentum_buffer"].double()
    return gradients + spec["momentum"] * previous


def step_transfer_masses(gradients, descent):
    """Neuron-level g_C dot descent_B, with ordered off-diagonal batch pairs.

    Normalize by sqrt(E||g_C||² E||descent_B||²). Unlike raw q, momentum
    transfer is not symmetric in B,C; using just the upper triangle is biased.
    """
    gradients, descent = gradients.double(), descent.double()
    count = len(gradients)
    if gradients.shape != descent.shape:
        raise ValueError("Candidate descents and probe gradients must have identical shapes")
    q = torch.bmm(descent.transpose(0, 1), gradients.transpose(0, 1).transpose(1, 2))
    keep = ~torch.eye(count, dtype=torch.bool)
    q = q[:, keep]
    g_energy = gradients.square().sum((1, 2)).mean()
    d_energy = descent.square().sum((1, 2)).mean()
    scale = (g_energy * d_energy).sqrt()
    if scale <= 0:
        return {k: math.nan for k in ["step_P", "step_N", "step_global_N", "step_p_negative"]}
    q /= scale
    total = q.sum(0)
    return {
        "step_P": q.clamp_min(0).sum(0).mean().item(),
        "step_N": (-q).clamp_min(0).sum(0).mean().item(),
        "step_global_N": (-total).clamp_min(0).mean().item(),
        "step_p_negative": (total < 0).double().mean().item(),
    }


def gradient_masses(gradients):
    """Off-diagonal batch pairs, normalized by whole-network gradient energy.

    Signs are taken before summing neuron blocks. This is attribution of batch
    conflict to neurons, not an inner product between different neurons.
    """
    gradients = gradients.double()
    count, width, _ = gradients.shape
    pairs = torch.triu_indices(count, count, offset=1)
    blocks = gradients.transpose(0, 1)
    q = torch.bmm(blocks, blocks.transpose(1, 2))[:, pairs[0], pairs[1]]
    energy = gradients.square().sum(-1).mean(0)
    scale = energy.sum()
    if scale <= 0:
        raise ValueError("Gradient energy must be positive")
    positive = q.clamp_min(0).mean(1) / scale
    negative = (-q).clamp_min(0).mean(1) / scale
    total = q.sum(0) / scale
    summary = {
        "P": positive.sum().item(),
        "N": negative.sum().item(),
        "global_P": total.clamp_min(0).mean().item(),
        "global_N": (-total).clamp_min(0).mean().item(),
        "pair_mean": total.mean().item(),
        "p_negative": (total < 0).double().mean().item(),
        "gradient_energy": scale.item(),
        "probe_batches": count,
    }
    summary["offset_negative"] = summary["N"] - summary["global_N"]
    # An independent identity catches normalization, pairing and axis errors.
    torch.testing.assert_close(
        positive.sum() - negative.sum(), total.mean(), atol=1e-12, rtol=1e-10
    )
    neurons = [
        {
            "neuron": r,
            "P": positive[r].item(),
            "N": negative[r].item(),
            "energy_share": (energy[r] / scale).item(),
            "p_negative": (q[r] < 0).double().mean().item(),
        }
        for r in range(width)
    ]
    return summary, neurons


def prediction_moves(parameters, updates, x):
    """Exact finite changes from independent candidate SGD steps in float64.

    Shapes: parameters [m,d+2], updates [b,m,d+2], x [n,d]. Each d_r is
    also exactly the change from applying only neuron r's block update.
    """
    parameters, updates, x = parameters.double(), updates.double(), x.double()
    dim = x.shape[-1]
    z = x @ parameters[:, :dim].T + parameters[:, dim]
    dz = torch.einsum("nd,bmd->bnm", x, updates[:, :, :dim])
    dz = dz + updates[:, None, :, dim]
    after_z = z[None] + dz
    scale = math.sqrt(len(parameters))
    before = z.relu() * parameters[:, -1] / scale
    after = after_z.relu() * (parameters[None, :, -1] + updates[:, :, -1])[:, None] / scale
    return before.sum(-1), after - before[None], (z[None] > 0) != (after_z > 0)


def source_improvement(parameters, updates, x, y):
    """Each candidate update evaluated on its own source minibatch."""
    parameters, updates, x, y = (parameters.double(), updates.double(), x.double(), y.double())
    dim = x.shape[-1]
    z = torch.einsum("bnd,md->bnm", x, parameters[:, :dim]) + parameters[:, dim]
    dz = torch.einsum("bnd,bmd->bnm", x, updates[:, :, :dim])
    dz = dz + updates[:, None, :, dim]
    scale = math.sqrt(len(parameters))
    before = (z.relu() * parameters[:, -1]).sum(-1) / scale
    after = ((z + dz).relu() * (parameters[None, :, -1] + updates[:, :, -1])[:, None]).sum(
        -1
    ) / scale
    change = after - before
    return -((before - y) * change + 0.5 * change.square()).mean(1)


def _ratio(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator, float("nan"))


def movement_statistics(residual, moves, mask=None):
    """Per-candidate-step diagnostics; loss is half-MSE.

    solo_harm sums hypothetical losses with one block updated at a time. It
    is NOT an additive loss attribution. The joint loss includes cross terms.
    R < 1 means net cancellation in squared prediction movement; R > 1 means
    reinforcement. Neither sign alone establishes useful learning.
    """
    residual, moves = residual.double(), moves.double()
    if mask is not None:
        residual, moves = residual[mask], moves[:, mask]
    if not len(residual):
        raise ValueError("Evaluation mask is empty")
    total = moves.sum(-1)
    linear_neuron = residual[None, :, None] * moves
    solo_change = linear_neuron + 0.5 * moves.square()
    joint_change = residual[None] * total + 0.5 * total.square()
    diagonal = moves.square().sum(-1).mean(1)
    joint_energy = total.square().mean(1)
    gram = torch.einsum("bnr,bns->brs", moves, moves) / len(residual)
    linear = (residual[None] * total).mean(1)
    error_energy = residual.square().mean()
    residual_dot = linear_neuron.mean(1)
    parallel = _ratio(linear.square(), error_energy)
    perpendicular = (joint_energy - parallel).clamp_min(0)
    perpendicular_diagonal = (
        diagonal - _ratio(residual_dot.square().sum(-1), error_energy)
    ).clamp_min(0)
    solo_harm_pointwise = solo_change.clamp_min(0).sum(-1)
    solo_harm = solo_harm_pointwise.mean(1)
    records = {
        "R": _ratio(joint_energy, diagonal),
        "R_perpendicular": _ratio(perpendicular, perpendicular_diagonal),
        "cancellation_fraction": 1 - _ratio(total.abs().mean(1), moves.abs().sum(-1).mean(1)),
        "absolute_joint_movement": total.abs().mean(1),
        "absolute_individual_movement": moves.abs().sum(-1).mean(1),
        "solo_harm": solo_harm,
        "joint_harm": joint_change.clamp_min(0).mean(1),
        "joint_benefit": (-joint_change).clamp_min(0).mean(1),
        "loss_change": joint_change.mean(1),
        "linear_progress": -linear,
        "joint_energy": joint_energy,
        "individual_energy": diagonal,
        "perpendicular_energy": perpendicular,
        "perpendicular_individual_energy": perpendicular_diagonal,
        "solo_harm_offset_fraction": _ratio(
            (solo_harm_pointwise * (joint_change <= 0)).mean(1), solo_harm
        ),
        "offset_solo_harm": (solo_harm_pointwise * (joint_change <= 0)).mean(1),
        "joint_harm_fraction": (joint_change > 0).double().mean(1),
        "neuron_harm_fraction": (solo_change > 0).double().mean((1, 2)),
        "linear_neuron_harm": linear_neuron.clamp_min(0).sum(-1).mean(1),
        "linear_neuron_benefit": (-linear_neuron).clamp_min(0).sum(-1).mean(1),
    }
    # Exact additive-network identity, including all ReLU gate crossings.
    expanded = solo_change.sum(-1) + 0.5 * (total.square() - moves.square().sum(-1))
    torch.testing.assert_close(expanded, joint_change, atol=1e-12, rtol=1e-9)
    return records, gram
