"""Residual work and bounded allocation of proposed neuron directions.

These are local diagnostics, not a decomposition of final optimization error.
All objectives are in half-MSE units. Tensor axes: model, example, neuron.
"""

import torch


def linearized_changes(p, delta, x, norm):
    """Jacobian times a block update, including both layers and hidden bias."""
    dim = x.shape[-1]
    z = x @ p[:, :, :dim].transpose(1, 2) + p[:, None, :, dim]
    dz = x @ delta[:, :, :dim].transpose(1, 2) + delta[:, None, :, dim]
    return (z.relu() * delta[:, None, :, -1]
            + (z > 0) * dz * p[:, None, :, -1]) / norm[:, :, None]


def quadratic_terms(residual, changes):
    return ((residual[:, :, None] * changes).mean(1),
            changes.transpose(1, 2) @ changes / changes.shape[1])


def objective(b, gram, alpha):
    return (b * alpha).sum(-1) + 0.5 * (
        alpha * (gram @ alpha[:, :, None]).squeeze(-1)).sum(-1)


def uniform_minimum(b, gram):
    linear = b.sum(-1)
    energy = gram.sum((-1, -2)).clamp_min(0)
    gamma = torch.where(energy > 0, -linear / energy.clamp_min(1e-300),
                        (linear < 0).to(b.dtype)).clamp(0, 1)
    return gamma[:, None].expand_as(b).clone()


@torch.jit.script
def box_minimum(b, gram, initial, tolerance: float = 1e-10, max_sweeps: int = 1000):
    """Cyclic exact coordinate descent with a convex duality-gap certificate.

    The returned gap bounds Q(alpha)-min Q over the unit box, even for singular
    Gram matrices. Scale-relative stopping avoids declaring tiny steps solved
    merely because their absolute objective is small.
    """
    alpha = initial.clone()
    gradient = b + torch.bmm(gram, alpha.unsqueeze(-1)).squeeze(-1)
    scale = b.abs().sum(-1) + gram.abs().sum((-1, -2))
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    gap = torch.zeros_like(scale)
    sweep = 0
    for sweep in range(max_sweeps):
        for j in range(b.shape[1]):
            old = alpha[:, j].clone()
            value = (old - gradient[:, j] / diagonal[:, j].clamp_min(1e-300)).clamp(0, 1)
            value = torch.where(diagonal[:, j] > 0, value, (b[:, j] < 0).to(b.dtype))
            change = value - old
            alpha[:, j] = value
            gradient = gradient + gram[:, :, j] * change[:, None]
        gradient = b + torch.bmm(gram, alpha.unsqueeze(-1)).squeeze(-1)
        gap = (gradient.clamp_min(0) * alpha
               + (-gradient).clamp_min(0) * (1 - alpha)).sum(-1)
        if bool(torch.all(gap <= tolerance * scale.clamp_min(1e-300))):
            break
    return alpha, gap, sweep + 1


def allocation_rules(b, gram, widths, seed=0, groups=4):
    """Uniform, sign-only, per-neuron, fixed-group, and shuffled controls."""
    uniform = uniform_minimum(b, gram)
    selected, gap, sweeps = box_minimum(b, gram, uniform)
    # Sign-only filtering uses no off-diagonal interaction information.
    sign = (b < 0).to(b.dtype)
    # Optimize one remaining scalar on that direction, in the same unit box.
    sb = (b * sign).sum(-1)
    sg = (gram * sign[:, :, None] * sign[:, None, :]).sum((-1, -2))
    gamma = torch.where(sg > 0, -sb / sg.clamp_min(1e-300), (sb < 0).to(b.dtype))
    sign = sign * gamma.clamp(0, 1)[:, None]
    mapping = torch.zeros(b.shape[0], b.shape[1], groups, dtype=b.dtype)
    shuffled = selected.clone()
    rng = torch.Generator().manual_seed(seed)
    for i, width in enumerate(widths.flatten().tolist()):
        order = torch.randperm(width, generator=rng)
        mapping[i, order, torch.arange(width) % groups] = 1
        shuffled[i, :width] = selected[i, torch.randperm(width, generator=rng)]
    grouped_b = (mapping.transpose(1, 2) @ b[:, :, None]).squeeze(-1)
    grouped_g = mapping.transpose(1, 2) @ gram @ mapping
    grouped, group_gap, _ = box_minimum(
        grouped_b, grouped_g, uniform_minimum(grouped_b, grouped_g))
    grouped = (mapping @ grouped[:, :, None]).squeeze(-1)
    return dict(uniform=uniform, sign=sign, neuron=selected, group4=grouped,
                shuffled=shuffled), dict(gap=gap, group_gap=group_gap, sweeps=sweeps)


def work_statistics(residual, changes):
    """Duplication-invariant P/N, corrective cosine, and exact work budget."""
    work = -(residual[:, :, None] * changes).mean(1)
    total = changes.sum(-1)
    residual_energy = residual.square().mean(1)
    movement_energy = total.square().mean(1)
    s = work.sum(-1)
    parallel = s.square() / residual_energy.clamp_min(1e-300)
    cosine = s.clamp_min(0).square() / (
        residual_energy * movement_energy).clamp_min(1e-300)
    cosine = torch.where((residual_energy > 0) & (movement_energy > 0),
                         cosine, torch.full_like(cosine, float('nan')))
    return dict(loss=0.5 * residual_energy, positive_work=work.clamp_min(0).sum(-1),
                negative_work=(-work).clamp_min(0).sum(-1), work=s,
                curvature=0.5 * movement_energy, parallel_cost=0.5 * parallel,
                perpendicular_cost=0.5 * (movement_energy - parallel),
                corrective_cosine=cosine, progress=s - 0.5 * movement_energy)
