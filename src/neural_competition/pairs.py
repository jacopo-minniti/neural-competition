"""Exact interference between two fixed neuron updates on their source batch."""

import math

import torch


def source_neuron_changes(parameters, updates, x):
    """Each candidate update is evaluated on its OWN minibatch.

    Parameters [m,d+2], updates [b,m,d+2], x [b,n,d]. No linearization;
    simultaneous layer changes and ReLU gate crossings are included.
    """
    parameters, updates, x = parameters.double(), updates.double(), x.double()
    dim = x.shape[-1]
    z = torch.einsum("bnd,md->bnm", x, parameters[:, :dim]) + parameters[:, dim]
    dz = torch.einsum("bnd,bmd->bnm", x, updates[:, :, :dim]) + updates[:, None, :, dim]
    scale = math.sqrt(len(parameters))
    before = z.relu() * parameters[:, -1] / scale
    after = (z + dz).relu() * (parameters[None, :, -1] + updates[:, :, -1])[:, None] / scale
    return before.sum(-1), after - before


def pair_loss_effects(residual, changes):
    """Return standalone loss changes A_r and interactions I_rs for half-MSE.

    A_r=E(e*d_r + d_r²/2), I_rs=E(d_r*d_s). Pair update loss is
    A_r+A_s+I_rs and the marginal loss change of s after r is A_s+I_rs.
    All expectations are averages within each source minibatch.
    """
    residual, changes = residual.double(), changes.double()
    standalone = (residual[..., None] * changes + 0.5 * changes.square()).mean(1)
    interaction = torch.bmm(changes.transpose(1, 2), changes) / changes.shape[1]
    return standalone, interaction


def pair_statistics(standalone, interaction, baseline_loss, *, relative_tolerance=1e-10):
    """One row per source batch. Test ordered r->s despite symmetric I_rs.

    Frequency is conditional on both A_r and A_s being negative. Severity is
    summed positive marginal harm divided by summed standalone benefit of the
    SECOND neuron over ALL eligible ordered pairs. Event severity conditions
    both numerator and denominator on the harmful cases instead.
    """
    standalone, interaction = standalone.double(), interaction.double()
    batches, width = standalone.shape
    tolerance = relative_tolerance * baseline_loss.double()
    helpful = standalone < -tolerance[:, None]
    off_diagonal = ~torch.eye(width, dtype=torch.bool)
    eligible = helpful[:, :, None] & helpful[:, None, :] & off_diagonal
    # Rows r are applied first; columns s are the candidate second update.
    marginal = standalone[:, None, :] + interaction
    harmful = eligible & (marginal > tolerance[:, None, None])
    harm = torch.where(harmful, marginal, 0.0)
    second_benefit = -standalone[:, None, :].expand(-1, width, -1)
    eligible_benefit = torch.where(eligible, second_benefit, 0.0)
    event_benefit = torch.where(harmful, second_benefit, 0.0)
    upper = torch.triu(torch.ones(width, width, dtype=torch.bool), diagonal=1)

    def quotient(a, b):
        return torch.where(b > 0, a / b, float("nan"))

    n_eligible = eligible.sum((1, 2))
    n_events = harmful.sum((1, 2))
    harm_sum = harm.sum((1, 2))
    benefit_sum = eligible_benefit.sum((1, 2))
    per_batch = {
        "ordered_pairs": torch.full((batches,), width * (width - 1)),
        "eligible_count": n_eligible,
        "event_count": n_events,
        "eligible_fraction": n_eligible.double() / (width * (width - 1)),
        "frequency": quotient(n_events.double(), n_eligible),
        "harm_sum": harm_sum,
        "eligible_benefit_sum": benefit_sum,
        "event_benefit_sum": event_benefit.sum((1, 2)),
        "severity": quotient(harm_sum, benefit_sum),
        "event_severity": quotient(harm_sum, event_benefit.sum((1, 2))),
        "helpful_neuron_fraction": helpful.double().mean(1),
        "positive_interaction_fraction": (interaction[:, upper] > tolerance[:, None])
        .double()
        .mean(1),
        "unordered_event_count": ((harmful | harmful.transpose(1, 2)) & upper).sum((1, 2)),
        "unordered_eligible_count": (eligible & upper).sum((1, 2)),
    }
    per_pair = {
        "eligible_count": eligible.sum(0),
        "event_count": harmful.sum(0),
        "harm_sum": harm.sum(0),
        "eligible_benefit_sum": eligible_benefit.sum(0),
    }
    return per_batch, per_pair


def summarize_batches(arrays):
    """Pool batches within a training seed; never average unstable per-pair ratios."""

    def quotient(a, b):
        return a / b if b > 0 else math.nan

    sums = {k: v.double().sum().item() for k, v in arrays.items()}
    result = {
        k: sums[k]
        for k in [
            "ordered_pairs",
            "eligible_count",
            "event_count",
            "harm_sum",
            "eligible_benefit_sum",
            "event_benefit_sum",
        ]
    }
    result.update(
        {
            "frequency": quotient(sums["event_count"], sums["eligible_count"]),
            "severity": quotient(sums["harm_sum"], sums["eligible_benefit_sum"]),
            "event_severity": quotient(sums["harm_sum"], sums["event_benefit_sum"]),
            "events_per_batch": sums["event_count"] / len(arrays["event_count"]),
            "harm_per_batch": sums["harm_sum"] / len(arrays["harm_sum"]),
            "event_benefit_share": quotient(
                sums["event_benefit_sum"], sums["eligible_benefit_sum"]
            ),
            "eligible_fraction": quotient(sums["eligible_count"], sums["ordered_pairs"]),
            "unordered_frequency": quotient(
                sums["unordered_event_count"], sums["unordered_eligible_count"]
            ),
            "helpful_neuron_fraction": arrays["helpful_neuron_fraction"].mean().item(),
            "positive_interaction_fraction": arrays["positive_interaction_fraction"].mean().item(),
        }
    )
    return result
