"""Population MoE probe for delayed modules and destructive specialization.

Three separated mixture components require incompatible scalar regressors.  The
router is a softmax over trainable one-dimensional expert centers.  Width three
can represent the three component slopes exactly.  The experiment asks whether
imbalanced early gradients can collapse the router before the weaker components
become relatively dominant, and whether one redundant expert prevents that loss.

This is an exploratory theory probe, not confirmatory repository evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

COMPONENT_LOCATIONS = torch.tensor([-2.0, 0.0, 2.0])
COMPONENT_MASSES = torch.tensor([0.7, 0.2, 0.1])
TARGET_SLOPES = torch.tensor([1.0, -1.0, 2.0])


def maximum_matching_size(option_masks: np.ndarray, role_count: int = 3) -> int:
    reachable = {0}
    for raw_options in option_masks:
        options = int(raw_options)
        updated = set(reachable)
        for assigned in reachable:
            available = options & ~assigned
            while available:
                bit = available & -available
                updated.add(assigned | bit)
                available -= bit
        reachable = updated
    return max(mask.bit_count() for mask in reachable)


def gate_matrix(centers: torch.Tensor, temperature: float) -> torch.Tensor:
    squared_distance = (COMPONENT_LOCATIONS[:, None] - centers[None, :]) ** 2
    return torch.softmax(-squared_distance / (2.0 * temperature**2), dim=1)


def option_masks(gates: np.ndarray, relative_threshold: float = 0.2) -> np.ndarray:
    # An expert is provisionally eligible when its router weight is a fixed
    # fraction of the strongest expert for that component.  A relative threshold
    # avoids mechanically deleting every edge as width grows.  This is only an
    # exploratory proxy: the intended theorem must derive reachability from the
    # center dynamics rather than select a gate threshold.
    masks = np.zeros(gates.shape[1], dtype=np.int64)
    for component in range(gates.shape[0]):
        cutoff = relative_threshold * gates[component].max()
        masks[gates[component] >= cutoff] |= 1 << component
    return masks


@dataclass
class Snapshot:
    step: int
    loss: float
    component_loss: np.ndarray
    weighted_residual: np.ndarray
    centers: np.ndarray
    slopes: np.ndarray
    gates: np.ndarray
    smallest_gate_singular_value: float
    hall_deficit: int


@dataclass
class Run:
    seed: int
    width: int
    train_centers: bool
    snapshots: list[Snapshot]

    @property
    def final(self) -> Snapshot:
        return self.snapshots[-1]


def train(
    seed: int,
    width: int,
    *,
    train_centers: bool,
    temperature: float = 0.7,
    learning_rate: float = 0.025,
    steps: int = 1_500,
) -> Run:
    generator = torch.Generator().manual_seed(seed)
    centers = (torch.rand(width, generator=generator) * 5.0 - 2.5).requires_grad_(
        train_centers
    )
    slopes = (0.05 * torch.randn(width, generator=generator)).requires_grad_()
    parameters = [slopes] + ([centers] if train_centers else [])
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    record_steps = {0, 10, 30, 100, 300, 800, steps}
    snapshots: list[Snapshot] = []

    for step in range(steps + 1):
        gates = gate_matrix(centers, temperature)
        predicted_slopes = gates @ slopes
        residual = TARGET_SLOPES - predicted_slopes
        component_loss = 0.5 * residual**2
        loss = torch.sum(COMPONENT_MASSES * component_loss)

        if step in record_steps:
            gates_np = gates.detach().numpy().copy()
            masks = option_masks(gates_np)
            snapshots.append(
                Snapshot(
                    step=step,
                    loss=float(loss.detach()),
                    component_loss=component_loss.detach().numpy().copy(),
                    weighted_residual=(
                        COMPONENT_MASSES * torch.abs(residual)
                    ).detach().numpy().copy(),
                    centers=centers.detach().numpy().copy(),
                    slopes=slopes.detach().numpy().copy(),
                    gates=gates_np,
                    smallest_gate_singular_value=float(
                        np.linalg.svd(gates_np, compute_uv=False)[-1]
                    ),
                    hall_deficit=3 - maximum_matching_size(masks),
                )
            )

        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return Run(seed, width, train_centers, snapshots)


def summarize(runs: list[Run], failure_threshold: float = 1e-3) -> None:
    print(
        "width,train_centers,mean_final_loss,median_final_loss,failure_rate,"
        "initial_hall_failure_rate,final_hall_failure_rate,"
        "hall_collapse_rate,hall_repair_rate,mean_smallest_gate_singular_value"
    )
    for width in sorted({run.width for run in runs}):
        for train_centers in (False, True):
            group = [
                run
                for run in runs
                if run.width == width and run.train_centers == train_centers
            ]
            losses = np.array([run.final.loss for run in group])
            initial_deficits = np.array([run.snapshots[0].hall_deficit for run in group])
            final_deficits = np.array([run.final.hall_deficit for run in group])
            singular_values = np.array(
                [run.final.smallest_gate_singular_value for run in group]
            )
            print(
                f"{width},{train_centers},{losses.mean():.8f},"
                f"{np.median(losses):.8f},{np.mean(losses > failure_threshold):.4f},"
                f"{np.mean(initial_deficits > 0):.4f},"
                f"{np.mean(final_deficits > 0):.4f},"
                f"{np.mean((initial_deficits == 0) & (final_deficits > 0)):.4f},"
                f"{np.mean((initial_deficits > 0) & (final_deficits == 0)):.4f},"
                f"{singular_values.mean():.6f}"
            )

    print("\nwidth,dynamically_harmed_rate,dynamically_helped_rate")
    for width in sorted({run.width for run in runs}):
        frozen = {
            run.seed: run.final.loss
            for run in runs
            if run.width == width and not run.train_centers
        }
        trained = {
            run.seed: run.final.loss
            for run in runs
            if run.width == width and run.train_centers
        }
        seeds = sorted(set(frozen) & set(trained))
        harmed = np.mean(
            [frozen[seed] <= failure_threshold < trained[seed] for seed in seeds]
        )
        helped = np.mean(
            [trained[seed] <= failure_threshold < frozen[seed] for seed in seeds]
        )
        print(f"{width},{harmed:.4f},{helped:.4f}")


def print_trace(run: Run) -> None:
    print(f"\nTRACE seed={run.seed} width={run.width} train_centers={run.train_centers}")
    print(
        "step,loss,hall_deficit,min_gate_sv,dominant_weighted_residual,"
        "weighted_residuals,centers"
    )
    for snapshot in run.snapshots:
        dominant = int(np.argmax(snapshot.weighted_residual))
        residuals = "/".join(f"{value:.3f}" for value in snapshot.weighted_residual)
        centers = "/".join(f"{value:.3f}" for value in snapshot.centers)
        print(
            f"{snapshot.step},{snapshot.loss:.6f},{snapshot.hall_deficit},"
            f"{snapshot.smallest_gate_singular_value:.6f},{dominant},"
            f"{residuals},{centers}"
        )


def main() -> None:
    runs = [
        train(seed, width, train_centers=train_centers)
        for seed in range(50)
        for width in (3, 4, 6)
        for train_centers in (False, True)
    ]
    summarize(runs)

    # Seed 3 is specified prospectively in the script as a readable example of
    # router movement destroying a well-conditioned initial width-three solution.
    print_trace(next(run for run in runs if run.seed == 3 and run.width == 3 and run.train_centers))
    print_trace(next(run for run in runs if run.seed == 3 and run.width == 3 and not run.train_centers))


if __name__ == "__main__":
    main()
