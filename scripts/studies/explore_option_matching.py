"""Exploratory tests for the task-structured option-matching abstraction.

This is deliberately not a neural-network benchmark.  It tests whether a
matching/Hall-slack object makes predictions that per-role survival rates cannot,
and whether online early commitment can destroy an initially feasible allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np


def maximum_matching_size(option_masks: np.ndarray, role_mask: int) -> int:
    """Maximum unit-capacity matching for a small bit-masked role set."""
    reachable = {0}
    for raw_options in option_masks:
        options = int(raw_options) & role_mask
        updated = set(reachable)
        for assigned in reachable:
            available = options & ~assigned
            while available:
                bit = available & -available
                updated.add(assigned | bit)
                available -= bit
        reachable = updated
    return max(mask.bit_count() for mask in reachable)


def hall_slack(option_masks: np.ndarray, role_count: int) -> int:
    """Minimum |neighbors(S)|-|S| over nonempty role subsets S."""
    best = len(option_masks)
    for subset in range(1, 1 << role_count):
        neighbors = sum(bool(int(mask) & subset) for mask in option_masks)
        best = min(best, neighbors - subset.bit_count())
    return best


@dataclass(frozen=True)
class TypeDistribution:
    name: str
    masks: tuple[int, ...]
    probabilities: tuple[float, ...]

    def sample(self, rng: np.random.Generator, width: int) -> np.ndarray:
        return rng.choice(self.masks, size=width, p=self.probabilities)

    def marginal_role_probability(self, role: int) -> float:
        bit = 1 << role
        return sum(p for mask, p in zip(self.masks, self.probabilities) if mask & bit)

    def bottleneck_ratio(self, role_count: int) -> tuple[float, int, float]:
        """Return max_S |S|/P(a neuron can serve something in S)."""
        best = (-np.inf, 0, 0.0)
        for subset in range(1, 1 << role_count):
            union_probability = sum(
                p
                for mask, p in zip(self.masks, self.probabilities)
                if mask & subset
            )
            ratio = subset.bit_count() / union_probability
            best = max(best, (ratio, subset, union_probability))
        return best

    def hall_failure_bounds(self, width: int, role_count: int) -> tuple[float, float]:
        """Lower/max-event and upper/union bounds for Hall failure."""
        event_probabilities = []
        for subset in range(1, 1 << role_count):
            union_probability = sum(
                p
                for mask, p in zip(self.masks, self.probabilities)
                if mask & subset
            )
            demand = subset.bit_count()
            probability = sum(
                comb(width, count)
                * union_probability**count
                * (1.0 - union_probability) ** (width - count)
                for count in range(min(demand, width + 1))
            )
            event_probabilities.append(probability)
        return max(event_probabilities), min(1.0, sum(event_probabilities))

    def exact_failure_probability(self, width: int, role_mask: int) -> float:
        """Enumerate multinomial type counts; used only for tiny toy models."""
        type_count = len(self.masks)
        failure_probability = 0.0

        def visit(prefix: list[int], remaining: int) -> None:
            nonlocal failure_probability
            if len(prefix) == type_count - 1:
                counts = prefix + [remaining]
                coefficient = factorial(width)
                probability = 1.0
                options: list[int] = []
                for count, mask, type_probability in zip(
                    counts, self.masks, self.probabilities
                ):
                    coefficient //= factorial(count)
                    probability *= type_probability**count
                    options.extend([mask] * count)
                if maximum_matching_size(np.asarray(options), role_mask) < role_mask.bit_count():
                    failure_probability += coefficient * probability
                return
            for count in range(remaining + 1):
                visit(prefix + [count], remaining - count)

        visit([], width)
        return failure_probability


def static_overlap_experiment(rng: np.random.Generator, trials: int) -> None:
    role_count = 4
    all_roles = (1 << role_count) - 1
    scenarios = (
        TypeDistribution("specialists", (1, 2, 4, 8), (0.25, 0.25, 0.25, 0.25)),
        TypeDistribution("paired_flexible", (0, 3, 12), (0.50, 0.25, 0.25)),
        TypeDistribution("all_flexible", (0, 15), (0.75, 0.25)),
    )

    print("STATIC OVERLAP: every role has marginal viability p=0.25")
    for scenario in scenarios:
        marginals = [scenario.marginal_role_probability(j) for j in range(role_count)]
        ratio, subset, union_probability = scenario.bottleneck_ratio(role_count)
        print(
            f"{scenario.name:16s} marginals={marginals} "
            f"Hall_bottleneck_roles={subset:04b} "
            f"union_p={union_probability:.2f} ratio={ratio:.1f}"
        )

    print("width,scenario,success_probability,mean_hall_deficit")
    for width in (4, 6, 8, 10, 12, 16, 20, 24, 32, 40):
        for scenario in scenarios:
            successes = 0
            deficits = 0
            for _ in range(trials):
                options = scenario.sample(rng, width)
                matched = maximum_matching_size(options, all_roles)
                successes += matched == role_count
                deficits += role_count - matched
            print(
                f"{width},{scenario.name},{successes / trials:.5f},"
                f"{deficits / trials:.5f}"
            )

    print("\nwidth,scenario,actual_failure,hall_lower_bound,hall_union_bound")
    for width in (8, 12, 20):
        for scenario in scenarios:
            lower, upper = scenario.hall_failure_bounds(width, role_count)
            exact_failure = scenario.exact_failure_probability(width, all_roles)
            print(
                f"{width},{scenario.name},{exact_failure:.5f},"
                f"{lower:.5f},{upper:.5f}"
            )


def choose_early_neuron(
    options: np.ndarray, rng: np.random.Generator, scarcity_aware: bool
) -> int | None:
    candidates = np.flatnonzero(options & 1)
    if len(candidates) == 0:
        return None
    if not scarcity_aware:
        return int(rng.choice(candidates))
    degrees = np.array([int(options[index]).bit_count() for index in candidates])
    least_flexible = candidates[degrees == degrees.min()]
    return int(rng.choice(least_flexible))


def online_commitment_experiment(rng: np.random.Generator, trials: int) -> None:
    # Roles: E is released first; L1 and L2 are released later.  A flexible neuron
    # can ultimately serve any one role, but specialization is exclusive.
    distribution = TypeDistribution(
        "early_late",
        (1, 2, 4, 7),
        (0.35, 0.20, 0.20, 0.25),
    )
    all_roles = 0b111
    late_roles = 0b110

    print("\nONLINE COMMITMENT: E releases before L1,L2")
    print(
        "width,offline_feasible_probability,random_success_probability,"
        "scarcity_success_probability,random_failure_given_offline_feasible,"
        "mean_initial_hall_slack"
    )
    for width in range(3, 13):
        offline_feasible = 0
        random_success = 0
        scarcity_success = 0
        random_failure_with_option = 0
        slack_sum = 0
        for _ in range(trials):
            options = distribution.sample(rng, width)
            offline = maximum_matching_size(options, all_roles) == 3
            offline_feasible += offline
            slack_sum += hall_slack(options, 3)

            random_early = choose_early_neuron(options, rng, scarcity_aware=False)
            if random_early is None:
                random_ok = False
            else:
                remaining = np.delete(options, random_early)
                random_ok = maximum_matching_size(remaining, late_roles) == 2
            random_success += random_ok
            random_failure_with_option += offline and not random_ok

            scarcity_early = choose_early_neuron(options, rng, scarcity_aware=True)
            if scarcity_early is None:
                scarcity_ok = False
            else:
                remaining = np.delete(options, scarcity_early)
                scarcity_ok = maximum_matching_size(remaining, late_roles) == 2
            scarcity_success += scarcity_ok

        conditional_failure = (
            random_failure_with_option / offline_feasible
            if offline_feasible
            else float("nan")
        )
        print(
            f"{width},{offline_feasible / trials:.5f},"
            f"{random_success / trials:.5f},{scarcity_success / trials:.5f},"
            f"{conditional_failure:.5f},{slack_sum / trials:.5f}"
        )


def main() -> None:
    rng = np.random.default_rng(20261023)
    static_overlap_experiment(rng, trials=20_000)
    online_commitment_experiment(rng, trials=40_000)


if __name__ == "__main__":
    main()
