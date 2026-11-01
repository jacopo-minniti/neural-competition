"""Independent binary features with heterogeneous frequencies and parity labels."""

from __future__ import annotations

import itertools

import numpy as np


def frequencies(n_features: int = 8, alpha: float = 1.0, maximum: float = 0.4) -> np.ndarray:
    """Return rank-power-law activation probabilities, with q[0] = maximum."""
    if not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(maximum) or not 0 < maximum < 1:
        raise ValueError("alpha must be nonnegative and maximum must be in (0, 1)")
    return maximum * np.arange(1, n_features + 1, dtype=float) ** (-alpha)


def _probabilities(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not q.size or not np.all(np.isfinite(q)) or np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must be a nonempty vector with entries strictly between 0 and 1")
    return q


def sample(q: np.ndarray, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw fresh examples; y is odd parity in {0,1}, suitable for squared loss."""
    q = _probabilities(q)
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    x = (rng.random((count, len(q))) < q).astype(float)
    return {"x": x, "y": (x.sum(axis=1) % 2).astype(float)}


def panels(q: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Enumerate support with population weights and a fixed uniform audit measure.

    The two panels share inputs and labels, but have distinct probability weights.
    Enumeration is intended for small dimensions (eight by default).
    """
    q = _probabilities(q)
    if len(q) > 20:
        raise ValueError("exact enumeration is limited to 20 features")
    x = np.array(list(itertools.product((0., 1.), repeat=len(q))))
    y = x.sum(axis=1) % 2
    p = np.prod(np.where(x == 1, q, 1 - q), axis=1)
    return {"x": x, "y": y, "p": p}, {"x": x.copy(), "y": y.copy(), "p": np.full(len(x), 1 / len(x))}
