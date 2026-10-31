"""Exact coherent-pressure transport and Gaussian option-loss probabilities.

These are analytic model calculations, not fits to trained-network outcomes.
The time variable is cumulative common pressure, not ordinary training time.
"""

from __future__ import annotations

import numpy as np


def transport(w, a, exposure):
    """Flow under one unit-norm common input (the first coordinate).

    Use the absorbing zero-subgradient solution at a common-gate hit.
    Other coordinates stay fixed. Arrays can have any leading batch shape.
    """
    if exposure < 0:
        raise ValueError("Exposure must be nonnegative.")
    w, a = np.asarray(w, dtype=float), np.asarray(a, dtype=float)
    out_w, out_a = w.copy(), a.copy()
    active = w[..., 0] > 0
    along = w[..., 0] * np.cosh(exposure) + a * np.sinh(exposure)
    readout = a * np.cosh(exposure) + w[..., 0] * np.sinh(exposure)
    hit = active & (along <= 0)
    out_w[..., 0] = np.where(active, np.maximum(along, 0), w[..., 0])
    out_a[...] = np.where(active, readout, a)
    out_a[...] = np.where(hit, -np.sqrt(np.maximum(a * a - w[..., 0] ** 2, 0)), out_a)
    return out_w, out_a


def blind_probability(theta, exposure):
    """Exact single-block blindness for v±=(-cos(theta), ±sin(theta))."""
    if not 0 < theta < np.pi / 2 or exposure < 0:
        raise ValueError("Require 0 < theta < pi/2 and nonnegative exposure.")
    tangent2 = np.tan(theta) ** 2
    variance = np.cosh(2 * exposure) + tangent2
    rho_w = np.cosh(exposure) / np.sqrt(variance)
    rho_pair = (np.cosh(2 * exposure) - tangent2) / variance
    return float(
        1 / 8
        + (2 * np.arcsin(np.clip(rho_w, -1, 1)) + np.arcsin(np.clip(rho_pair, -1, 1)))
        / (4 * np.pi)
    )


def blind_joint_probability(theta, exposure):
    """P(blind initially AND blind at exposure), by a polar antiderivative.

    atan2/hypot avoid loss of accuracy near the zero-exposure endpoint.
    """
    blind_probability(theta, exposure)  # Validate arguments.
    if exposure == 0:
        return 0.5 - theta / np.pi
    limit = np.pi / 2 - theta
    c, s, tangent = np.cosh(exposure), np.sinh(exposure), np.tan(theta)
    end = np.arctan2(c * np.sin(limit) + tangent * np.cos(limit),
                    np.hypot(s, c * np.cos(limit) - tangent * np.sin(limit)))
    start = np.arctan2(tangent, np.hypot(s, c))
    return float((limit + end - start) / (2 * np.pi))


def transition_probabilities(theta, exposure, width):
    """Network null-response events; 'open' means positive response to d.

    This does not equate positive response with complete span coverage.
    Endpoint recovery does not certify uninterrupted survival between times.
    """
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("Width must be a positive integer.")
    b0 = blind_probability(theta, 0)
    bs = blind_probability(theta, exposure)
    b00 = blind_joint_probability(theta, exposure)
    closed = bs**width
    newly_closed = max(closed - b00**width, 0)
    reopened = max(b0**width - b00**width, 0)
    return dict(
        block_blind=bs,
        block_blind_both=b00,
        closed=closed,
        newly_closed=newly_closed,
        closed_given_initially_open=newly_closed / (1 - b0**width),
        reopened=reopened,
        open_given_initially_closed=reopened / b0**width,
    )


def contrast_kernel(w, a, theta, q):
    """Per-block r=d^T Z d for normalized rare contrast in weighted space."""
    probes = np.array([[-np.cos(theta), np.sin(theta)], [-np.cos(theta), -np.sin(theta)]])
    pre = np.asarray(w) @ probes.T
    gate = pre > 0
    h = np.maximum(pre, 0)
    incoming = gate[..., 0, None] * probes[0] - gate[..., 1, None] * probes[1]
    return q / 4 * ((h[..., 0] - h[..., 1]) ** 2 + a**2 * np.sum(incoming**2, axis=-1))
