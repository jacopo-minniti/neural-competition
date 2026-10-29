"""Theory-led probes of activation erosion for default-with-exception tasks.

No fitted roles, gate cutoffs, or endpoint-estimated survival probabilities.
Population inputs are k common prototypes e_i and one exception
(1,...,1, epsilon)/sqrt(k+epsilon**2). Targets after subtracting a known
default are 0 on the common prototypes and -1 on the exception.
The correction network is -mean(ReLU(w_r @ x)), with hidden learning rate m.
This deliberately restrictive architecture isolates activation loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def projected(weights: np.ndarray, k: int) -> np.ndarray:
    result = weights.copy()
    result[..., :k] = np.minimum(result[..., :k], 0)
    return result


def exception_margin(weights: np.ndarray, k: int, epsilon: float) -> np.ndarray:
    return (weights[..., :k].sum(axis=-1) + epsilon * weights[..., k]) / np.sqrt(
        k + epsilon**2
    )


def loss_and_gradient(weights, k, epsilon, mass, leak=0.0):
    common = weights[..., :k]
    special = exception_margin(weights, k, epsilon)
    act_c = np.maximum(common, leak * common)
    act_s = np.maximum(special, leak * special)
    # Residuals prediction minus target; constant default cancels exactly.
    residual_c = -act_c.mean(axis=1)
    residual_s = 1 - act_s.mean(axis=1)
    grad_c = np.where(common > 0, 1.0, leak)
    grad_s = np.where(special > 0, 1.0, leak)
    # Negative gradient times width: mean-field time, not a wider learning rate.
    update = np.zeros_like(weights)
    update[..., :k] = (1 - mass) / k * residual_c[:, None, :] * grad_c
    pressure = mass * residual_s[:, None] * grad_s / np.sqrt(k + epsilon**2)
    update[..., :k] += pressure[..., None]
    update[..., k] = epsilon * pressure
    loss = 0.5 * ((1 - mass) / k * (residual_c**2).sum(axis=1)
                  + mass * residual_s**2)
    return loss, update


def train(weights, k, epsilon, mass, horizon, dt, leak=0.0, record=False):
    weights = weights.copy()
    trace = []
    steps = round(horizon / dt)
    checkpoints = {0, steps // 100, steps // 10, steps // 2, steps}
    for step in range(steps + 1):
        loss, update = loss_and_gradient(weights, k, epsilon, mass, leak)
        if record and step in checkpoints:
            alive = (exception_margin(weights, k, epsilon) > 0).sum(axis=1)
            trace.append(dict(time=step * dt, mean_loss=float(loss.mean()),
                              no_exception_activation=float(np.mean(alive == 0))))
        if step < steps:
            weights += dt * update
    return weights, loss, trace


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, se=se, normal_95=[mean - 1.96 * se, mean + 1.96 * se])


def run(args):
    output = dict(protocol={
        "seed": args.seed, "seeds": args.seeds, "dt": args.dt,
        "horizon": args.horizon, "initialization_radius": 0.2,
        "exception_mass": 0.1, "widths": [1, 2, 4, 8, 16],
        "prediction": "No surviving exception activation after common fitting is absorbing; "
                      "its probability is (1-p)^m and its exact half-MSE is mass/2.",
        "status": "Exploratory; analytical predictions specified before these simulations.",
    }, geometry=[], training=[], controls=[], stationary_rare=[], negative_candidates={})
    raw = {}
    # Separate initialization-only integration stream; never fit p to endpoints.
    geometry_rng = np.random.default_rng(args.seed + 1)
    geometry_ps = {}
    for k, epsilon in [(1, 1.0), (1, 0.25), (2, 1.0), (4, 1.0), (4, 0.25)]:
        samples = geometry_rng.normal(size=(args.geometry_samples, k + 1))
        margins = exception_margin(projected(samples, k), k, epsilon)
        p_hat = float(np.mean(margins > 0))
        p_exact = 0.25 + np.arctan(epsilon) / (2 * np.pi) if k == 1 else None
        p = p_exact if p_exact is not None else p_hat
        geometry_ps[k, epsilon] = p
        output["geometry"].append(dict(k=k, epsilon=epsilon, p_mc=p_hat,
            p_exact=p_exact, samples=args.geometry_samples,
            se=float(np.sqrt(p_hat * (1-p_hat) / args.geometry_samples)),
            small_epsilon_first_order=2.0 ** (-k-1) + k * epsilon / (2**k * np.pi)))

    # Nested weights and paired seeds within each k. All results retained.
    for k in (1, 4):
        epsilon = 1.0
        rng = np.random.default_rng(args.seed + 100 + k)
        initial = rng.normal(size=(args.seeds, 16, k + 1))
        initial *= 0.2 / np.linalg.norm(initial, axis=-1, keepdims=True)
        for width in (1, 2, 4, 8, 16):
            start = initial[:, :width].copy()
            pre = projected(start, k)
            dead0 = (exception_margin(start, k, epsilon) <= 0).all(axis=1)
            dead = (exception_margin(pre, k, epsilon) <= 0).all(axis=1)
            for schedule in ("staged", "stationary"):
                trained, loss, trace = train(
                    pre if schedule == "staged" else start,
                    k, epsilon, 0.1, args.horizon, args.dt, record=True)
                p = geometry_ps[k, epsilon]
                entry = dict(k=k, epsilon=epsilon, width=width, schedule=schedule,
                    predicted_extinction=(1-p)**width,
                    predicted_dynamic_extinction=(1-p)**width - 0.5**width,
                    initial_no_activation=mean_ci(dead0),
                    projected_extinction=mean_ci(dead),
                    newly_extinguished=mean_ci(dead & ~dead0),
                    loss=mean_ci(loss),
                    predicted_loss_lower_bound=0.05 * (1-p)**width,
                    loss_if_projected_alive=float(loss[~dead].mean()),
                    loss_if_projected_dead=float(loss[dead].mean()) if dead.any() else None,
                    final_all_inputs_dead=float(np.mean(
                        (trained[..., :k] <= 0).all(axis=(1, 2)) &
                        (exception_margin(trained, k, epsilon) <= 0).all(axis=1))),
                    trace=trace)
                final_dead = (exception_margin(trained, k, epsilon) <= 0).all(axis=1)
                entry["final_no_exception_activation"] = mean_ci(final_dead)
                entry["actual_dynamic_extinction"] = mean_ci(final_dead & ~dead0)
                # Positive projected margins are a priori protected, even on
                # stationary trajectories. This is stronger than loss correlation.
                protected = exception_margin(pre, k, epsilon) > 0
                assert np.all(exception_margin(trained, k, epsilon)[protected] > 0)
                output["training"].append(entry)
                raw[f"k{k}_m{width}_{schedule}_loss"] = loss
                raw[f"k{k}_m{width}_initial_dead"] = dead0
                raw[f"k{k}_m{width}_projected_dead"] = dead
                raw[f"k{k}_m{width}_{schedule}_final_dead"] = final_dead
                print(json.dumps({key: entry[key] for key in
                    ("k", "width", "schedule", "predicted_extinction", "loss",
                     "loss_if_projected_alive", "final_all_inputs_dead")}), flush=True)
        # Structural controls at minimal width, paired with the same start.
        start = initial[:, :1].copy()
        pre = projected(start, k)
        for name, weights, leak in [
            ("leak_after_erosion", pre, 0.1),
            ("orient_novel_coordinate", np.concatenate(
                [pre[..., :k], np.abs(pre[..., k:])], axis=-1), 0.0),
            ("novel_only_fresh_unit", np.concatenate(
                [np.zeros_like(pre[..., :k]), np.abs(pre[..., k:])], axis=-1), 0.0),
        ]:
            _, loss, trace = train(weights, k, epsilon, 0.1,
                                    args.horizon, args.dt, leak, True)
            output["controls"].append(dict(k=k, name=name, loss=mean_ci(loss), trace=trace))
            raw[f"k{k}_{name}_loss"] = loss
        # ReLU discretization/horizon checks use the same 32 seeds, not extra evidence.
        for name, horizon, dt in [("half_step", args.horizon, args.dt / 2),
                                   ("fourfold_horizon", args.horizon * 4, args.dt)]:
            _, loss, _ = train(pre[:32], k, epsilon, 0.1, horizon, dt)
            output["controls"].append(dict(k=k, name=name, seeds=min(32, args.seeds),
                                          loss=mean_ci(loss)))
        wide_pre = projected(initial[:128], k)
        _, loss, _ = train(wide_pre, k, epsilon, 0.1, args.horizon * 4, args.dt)
        wide_dead = (exception_margin(wide_pre, k, epsilon) <= 0).all(axis=1)
        output["controls"].append(dict(k=k, name="wide_fourfold_horizon", width=16,
            seeds=len(wide_pre), loss=mean_ci(loss),
            seed_specific_floor=float(0.05 * wide_dead.mean()),
            live_loss=float(loss[~wide_dead].mean())))
        fitted, _, _ = train(initial[:32, :4], k, epsilon, 0.0, 1000, args.dt)
        projection_error = float(np.max(np.abs(fitted - projected(initial[:32, :4], k))))
        # Euler steps can overshoot a ReLU boundary. Report, do not hide, that
        # discrepancy from the exact continuous-time common-task limit.
        output["controls"].append(dict(k=k, name="numerical_common_pretraining",
                                      max_projection_error=projection_error))
        # No staged training here: test whether rare exceptions approximate the
        # erosion limit. Compare normalized risk because rarity also reduces damage.
        for mass in (0.01, 0.001):
            for width in (1, 4):
                w = initial[:128, :width].copy()
                trained, loss, trace = train(w, k, epsilon, mass,
                    horizon=20 / mass, dt=0.5, record=True)
                dead0 = (exception_margin(w, k, epsilon) <= 0).all(axis=1)
                dead_final = (exception_margin(trained, k, epsilon) <= 0).all(axis=1)
                output["stationary_rare"].append(dict(k=k, mass=mass, width=width,
                    seeds=len(w), horizon=20/mass, dt=0.5,
                    projected_limit_prediction=(1-geometry_ps[k, epsilon])**width,
                    loss=mean_ci(loss), normalized_loss=mean_ci(loss / (0.5 * mass)),
                    no_exception_activation=mean_ci(dead_final),
                    dynamic_extinction=mean_ci(dead_final & ~dead0), trace=trace))
                raw[f"k{k}_m{width}_q{mass}_rare_loss"] = loss
        # Small-step audit of the hardest stationary extrapolation, same seeds.
        for dt in (0.5, 0.25):
            w = initial[:32, :4]
            trained, loss, _ = train(w, k, epsilon, 0.001, 20000, dt)
            output["controls"].append(dict(k=k, name="rare_step_audit", dt=dt,
                width=4, seeds=len(w), loss=mean_ci(loss),
                final_no_exception_activation=float(np.mean(
                    (exception_margin(trained, k, epsilon) <= 0).all(axis=1)))))

    # Negative candidate A: factorized scalar modes with fixed aggregate initial energy.
    # q_i= sum a_ir^2, f_i=q_i, da_ir=(s_i-q_i)a_ir.
    # Replicating each factor 1/sqrt(m) leaves q_i and its ODE invariant.
    strengths = np.array([1.0, 0.1])
    factorized = []
    for width in (1, 4, 16):
        a = np.full((2, width), np.sqrt(0.001 / width))
        for _ in range(20000):
            a += 0.01 * (strengths - (a*a).sum(axis=1))[:, None] * a
        factorized.append(dict(copies_per_mode=width, predictions=(a*a).sum(axis=1).tolist()))
    output["negative_candidates"]["factorized_delay"] = factorized
    # Negative candidate B: uniform attenuation with fixed total feature energy.
    # bdot=-g b, g=sum_j (a/sqrt(m))^2=a^2; width cannot repair attenuation.
    output["negative_candidates"]["uniform_attenuation"] = [
        dict(width=m, gramian=float(np.sum(np.full(m, 0.1 / np.sqrt(m))**2)),
             loss_at_100=float(0.5 * np.exp(-2 * np.sum(
                 np.full(m, 0.1 / np.sqrt(m))**2) * 100))) for m in (1, 4, 16)]
    # Analytic gradient check away from gates.
    rng = np.random.default_rng(801)
    w = rng.normal(size=(1, 3, 3))
    _, update = loss_and_gradient(w, 2, 1.0, 0.1)
    errors = []
    for index in np.ndindex(w.shape):
        wp, wm = w.copy(), w.copy()
        wp[index] += 1e-6
        wm[index] -= 1e-6
        numerical = (loss_and_gradient(wp, 2, 1.0, 0.1)[0][0] -
                     loss_and_gradient(wm, 2, 1.0, 0.1)[0][0]) / 2e-6
        errors.append(abs(numerical + update[index] / 3))
    output["gradient_check_max_error"] = max(errors)
    assert max(errors) < 1e-8
    output["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **raw)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--geometry-samples", type=int, default=1_000_000)
    parser.add_argument("--horizon", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path(
        "docs/theory/innovation_survival_results.json"))
    run(parser.parse_args())
