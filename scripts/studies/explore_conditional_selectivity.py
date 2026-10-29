"""One jointly trained network: context-specific detail versus additive rules.

The task has four population atoms (R,z), with R rare and z balanced.
y=A+D R z. No supplied default, frozen signs, assigned specialists or router.
The missing additive-model interaction is s=(R-q)z/sqrt(q(1-q)).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def task(q=0.1, amplitude=2.0, detail=2.0):
    context = np.array([-1, -1, 1, 1.0])
    z = np.array([-1, 1, -1, 1.0])
    rare = (context + 1) / 2
    x = np.stack([context, z, np.ones(4)], axis=1)
    mass = np.array([(1-q)/2, (1-q)/2, q/2, q/2])
    basis = np.stack([np.ones(4), (rare-q)/np.sqrt(q*(1-q)), z,
                      (rare-q)*z/np.sqrt(q*(1-q))])
    y = amplitude + detail * rare * z
    assert np.allclose((basis * mass) @ basis.T, np.eye(4))
    return x, mass, basis, y


def probe(w, a, c, x, mass, basis, y, intercept_rate=1.0):
    m = a.shape[1]
    z = w @ x.T
    activation = np.maximum(z, 0)
    prediction = c[:, None] + (a[..., None] * activation).mean(axis=1)
    residual = prediction - y
    # Local derivatives of each particle contribution; sqrt(m) scaling gives
    # the actual mean-field metric kernel after averaging over particles.
    jac = np.concatenate([activation[..., None],
        a[..., None, None] * (z > 0)[..., None] * x[None, None]], axis=-1)
    moments = np.einsum('ln,smnd->slmd', basis * mass, jac)
    kernel = np.einsum('slmd,skmd->slk', moments, moments) / m
    kernel[:, 0, 0] += intercept_rate
    nuisance = kernel[:, :3, :3]
    beta = np.einsum('sij,sj->si', np.linalg.pinv(nuisance, rcond=1e-12),
                     kernel[:, :3, 3])
    kappa = kernel[:, 3, 3] - (beta * kernel[:, :3, 3]).sum(axis=1)
    contrast_residual = (residual * (basis[3] * mass)).sum(axis=1)
    feature_contrast = (activation * (basis[3] * mass)).sum(axis=2)
    return dict(loss=0.5*(mass*residual**2).sum(axis=1),
        contrast_loss=0.5*contrast_residual**2, kappa=np.maximum(kappa, 0),
        feature_signal=(feature_contrast**2).mean(axis=1),
        contrast_residual=contrast_residual, kernel=kernel,
        residual_modes=(residual*mass) @ basis.T,
        common_min_rate=np.linalg.eigvalsh(nuisance)[:, 0])


def train(w, a, x, mass, y, horizon=1000, dt=0.05, intercept_rate=1.0,
          basis=None, forecast_time=20):
    w, a = w.copy(), a.copy()
    c = np.zeros(len(w))
    checkpoints = {0, round(forecast_time/dt), round(horizon/dt)}
    snapshots = {}
    for step in range(round(horizon/dt)+1):
        if step in checkpoints:
            snapshots[step*dt] = probe(w, a, c, x, mass, basis, y, intercept_rate)
        if step == round(horizon/dt):
            break
        z = w @ x.T
        activation = np.maximum(z, 0)
        residual = c[:, None] + (a[..., None]*activation).mean(axis=1) - y
        outgoing_update = -(activation * (mass*residual)[:, None]).sum(axis=2)
        incoming_update = -a[..., None] * (((z > 0) * (mass*residual)[:, None]) @ x)
        c -= dt * intercept_rate * (mass*residual).sum(axis=1)
        a += dt * outgoing_update
        w += dt * incoming_update
    return snapshots


def main():
    x, mass, basis, y = task()
    rng = np.random.default_rng(82031)
    seeds = 128
    initial_w = .2 * rng.normal(size=(seeds, 12, 3))
    initial_a = .2 * rng.normal(size=(seeds, 12))
    rows, arrays = [], {}
    for intercept_rate in (1.0, 0.0):
        for width in (3, 6, 12):
            snapshots = train(initial_w[:, :width], initial_a[:, :width], x, mass, y,
                              basis=basis, intercept_rate=intercept_rate)
            final = snapshots[1000.0]
            row = dict(width=width, intercept_rate=intercept_rate,
                       mean_loss=float(final['loss'].mean()),
                       median_loss=float(np.median(final['loss'])), snapshots={})
            for time, snap in snapshots.items():
                row['snapshots'][time] = {key: float(value.mean()) for key, value in snap.items()}
                for key, value in snap.items():
                    arrays[f'c{intercept_rate}_m{width}_t{time}_{key}'] = value
            # A genuinely prospective local approximation: freeze the kernel
            # measured at t=20, not the one measured at convergence.
            early = snapshots[20.0]
            predicted = early['contrast_loss'] * np.exp(-2*980*early['kappa'])
            row['early_frozen_forecast_mae'] = float(np.abs(predicted-final['contrast_loss']).mean())
            row['early_forecast_mean'] = float(predicted.mean())
            row['final_contrast_loss'] = float(final['contrast_loss'].mean())
            eig, vectors = np.linalg.eigh(early['kernel'])
            coordinates = np.einsum('sji,sj->si', vectors, early['residual_modes'])
            full_forecast = 0.5*np.sum(coordinates**2*np.exp(-2*980*np.maximum(eig, 0)),axis=1)
            row['full_frozen_forecast_mean'] = float(full_forecast.mean())
            row['full_frozen_forecast_mae'] = float(np.abs(full_forecast-final['loss']).mean())
            arrays[f'c{intercept_rate}_m{width}_scalar_forecast'] = predicted
            arrays[f'c{intercept_rate}_m{width}_full_forecast'] = full_forecast
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Counterexample to activation-only diagnostics: a feature has zero
    # interaction moment, but its trainable incoming weights have a nonzero one.
    w = np.array([[[1., 0., 0.], [0., 0., 1.]]])
    a = np.ones((1, 2))
    counterexample = probe(w, a, np.zeros(1), x, mass, basis, y)
    # Width-three exact representation, with or without an output intercept:
    # A + D/2 [ReLU(c+z)-ReLU(c-z)]. One constant hidden unit supplies A.
    witness_w = np.array([[0., 0., 1.], [1., 1., 0.], [1., -1., 0.]])
    witness_a = np.array([6., 3., -3.])
    assert np.allclose((witness_a[:, None]*np.maximum(witness_w @ x.T, 0)).mean(0), y)
    controls = []
    for name, amplitude, horizon, dt, count in [
        ('remove_large_default', 0.0, 1000, .05, seeds),
        ('half_step', 2.0, 1000, .025, 32),
        ('longer', 2.0, 4000, .05, 32),
    ]:
        _, _, _, cy = task(amplitude=amplitude)
        snapshots = train(initial_w[:count, :3], initial_a[:count, :3], x, mass, cy,
                          horizon=horizon, dt=dt, basis=basis)
        end = snapshots[float(horizon)]
        controls.append(dict(name=name, seeds=count, amplitude=amplitude,
                             horizon=horizon, dt=dt,
                             mean_loss=float(end['loss'].mean()),
                             final_losses=end['loss'].tolist(),
                             early_mean_kappa=float(snapshots[20.0]['kappa'].mean())))
    # Finite-difference audit of the complete trainable network Jacobian/kernel.
    ww, aa, cc = initial_w[:1, :3].copy(), initial_a[:1, :3].copy(), np.zeros(1)
    reference = probe(ww, aa, cc, x, mass, basis, y)['kernel'][0]
    parameter = np.concatenate([aa.ravel(), ww.ravel(), cc])
    numerical_jac = []
    def predictions(param):
        wa, wi, bias = param[:3], param[3:-1].reshape(3, 3), param[-1]
        return bias + (wa[:, None]*np.maximum(wi@x.T, 0)).mean(0)
    for i in range(len(parameter)):
        plus, minus = parameter.copy(), parameter.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numerical_jac.append((predictions(plus)-predictions(minus))/2e-6)
    numerical_jac = np.array(numerical_jac).T
    numerical_moments = (basis*mass)@numerical_jac
    metric = np.diag([3.]*(len(parameter)-1)+[1.])
    kernel_error = float(np.max(np.abs(numerical_moments@metric@numerical_moments.T-reference)))
    assert kernel_error < 1e-8
    assert counterexample['feature_signal'][0] < 1e-20
    assert counterexample['kappa'][0] > 0.01
    result = dict(seed=82031, seeds=seeds, q=.1, amplitude=2, detail=2,
                  horizon=1000, dt=.05, forecast_time=20,
                  damage=0.5*4*.1*.9, rows=rows, controls=controls,
                  kernel_finite_difference_error=kernel_error,
                  activation_only_counterexample={k: v.tolist() for k, v in counterexample.items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(__file__).resolve().parents[2]/'docs/theory/conditional_selectivity_results.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'), **arrays)


if __name__ == '__main__':
    main()
