"""Prospective width forecasts from common-target particle transport.

Forecast phase writes its results before held-out finite networks are trained.
No final losses determine the checkpoint, task decomposition, or gate masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def task(kind, points, q=.1, amplitude=2., detail=2.):
    if points == 2:
        z, pz = np.array([-1., 1.]), np.ones(2)/2
    else:
        z, pz = np.polynomial.legendre.leggauss(points)
        pz = pz/2
    context = np.repeat([-1., 1.], points)
    attribute = np.tile(z, 2)
    rare = (context+1)/2
    x = np.stack([context, attribute, np.ones(2*points)], axis=1)
    mass = np.concatenate([(1-q)*pz, q*pz])
    main = {'context': context, 'attribute': attribute,
            'hinge': np.maximum(attribute, 0)}[kind]
    y = amplitude*main + detail*rare*attribute
    y_common = amplitude*main + detail*q*attribute
    damage = .5*detail**2*q*(1-q)*np.sum(pz*z*z)
    return x, mass, y, y_common, damage


def initialize(seed, seeds, width):
    rng = np.random.default_rng(seed)
    return (.2*rng.normal(size=(seeds, width, 3)),
            .2*rng.normal(size=(seeds, width)), np.zeros(seeds))


def train(state, x, mass, y, horizon, dt=.05):
    w, a, c = (v.copy() for v in state)
    for _ in range(round(horizon/dt)):
        pre = w @ x.T
        act = np.maximum(pre, 0)
        residual = c[:, None]+(a[..., None]*act).mean(axis=1)-y
        da = -(act*(mass*residual)[:, None]).sum(axis=2)
        dw = -a[..., None]*(((pre>0)*(mass*residual)[:, None]) @ x)
        c -= dt*(mass*residual).sum(axis=1)
        a += dt*da
        w += dt*dw
    return w, a, c


def losses(state, x, mass, y):
    w, a, c = state
    prediction = c[:, None]+(a[..., None]*np.maximum(w@x.T, 0)).mean(axis=1)
    return .5*np.sum(mass*(prediction-y)**2, axis=1)


def gate_floor(w, x, mass, y):
    """Exact projection onto the fixed-gate linear span, up to rank roundoff.

    All readouts are nonzero almost surely. Their magnitudes do not affect span.
    This is a bound if gates remain fixed, not a general future loss guarantee.
    """
    gates = (w@x.T)>0
    design = (gates[..., None]*x[None, None]).transpose(0, 2, 1, 3)
    design = design.reshape(len(w), len(x), -1)
    design = np.concatenate([np.ones((len(w), len(x), 1)), design], axis=2)
    design *= np.sqrt(mass)[None, :, None]
    gram = design@design.transpose(0, 2, 1)
    eig, vec = np.linalg.eigh(gram)
    coordinates = np.einsum('sji,j->si', vec, np.sqrt(mass)*y)
    tolerance = 1e-10*np.maximum(eig[:, -1:], 1)
    floor = .5*np.sum(coordinates**2*(eig<tolerance), axis=1)
    return floor


def forecasts(output):
    rows = []
    rng = np.random.default_rng(734201)
    for kind, points in [('context', 2), ('attribute', 2), ('attribute', 5), ('hinge', 5)]:
        x, mass, y, common, damage = task(kind, points)
        bank = initialize(19031, 1, 4096)
        transported = train(bank, x, mass, common, 20)
        for width in (4, 8, 16):
            ids = rng.integers(4096, size=(4096, width))
            before = gate_floor(bank[0][0, ids], x, mass, y)
            after = gate_floor(transported[0][0, ids], x, mass, y)
            rows.append(dict(kind=kind, points=points, width=width, damage=damage,
                forecast_initial_floor=float(before.mean()),
                forecast_transported_floor=float(after.mean()),
                transported_se=float(after.std(ddof=1)/np.sqrt(len(after))),
                common_bank_loss=float(losses(transported, x, mass, common)[0])))
    result = dict(protocol=dict(bank_seed=19031, bank_width=4096, bank_horizon=20,
        forecast_draws=4096, forecast_seed=734201, heldout_seed=882071,
        heldout_seeds=128, horizon=1000, dt=.05, q=.1, amplitude=2, detail=2,
        status='Forecasts written before held-out endpoints.'), forecasts=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def validate(output):
    result = json.loads(output.read_text())
    assert result['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, raw = [], {}
    for kind, points in [('context', 2), ('attribute', 2), ('attribute', 5), ('hinge', 5)]:
        x, mass, y, common, _ = task(kind, points)
        initial = initialize(882071, 128, 16)
        for width in (4, 8, 16):
            state = tuple(v[:, :width].copy() if v.ndim>1 else v.copy() for v in initial)
            actual_early = train(state, x, mass, y, 20)
            common_early = train(state, x, mass, common, 20)
            end = train(actual_early, x, mass, y, 980)
            loss = losses(end, x, mass, y)
            f0 = gate_floor(state[0], x, mass, y)
            fc = gate_floor(common_early[0], x, mass, y)
            fe = gate_floor(actual_early[0], x, mass, y)
            ff = gate_floor(end[0], x, mass, y)
            row = dict(kind=kind, points=points, width=width,
                actual_mean_loss=float(loss.mean()), actual_se=float(loss.std(ddof=1)/np.sqrt(len(loss))),
                initial_floor=float(f0.mean()), paired_common_floor=float(fc.mean()),
                early_full_floor=float(fe.mean()), final_gate_floor=float(ff.mean()),
                early_full_forecast_mae=float(abs(fe-loss).mean()),
                paired_common_forecast_mae=float(abs(fc-loss).mean()))
            rows.append(row)
            key=f'{kind}_{points}_m{width}'
            for name, val in [('loss',loss),('initial',f0),('common',fc),('early',fe),('final',ff)]:
                raw[f'{key}_{name}']=val
            print(json.dumps(row), flush=True)
    result['validation'] = rows
    dest = output.with_name('transported_gates_validation.json')
    dest.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(dest.with_suffix('.npz'), **raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['forecast', 'validate'])
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[2]/'docs/evidence/option_transport_resilience/transported_gates_forecasts.json'
    if args.phase == 'forecast':
        forecasts(output)
    else:
        validate(output)
