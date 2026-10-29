"""Test derived small-output transport and forecast finite-horizon spectrum.

Forecast files precede the held-out continuations. The full kernel is used;
no fast-main-effect elimination is assumed in this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from explore_stable_distinctions import linear_span, task
from explore_transported_gates import initialize, losses, train
from explore_transported_gates import task as transport_task

ROOT = Path(__file__).resolve().parents[2] / 'docs/theory'


def spectral_forecast(state, x, p, y, horizon):
    w, a, c = state
    pre = w @ x.T
    act = np.maximum(pre, 0)
    gates = pre > 0
    residual = c[:, None] + (a[..., None]*act).mean(1) - y
    kernel = np.einsum('smi,smj->sij', act, act)
    kernel += np.einsum('sm,smi,smj->sij', a*a, gates, gates) * (x@x.T)
    kernel = 1 + kernel / w.shape[1]
    weighted = kernel * np.sqrt(p[:, None]*p[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(weighted)
    coordinates = np.einsum('sji,sj->si', eigenvectors, residual*np.sqrt(p))
    return .5*np.sum(coordinates**2*np.exp(-2*horizon*np.maximum(eigenvalues, 0)), axis=1)


def alignment():
    x, p, y, _, _ = transport_task('attribute', 5)
    rows = []
    for epsilon in (.02, .01, .005):
        initial = initialize(98122, 64, 16)
        initial = (initial[0]*epsilon/.2, initial[1]*epsilon/.2, initial[2])
        actual = train(initial, x, p, y, .5, dt=.001)
        w, a, _ = (v.copy() for v in initial)
        for _ in range(500):
            force = ((w@x.T > 0)*(p*y)[None, None, :]) @ x
            da = np.sum(w*force, axis=-1)
            dw = a[..., None]*force
            w += .001*dw
            a += .001*da
        relative = np.sqrt(np.sum((w-actual[0])**2)+np.sum((a-actual[1])**2))
        relative /= np.sqrt(np.sum(actual[0]**2)+np.sum(actual[1]**2))
        rows.append(dict(epsilon=epsilon, relative_parameter_error=float(relative),
            gate_disagreement=float(np.mean((w@x.T>0)!=(actual[0]@x.T>0)))))
    (ROOT/'distinction_alignment_results.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


def forecast():
    x, p, y = task(3, .15, .1)
    initial = initialize(997031, 48, 16)
    raw, rows = {}, []
    for m in (4, 8, 16):
        state = tuple(v[:, :m].copy() if v.ndim > 1 else v.copy() for v in initial)
        state = train(state, x, p, y, 100)
        pred = spectral_forecast(state, x, p, y, 1900)
        floor = linear_span(state[0], x, p, y)
        for name, value in zip(('w', 'a', 'c'), state):
            raw[f'm{m}_{name}'] = value
        raw[f'm{m}_spectral'] = pred
        raw[f'm{m}_floor'] = floor
        rows.append(dict(width=m, spectral=float(pred.mean()), floor=float(floor.mean())))
    np.savez_compressed(ROOT/'distinction_spectral_checkpoints.npz', **raw)
    result = dict(protocol=dict(seed=997031, seeds=48, checkpoint=100, endpoint=2000,
        dt=.05, status='Written before these held-out continuations.'), forecasts=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'distinction_spectral_forecasts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def validate():
    result = json.loads((ROOT/'distinction_spectral_forecasts.json').read_text())
    assert result['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = np.load(ROOT/'distinction_spectral_checkpoints.npz')
    x, p, y = task(3, .15, .1)
    rows, endpoints = [], {}
    for m in (4, 8, 16):
        state = tuple(raw[f'm{m}_{name}'] for name in ('w', 'a', 'c'))
        end = train(state, x, p, y, 1900)
        actual = losses(end, x, p, y)
        dense = losses(end, *task(9, .15, .1))
        row = dict(width=m, mean_loss=float(actual.mean()), dense_loss=float(dense.mean()),
            se=float(actual.std(ddof=1)/np.sqrt(len(actual))))
        for name in ('spectral', 'floor'):
            pred = raw[f'm{m}_{name}']
            row[name] = dict(mean=float(pred.mean()), mae=float(abs(pred-actual).mean()))
        endpoints[f'm{m}_loss'] = actual
        rows.append(row)
        print(json.dumps(row), flush=True)
    result['validation'] = rows
    (ROOT/'distinction_spectral_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(ROOT/'distinction_spectral_endpoints.npz', **endpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['alignment', 'forecast', 'validate'])
    args = parser.parse_args()
    {'alignment': alignment, 'forecast': forecast, 'validate': validate}[args.phase]()
