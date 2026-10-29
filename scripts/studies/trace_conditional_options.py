"""Dynamic conditional-option accounting and function-preserving growth controls."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from explore_stable_distinctions import task
from explore_transported_gates import initialize, losses, train

ROOT = Path(__file__).resolve().parents[2] / 'docs/theory'


def save_trace(arrays):
    """Lossless standard NPZ archive; LZMA keeps the full-precision trace compact."""
    destination = ROOT/'conditional_options_trajectories.npz'
    temporary = destination.with_suffix('.npz.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_LZMA) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            ordered = np.asfortranarray(value) if value.ndim > 1 else value
            np.save(buffer, ordered, allow_pickle=False)
            archive.writestr(name+'.npy', buffer.getvalue())
    temporary.replace(destination)


def basis(x, p):
    r = (x[:, 0] > 0).astype(float)
    q = p @ r
    z = x[:, 1] / np.sqrt(p @ x[:, 1]**2)
    context = (r-q) / np.sqrt(q*(1-q))
    phi = np.stack([np.ones(len(p)), context, z, context*z], axis=1)
    assert np.allclose(phi.T @ (p[:, None]*phi), np.eye(4))
    return phi


def probe(state, x, p, y, phi, gate_override=None):
    w, a, c = state
    m = a.shape[1]
    pre = w @ x.T
    gates = pre > 0 if gate_override is None else gate_override
    act = gates*pre
    e = c[:, None] + (a[..., None]*act).mean(1) - y
    coefficients = e @ (p[:, None]*phi)
    outside = e - coefficients @ phi.T
    ja = np.einsum('smi,ik->smk', act, p[:, None]*phi)
    jw = a[..., None, None]*np.einsum('smi,ik,ij->smkj', gates, p[:, None]*phi, x)
    jac = np.concatenate([ja[..., None], jw], axis=-1)
    gram = np.einsum('smik,smjk->sij', jac, jac)/m
    gram[:, 0, 0] += 1
    beta = np.einsum('sij,sj->si', np.linalg.pinv(gram[:, :3, :3], rcond=1e-12), gram[:, :3, 3])
    adjusted = jac[:, :, 3] - np.einsum('si,smij->smj', beta, jac[:, :, :3])
    contributions = np.sum(adjusted**2, axis=-1)/m
    contributions = np.concatenate([contributions, beta[:, :1]**2], axis=1)
    kappa = contributions.sum(1)
    neff = np.divide(kappa**2, np.sum(contributions**2, axis=1),
                     out=np.zeros_like(kappa), where=kappa > 1e-24)

    def velocity(residual):
        da = -(act*(p*residual)[:, None]).sum(2)
        dw = -a[..., None]*(((gates)*(p*residual)[:, None]) @ x)
        dc = -(residual*p).sum(1)
        df = dc[:, None] + (da[..., None]*act + a[..., None]*gates*(dw@x.T)).mean(1)
        return df, (dw, da, dc)

    df, _ = velocity(e)
    outside_df, _ = velocity(outside)
    derivative = df @ (p[:, None]*phi)
    outside_derivative = outside_df @ (p[:, None]*phi)
    main_force = np.sum(beta*derivative[:, :3], axis=1)
    outside_force = outside_derivative[:, 3] - np.sum(beta*outside_derivative[:, :3], axis=1)
    direct = -kappa*coefficients[:, 3]
    error = derivative[:, 3] - direct - main_force - outside_force
    assert np.max(abs(error)) < 1e-9
    signature = np.sum(gates.astype(np.uint64)*(np.uint64(1) << np.arange(len(p), dtype=np.uint64)), axis=2)
    same_gate = signature[:, :, None] == signature[:, None, :]
    family_square = np.einsum('sij,si,sj->s', same_gate, contributions[:, :-1], contributions[:, :-1])
    family_square += contributions[:, -1]**2
    family_neff = np.divide(kappa**2, family_square, out=np.zeros_like(kappa), where=kappa > 1e-24)

    # All non-main residuals, so the energy law needs no omitted-mode forcing.
    conditional = e - coefficients[:, :3] @ phi[:, :3].T
    energy2 = np.sum(p*conditional**2, axis=1)
    jr_a = np.einsum('smi,si->sm', act, p*conditional)
    jr_w = a[..., None]*np.einsum('smi,si,ij->smj', gates, p*conditional, x)
    jr = np.concatenate([jr_a[..., None], jr_w], axis=-1)
    cross = np.einsum('smik,smk->si', jac[:, :, :3], jr)/m
    beta_r = np.einsum('sij,sj->si', np.linalg.pinv(gram[:, :3, :3], rcond=1e-12), cross)
    editable = jr - np.einsum('si,smij->smj', beta_r, jac[:, :, :3])
    power = np.sum(editable**2, axis=(1, 2))/m + beta_r[:, 0]**2
    rate = np.divide(power, energy2, out=np.zeros_like(power), where=energy2 > 1e-24)
    transfer = np.sum(beta_r*derivative[:, :3], axis=1)
    energy_derivative = np.sum(p*conditional*df, axis=1)
    assert np.max(abs(energy_derivative + power - transfer)) < 1e-9
    return np.stack([.5*np.sum(p*e**2, axis=1), coefficients[:, 3], kappa, neff,
                     direct, main_force, outside_force, derivative[:, 3],
                     .5*np.sum(p*outside**2, axis=1), np.max(contributions, axis=1),
                     family_neff, energy2/2, rate, transfer, energy_derivative], axis=1)


def trace(state, x, p, y, horizon=2000, dt=.05):
    phi = basis(x, p)
    history, times = [], []
    snapshots = {}
    old_gates = None
    old_kappa = None
    cumulative = np.zeros((len(state[0]), 3))
    previous = None
    for step in range(round(horizon/dt)+1):
        observed = probe(state, x, p, y, phi)
        if previous is not None:
            cumulative[:, 0] += dt*2*previous[:, 11]*previous[:, 12]
            cumulative[:, 1] += dt*previous[:, 13]
            cumulative[:, 2] += observed[:, 11]-previous[:, 11]-dt*previous[:, 14]
        previous = observed
        # Resolve initial alignment every optimizer step; later every 0.5 time units.
        if step <= 400 or step % 10 == 0:
            gates = state[0] @ x.T > 0
            if old_gates is None:
                changes = np.zeros((len(state[0]), 2))
            else:
                virtual = probe(state, x, p, y, phi, gate_override=old_gates)[:, 2]
                changes = np.stack([virtual-old_kappa, observed[:, 2]-virtual], axis=1)
            history.append(np.concatenate([observed, changes, cumulative.copy()], axis=1))
            old_gates, old_kappa = gates, observed[:, 2]
            times.append(step*dt)
        if step in (400, 2000, round(horizon/dt)):
            snapshots[step*dt] = tuple(v.copy() for v in state)
        if step < round(horizon/dt):
            state = train(state, x, p, y, dt, dt=dt)
    return np.array(times), np.stack(history), snapshots


def grow(state, kind):
    w, a, c = state
    m = a.shape[1]
    if kind == 'clone':
        return np.repeat(w, 2, axis=1), np.repeat(a, 2, axis=1), c.copy()
    # Positive homogeneity preserves both the old function and old tangent kernel.
    if kind == 'dormant':
        new_w = np.tile([0., 0., -1.], (len(w), m, 1))
    else:
        sign = 1 if kind == 'rare' else -1
        patterns = np.array([[sign, 1., 0.], [sign, -1., 0.]])/np.sqrt(2)
        new_w = np.broadcast_to(patterns[np.arange(m) % 2], (len(w), m, 3)).copy()
    return np.concatenate([np.sqrt(2)*w, new_w], axis=1), np.concatenate(
        [np.sqrt(2)*a, np.zeros_like(a)], axis=1), c.copy()


def stationary_projection(state, x, p, y):
    """Construct the exact cellwise least-squares projection; check gate feasibility."""
    w, a, c = state
    pre = w @ x.T
    gates = pre > 0
    design = (gates[..., None]*x[None, None]).transpose(0, 2, 1, 3)
    design = np.concatenate([np.ones((len(w), len(x), 1)), design.reshape(len(w), len(x), -1)], axis=2)
    coef = np.concatenate([c[:, None], (a[..., None]*w/a.shape[1]).reshape(len(w), -1)], axis=1)
    residual = y - np.einsum('sni,si->sn', design, coef)
    delta = np.linalg.pinv(design*np.sqrt(p)[None, :, None], rcond=1e-12) @ (residual*np.sqrt(p))[:, :, None]
    updated = coef + delta[..., 0]
    new_w = updated[:, 1:].reshape(w.shape)*a.shape[1]/a[..., None]
    feasible = np.all((new_w @ x.T > 0) == gates, axis=(1, 2))
    pred = np.einsum('sni,si->sn', design, updated)
    normal_error = np.max(abs(np.einsum('sni,sn->si', design, p*(pred-y))), axis=1)
    margin = np.min(abs(new_w@x.T), axis=(1, 2))
    return dict(feasible=feasible.tolist(), margin=margin.tolist(),
                normal_error=normal_error.tolist(), loss=(.5*np.sum(p*(pred-y)**2, axis=1)).tolist())


def main():
    raw, rows, interventions = {}, [], []
    protocol = dict(seed=997031, seeds=48, widths=[4, 8, 16], q=.1, dt=.05,
                    trace_horizon=2000, extension_horizon=10000, growth_time=100,
                    probe_columns=['loss', 'b', 'kappa', 'effective_blocks', 'direct',
                                   'main_force', 'outside_force', 'db', 'outside_loss', 'max_block',
                                   'effective_gate_families', 'conditional_energy', 'conditional_rate',
                                   'conditional_transfer', 'conditional_energy_derivative',
                                   'kappa_parameter_change', 'kappa_gate_change',
                                   'cumulative_direct_reduction', 'cumulative_main_transfer',
                                   'cumulative_finite_step_correction'])
    (ROOT/'conditional_options_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name, nodes, spread in [('discrete', 1, 0.), ('clusters', 3, .15)]:
        x, p, y = task(nodes, spread, .1)
        initial = initialize(997031, 48, 16)
        for m in (4, 8, 16):
            state = tuple(v[:, :m].copy() if v.ndim > 1 else v.copy() for v in initial)
            times, history, snapshots = trace(state, x, p, y)
            key = f'{name}_m{m}'
            raw[f'{key}_time'], raw[f'{key}_history'] = times, history
            for t, snapshot in snapshots.items():
                for label, value in zip(('w', 'a', 'c'), snapshot):
                    raw[f'{key}_t{int(t)}_{label}'] = value
            extension = train(snapshots[2000.], x, p, y, 8000)
            loss_long = losses(extension, x, p, y)
            raw[f'{key}_loss10000'] = loss_long
            row = dict(case=name, width=m, loss2000=float(history[-1, :, 0].mean()),
                       loss10000=float(loss_long.mean()),
                       stationary_projection=stationary_projection(extension, x, p, y))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'stationary_projection'}), flush=True)
            if m == 4:
                checkpoint = snapshots[100.]
                for kind in ('clone', 'dormant', 'common', 'rare'):
                    grown = grow(checkpoint, kind)
                    assert np.allclose(losses(grown, x, p, y), losses(checkpoint, x, p, y))
                    before = probe(grown, x, p, y, basis(x, p))
                    end = train(grown, x, p, y, 1900)
                    final = losses(end, x, p, y)
                    raw[f'{name}_growth_{kind}_probe'] = before
                    raw[f'{name}_growth_{kind}_loss'] = final
                    inter = dict(case=name, kind=kind, mean_loss=float(final.mean()),
                                 mean_kappa=float(before[:, 2].mean()))
                    interventions.append(inter)
                    print(json.dumps(inter), flush=True)
            np.savez_compressed(ROOT/'conditional_options_trajectories.npz', **raw)
            (ROOT/'conditional_options_results.json').write_text(json.dumps(
                dict(protocol=protocol, results=rows, interventions=interventions), indent=2)+'\n')
    save_trace(raw)


if __name__ == '__main__':
    main()
