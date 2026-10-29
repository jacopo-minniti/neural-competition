"""Prospective option/residual tracking during paired, uninterrupted MNIST SGD.

New files only. Every forecast is appended before its future training interval.
"""

import copy
import json

import numpy as np
import torch
from explore_conditional_editability import ROOT, Net, contrasts, kernel
from torch.nn import functional as F

OUT = ROOT/'docs/theory/option_mismatch_tracking'
torch.set_num_threads(2)
SEEDS = list(range(281801, 281809))
RATE, INTERVAL, STEPS = .05, 20, 800
N = 64


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def evaluate(model, x, y, gradient=False):
    model.zero_grad()
    total = 0.
    for start in range(0, len(x), 1024):
        loss = F.cross_entropy(model(x[start:start+1024]), y[start:start+1024], reduction='sum')/len(x)
        total += float(loss.detach())
        if gradient:
            loss.backward()
    return total


def measure(model, x, y, full_grad):
    probe = copy.deepcopy(model).double()
    u = contrasts()
    with torch.no_grad():
        logits = probe(x)
        p = logits.softmax(-1)
        e = ((F.one_hot(y, 10)-p) @ u).flatten()
        h = torch.diag_embed(p)-p[:, :, None]*p[:, None, :]
        h = torch.einsum('ac,nab,bd->ncd', u, h, u)
        hv, hu = torch.linalg.eigh(h)
        root = torch.block_diag(*((hu*hv.clamp_min(0).sqrt()[:, None, :])@hu.transpose(1, 2)))
        raw = kernel(probe, x, u)
        g = raw/len(x)
        k = root@g@root
        kv, ku = torch.linalg.eigh(k)
        kv = kv.clamp_min(0)
        cross = g@root@ku
        tau = RATE*INTERVAL
        s = 1/(2*tau)
        benefit = (g-(cross/(kv+s))@cross.T)/s
        benefit = (benefit+benefit.T)/2
        bv, bu = torch.linalg.eigh(benefit)
        bv = bv.clamp_min(0)
        loading = (bu.T@e).square()
        norm = e.square().sum()
        mass = bv.sum()
        value = (bv*loading).sum()/(2*len(x))
        alignment = (bv*loading).sum()/(norm*mass/len(e)).clamp_min(1e-20)
        percentile = (loading*torch.linspace(0, 1, len(e))).sum()/norm.clamp_min(1e-20)
        gw, gb, ga, gc = [v.double() for v in full_grad]
        pre = F.linear(x, probe.w, probe.b)
        dg = (((x@gw.T+gb)*(pre>0))@probe.a.T+pre.relu()@ga.T)/np.sqrt(len(probe.w))+gc
        dg = (dg@u).flatten()
        norm_grad = sum(float(v.square().sum()) for v in full_grad)
        zg = ku.T@root@dg/np.sqrt(len(x))
        # Machine-precision restriction, not a theoretical viability cutoff.
        valid = kv > max(float(kv.max())*1e-12, 1e-14)
        projection = torch.zeros_like(kv)
        projection[valid] = zg[valid]/kv[valid].sqrt()
        powers = (1-RATE*kv)**(2*INTERVAL)
        psi = torch.full_like(kv, 2*RATE*INTERVAL)
        psi[valid] = (1-powers[valid])/kv[valid]
        spectral = RATE*INTERVAL*norm_grad+.5*((psi-2*RATE*INTERVAL)*projection.square()).sum()
        scalar_second = RATE*INTERVAL*norm_grad-.5*(RATE*INTERVAL)**2*zg.square().sum()
        # Frozen additive SGD noise estimate, from per-example loss gradients.
        responses = torch.einsum('aic,ic->ai', raw.reshape(len(e), len(x), 9), -e.reshape(-1, 9))
        modes = ku.T@root@responses/np.sqrt(len(x))
        modes[valid] /= kv[valid].sqrt()[:, None]
        modes[~valid] = 0
        noise_var = modes.var(1, unbiased=True)/128
        accumulated = sum((1-RATE*kv)**(2*j) for j in range(INTERVAL))
        noise_cost = .5*RATE**2*(kv*accumulated*noise_var).sum()
        row = dict(panel_loss=float(F.cross_entropy(logits, y)), residual_energy=float(norm),
            mass=float(mass), breadth=float(mass**2/bv.square().sum().clamp_min(1e-20)),
            value=float(value), alignment=float(alignment), residual_strength_percentile=float(percentile),
            gradient_norm_squared=norm_grad, first_order_prediction=RATE*INTERVAL*norm_grad,
            scalar_curvature_prediction=float(scalar_second), spectral_prediction=float(spectral),
            spectral_noise_prediction=float(spectral-noise_cost), estimated_noise_cost=float(noise_cost),
            stable=float(RATE*kv.max())<2,
            gradient_projection_fraction=float(projection.square().sum()/max(norm_grad, 1e-20)))
    return row, benefit, e, bv.numpy(), loading.numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'forecasts.jsonl').exists():
        raise RuntimeError('Preserving existing run; use its saved results rather than overwriting.')
    protocol = dict(seeds=SEEDS, calibration_seeds=SEEDS[:4], test_seeds=SEEDS[4:],
        widths=[32, 128], steps=STEPS, interval=INTERVAL, learning_rate=RATE, batch_size=128,
        training_examples=8192, probe_examples=N, validation_examples=1024,
        audit_seed=281800, paired_minibatches=True, primary_target='next-20-step full-training loss improvement: width128 minus width32',
        evaluation='seed-held-out fixed ridge baselines and uncalibrated dynamical predictions',
        note='Recorded before these training outcomes. Probe is not a separate training objective.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    train, labels = data['xtrain'][:8192], data['train_labels'][:8192]
    validation, vy = data['xval'][:1024], data['val_labels'][:1024]
    ids = torch.randperm(8192, generator=torch.Generator().manual_seed(281800))[:N]
    px, py = train[ids].double(), labels[ids]
    for seed in SEEDS:
        models = {}
        for width in (32, 128):
            torch.manual_seed(seed)
            models[width] = Net(width)
        optimizers = {m:torch.optim.SGD(net.parameters(), lr=RATE) for m, net in models.items()}
        generator = torch.Generator().manual_seed(seed+500)
        prior, previous_b, initial_b, arrays = None, {}, {}, {}
        for step in range(STEPS+1):
            if step % INTERVAL == 0:
                summaries, matrices, residuals = {}, {}, {}
                for width, model in models.items():
                    full_loss = evaluate(model, train, labels, gradient=True)
                    gradients = [p.grad.detach().clone() for p in model.parameters()]
                    val_loss = evaluate(model, validation, vy)
                    row, b, e, eigen, load = measure(model, px, py, gradients)
                    row.update(seed=seed, width=width, step=step, full_loss=full_loss, validation_loss=val_loss)
                    if width in previous_b:
                        change = torch.linalg.eigvalsh(b-previous_b[width])
                        row.update(lost=float(-change[change<0].sum()), gained=float(change[change>0].sum()))
                    else:
                        initial_b[width] = b.clone()
                    total_change = torch.linalg.eigvalsh(b-initial_b[width])
                    row['lost_from_initial'] = float(-total_change[total_change<0].sum())
                    row['gained_from_initial'] = float(total_change[total_change>0].sum())
                    summaries[width], matrices[width], residuals[width] = row, b, e
                    previous_b[width] = b
                    arrays[f'{width}_{step}_strength'] = eigen
                    arrays[f'{width}_{step}_residual_loading'] = load
                def value(b, e):
                    return float(e@b@e/(2*N))
                bn, bw = matrices[32], matrices[128]
                en, ew = residuals[32], residuals[128]
                common, excess = (en+ew)/2, en-ew
                cross = dict(common_value_narrow=value(bn, common), common_value_wide=value(bw, common),
                    excess_value_narrow=value(bn, excess), excess_value_wide=value(bw, excess),
                    capability_component=.5*(value(bw,en)+value(bw,ew)-value(bn,en)-value(bn,ew)),
                    demand_component=.5*(value(bw,ew)+value(bn,ew)-value(bw,en)-value(bn,en)))
                current = dict(seed=seed, step=step, narrow=summaries[32], wide=summaries[128], cross=cross)
                # Predictions and diagnostics are recorded before the upcoming SGD interval.
                append(OUT/'forecasts.jsonl', current)
                if prior is not None:
                    gains = {width:prior[name]['full_loss']-summaries[width]['full_loss']
                             for width, name in [(32,'narrow'), (128,'wide')]}
                    vgains = {width:prior[name]['validation_loss']-summaries[width]['validation_loss']
                              for width, name in [(32,'narrow'), (128,'wide')]}
                    append(OUT/'outcomes.jsonl', dict(seed=seed, start=step-INTERVAL, end=step,
                        narrow_gain=gains[32], wide_gain=gains[128], gain_difference=gains[128]-gains[32],
                        validation_gain_difference=vgains[128]-vgains[32]))
                prior = current
                np.savez_compressed(OUT/f'spectra_{seed}.npz', **arrays)
                if step % 100 == 0:
                    print(json.dumps(dict(seed=seed,step=step,loss32=summaries[32]['full_loss'],
                        loss128=summaries[128]['full_loss'])), flush=True)
            if step < STEPS:
                batch = torch.randint(len(train), (128,), generator=generator)
                for width, model in models.items():
                    optimizers[width].zero_grad()
                    F.cross_entropy(model(train[batch]), labels[batch]).backward()
                    optimizers[width].step()
        for width, model in models.items():
            torch.save(model.state_dict(), OUT/f'final_{seed}_{width}.pt')


if __name__ == '__main__':
    main()
