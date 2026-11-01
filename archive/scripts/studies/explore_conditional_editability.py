"""Teacher-free conditional editability on ordinary MNIST training.

All output contrasts and random examples are probed, with no selected classes.
The exact shallow-network kernel is used to audit a matrix-free-capable object.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/evidence/conditional_editability'
torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w = nn.Parameter(torch.randn(width, 784) / np.sqrt(784))
        self.b = nn.Parameter(torch.zeros(width))
        self.a = nn.Parameter(torch.randn(10, width))
        self.c = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return F.linear(F.relu(F.linear(x, self.w, self.b)), self.a) / np.sqrt(
            self.w.shape[0]) + self.c


def contrasts():
    u = torch.zeros(10, 9, dtype=torch.float64)
    for k in range(1, 10):
        u[:k, k-1] = 1 / np.sqrt(k * (k+1))
        u[k, k-1] = -k / np.sqrt(k * (k+1))
    return u


def kernel(model, x, u):
    """Full trainable-parameter Jacobian Gram, in centered logit coordinates."""
    with torch.no_grad():
        w, b, a = model.w.double(), model.b.double(), model.a.double()
        z = x @ w.T + b
        h, gate = z.relu(), (z > 0).double()
        ac = u.T @ a
        width = w.shape[0]
        hidden = torch.einsum('ir,jr,cr,dr->icjd', gate, gate, ac, ac)
        hidden *= (x @ x.T + 1)[:, None, :, None] / width
        output = (h @ h.T / width + 1)[:, None, :, None] * torch.eye(9)[None, :, None, :]
        return (hidden + output).reshape(len(x)*9, len(x)*9)


def transform(n, coherent=True, focal=None):
    """Focal mean, background mean, and within-instance view disagreement."""
    half = n // 2 if focal is None else focal
    t = torch.zeros(2*n, 2*n, dtype=torch.float64)
    if coherent:
        t[:n, :n] = torch.eye(n) / 2
        t[:n, n:] = torch.eye(n) / 2
        t[n:, :n] = torch.eye(n) / 2
        t[n:, n:] = -torch.eye(n) / 2
        t[n:n+half] /= np.sqrt(2*half)
        t[n+half:] /= np.sqrt(2*(n-half))
    else:
        t[:n, :n] = torch.eye(n)
    t[:half] /= np.sqrt(half)
    t[half:n] /= np.sqrt(n-half)
    return torch.kron(t, torch.eye(9, dtype=torch.float64))


def solve(k, q, ridge):
    eye = torch.eye(len(k), dtype=k.dtype)
    inv = torch.linalg.solve(k + ridge * eye, eye[:, :q])
    r = (k @ inv)[:q]
    r = (r + r.T) / 2
    eig = torch.linalg.eigvalsh(r).clamp(0, 1)
    return r, eig, inv


def shift(x, dy=0, dx=1, fill=0.):
    images = x.reshape(-1, 28, 28)
    moved = torch.full_like(images, fill)
    ys, yt = slice(max(0, -dy), min(28, 28-dy)), slice(max(0, dy), min(28, 28+dy))
    xs, xt = slice(max(0, -dx), min(28, 28-dx)), slice(max(0, dx), min(28, 28+dx))
    moved[:, yt, xt] = images[:, ys, xs]
    return moved.flatten(1)


def intervention(model, x, u, t, inv, r, y, ridge, background):
    """Audit nonlinear realization of the ridge optimum, including an unseen view."""
    q = r.shape[0]
    third_view = shift(x[:len(y)], dx=-1, fill=float(x.min()))
    with torch.no_grad():
        f = model(x) @ u
        p = model(x[:len(y)]).softmax(-1)
        demand = ((F.one_hot(y, 10).double()-p) @ u)[:len(y)//2].flatten()
        demand /= demand.norm()
    # Also audit a derivative-selected weakest/strongest direction, not hand roles.
    _, vectors = torch.linalg.eigh(r)
    rows = []
    for name, target in [('residual', demand), ('weak', vectors[:, 0]), ('strong', vectors[:, -1])]:
        alpha = inv @ target
        coefficients = t.T @ alpha
        model.zero_grad()
        scalar = ((model(x) @ u).flatten() * coefficients).sum()
        direction = torch.autograd.grad(scalar, tuple(model.parameters()))
        predicted = t @ (kernel(model, x, u) @ coefficients)
        for amplitude in (.01, .1):
            edited = copy.deepcopy(model)
            with torch.no_grad():
                for parameter, delta in zip(edited.parameters(), direction):
                    parameter.add_(amplitude * delta)
                change = t @ ((edited(x) @ u - f).flatten()) / amplitude
                fit = .5 * (change[:q] - target).square().sum()
                collateral = .5 * change[q:].square().sum()
                movement = .5 * ridge * sum(d.square().sum() for d in direction)
                expected = .5 * (target @ target - target @ r @ target)
                heldout = (edited(background) - model(background)) @ u
                third = ((edited(third_view)-model(third_view)) @ u).flatten() / amplitude
                means = change[:len(y)*9] * np.sqrt(len(y)//2)
                rows.append(dict(direction=name, amplitude=amplitude,
                    predicted_objective=float(expected), actual_objective=float(fit+collateral+movement),
                    response_relative_error=float((change-predicted).norm()/predicted.norm().clamp_min(1e-12)),
                    unseen_background_rms=float(heldout.square().mean().sqrt()/amplitude),
                    fitted_background_rms=float(change[q:len(y)*9].square().mean().sqrt()*np.sqrt(len(y)//2)),
                    third_view_relative_disagreement=float((third-means).norm()/means.norm().clamp_min(1e-12))))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=[281501, 281502], widths=[32, 128], steps=[0, 100, 600],
        train_examples=8192, batch_size=128, learning_rate=.05, probe_examples=32,
        ridge_multipliers=[.01, .1, 1.], views='original and one-pixel right shift',
        note='Recorded before training. Ordinary cross-entropy SGD, no teacher or designated task roles.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    train = data['xtrain'][:8192]
    labels = data['train_labels'][:8192]
    test = data['xval'][:1024]
    targets = data['val_labels'][:1024]
    generator = torch.Generator().manual_seed(281500)
    ids = torch.randperm(5000, generator=generator)[:96]
    original = data['xval'][ids[:32]].double()
    probe_labels = data['val_labels'][ids[:32]]
    unseen = data['xval'][ids[32:]].double()
    fill = float(data['xval'].min())
    views = torch.cat([original, shift(original, fill=fill)])
    u, t = contrasts(), transform(32)
    q = 16*9
    rows, tests, arrays = [], [], {}
    for seed in protocol['seeds']:
        for width in protocol['widths']:
            torch.manual_seed(seed)
            model = Net(width)
            optimizer = torch.optim.SGD(model.parameters(), lr=.05)
            batches = torch.Generator().manual_seed(seed+1)
            initial_scale = None
            for step in range(601):
                if step in protocol['steps']:
                    probe = copy.deepcopy(model).double()
                    k0 = kernel(probe, views, u)
                    k = t @ k0 @ t.T
                    scale = float(k.trace()/len(k))
                    if initial_scale is None:
                        initial_scale = scale
                    with torch.no_grad():
                        accuracy = float((model(test).argmax(-1)==targets).float().mean())
                        loss = float(F.cross_entropy(model(test), targets))
                    row = dict(seed=seed, width=width, step=step, accuracy=accuracy,
                               cross_entropy=loss, kernel_scale=scale)
                    for multiplier in protocol['ridge_multipliers']:
                        r, eig, inv = solve(k, q, multiplier*scale)
                        row[f'options_shape_{multiplier}'] = float(eig.sum())
                        fixed_r, fixed, _ = solve(k, q, multiplier*initial_scale)
                        row[f'options_fixed_{multiplier}'] = float(fixed.sum())
                        if multiplier == .1:
                            arrays[f'{seed}_{width}_{step}_spectrum'] = eig.numpy()
                            arrays[f'{seed}_{width}_{step}_R'] = r.numpy()
                            arrays[f'{seed}_{width}_{step}_R_fixed'] = fixed_r.numpy()
                            if step in (0, 600) and seed == protocol['seeds'][0]:
                                for result in intervention(probe, views, u, t, inv, r, probe_labels,
                                                           multiplier*scale, unseen):
                                    tests.append(dict(seed=seed, width=width, step=step, **result))
                    plain = transform(32, coherent=False)
                    kp = plain @ k0 @ plain.T
                    _, eigen, _ = solve(kp, q, .1*scale)
                    row['options_without_view_constraint'] = float(eigen.sum())
                    # Unconditional editability, same effort scale; excludes protection cost.
                    _, eigen, _ = solve(k[:q, :q], q, .1*scale)
                    row['options_without_background'] = float(eigen.sum())
                    with torch.no_grad():
                        features = F.relu(F.linear(original, probe.w, probe.b))
                        singular = torch.linalg.svdvals(features)
                        distribution = singular/singular.sum()
                        row['feature_effective_rank'] = float(torch.exp(-(distribution*distribution.clamp_min(1e-30).log()).sum()))
                    rows.append(row)
                    torch.save(probe.state_dict(), OUT/f'model_{seed}_{width}_{step}.pt')
                    print(json.dumps(row), flush=True)
                    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol, measurements=rows,
                        interventions=tests), indent=2)+'\n')
                    np.savez_compressed(OUT/'operators.npz', **arrays)
                if step < 600:
                    batch = torch.randint(len(train), (128,), generator=batches)
                    optimizer.zero_grad()
                    F.cross_entropy(model(train[batch]), labels[batch]).backward()
                    optimizer.step()
    print('Completed', flush=True)


if __name__ == '__main__':
    main()
