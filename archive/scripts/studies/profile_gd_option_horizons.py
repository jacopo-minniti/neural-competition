"""Check whether the residual-option comparison depends on horizon or panel size."""

import json

import torch
from explore_conditional_editability import OUT as SOURCE
from explore_conditional_editability import ROOT, Net, contrasts
from explore_gd_option_dynamics import OUT, geometry


def main():
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(8192, generator=torch.Generator().manual_seed(281710))[:128]
    u = contrasts()
    rows = []
    for width in (32, 128):
        for checkpoint in (0, 600):
            model = Net(width).double()
            model.load_state_dict(torch.load(SOURCE/f'model_281501_{width}_{checkpoint}.pt', weights_only=True))
            for n in (64, 128):
                x, y = data['xtrain'][ids[:n]].double(), data['train_labels'][ids[:n]]
                _, residual, raw, weighted, _, root = geometry(model, x, y, u)
                eigen, vectors = torch.linalg.eigh(weighted)
                eigen = eigen.clamp_min(0)
                g = raw/n
                v = g @ root @ vectors
                coefficients = vectors.T @ residual
                for tau in (.005, .025, .1, 1.):
                    s = 1/(2*tau)
                    benefit = (g-(v/(eigen+s))@v.T)/s
                    mass = benefit.trace()
                    rows.append(dict(width=width, checkpoint=checkpoint, examples=n, tau=tau,
                        mass=float(mass), breadth=float(mass**2/benefit.square().sum()),
                        quadratic_discounted_gain=float(.5*((eigen/(eigen+s))*coefficients.square()).sum())))
    (OUT/'horizon_profile.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
