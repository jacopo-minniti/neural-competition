"""Follow-up: does a larger protection buffer reduce optimistic option value?"""

import json

import numpy as np
import torch
from explore_conditional_editability import (
    OUT,
    ROOT,
    Net,
    contrasts,
    kernel,
    shift,
    solve,
    transform,
)


def main():
    data = torch.load(ROOT/'data/mnist/prepared.pt', weights_only=True)
    ids = torch.randperm(5000, generator=torch.Generator().manual_seed(281500))[:144]
    u = contrasts()
    original = data['xval'][ids].double()
    fill = float(data['xval'].min())
    rows = []
    base = json.loads((OUT/'results.json').read_text())['measurements']
    for width in (32, 128):
        model = Net(width).double()
        model.load_state_dict(torch.load(OUT/f'model_281501_{width}_600.pt', weights_only=True))
        ridge = .1*next(r['kernel_scale'] for r in base if r['width']==width and
                       r['seed']==281501 and r['step']==600)
        for anchors in (16, 64):
            n = 16+anchors
            x = torch.cat([original[:n], shift(original[:n], fill=fill)])
            fresh = original[80:]
            fresh_x = torch.cat([fresh, shift(fresh, fill=fill)])
            t = transform(n, focal=16)
            gram = kernel(model, torch.cat([x, fresh_x]), u)
            k = t @ gram[:len(x)*9, :len(x)*9] @ t.T
            r, _, dual = solve(k, 144, ridge)
            cross = gram[len(x)*9:, :len(x)*9]
            cross = (cross[:len(fresh)*9]+cross[len(fresh)*9:])/(2*np.sqrt(len(fresh))) @ t.T
            fit = (k @ dual)[144:n*9]
            validation = cross @ dual
            excess = float(validation.square().sum()-fit.square().sum())
            rows.append(dict(width=width, anchors=anchors, options=float(r.trace()),
                validated_background_value=float(r.trace())-excess, excess_collateral=excess))
            print(json.dumps(rows[-1]), flush=True)
    (OUT/'buffer_audit.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
