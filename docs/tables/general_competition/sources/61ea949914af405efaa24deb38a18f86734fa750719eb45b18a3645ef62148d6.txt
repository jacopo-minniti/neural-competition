"""Audit the exploratory input-law transfer probe and retain paired uncertainty."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_competition.training import components

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[2]
folder = root / 'docs/tables/general_competition'
frame = pd.DataFrame(json.loads((root / 'runs/general_competition/results.json').read_text()))
checks = []
for path in (root / 'runs/general_competition').glob('*.pt'):
    distribution, optimizer = path.stem.split('_')
    b = torch.load(path, weights_only=False)
    norm = torch.tensor([[np.sqrt(32 if a == 'ordinary32' else 8)]
                         for s, a in b['models']], dtype=torch.float64)
    pred = components(b['parameters'], b['x'][4096:][None].expand(30, -1, -1), norm).sum(-1)
    mse = (pred-b['y'][4096:]).square().mean(1).numpy()
    saved = frame.query('distribution==@distribution and optimizer==@optimizer').mse.to_numpy()
    np.testing.assert_allclose(mse, saved, atol=1e-12, rtol=1e-10)
    checks.append(dict(bundle=path.stem, max_error=float(np.max(abs(mse-saved)))))
frame.to_csv(folder / 'endpoints.csv', index=False)
rows = []
for (dist,opt), group in frame.groupby(['distribution','optimizer']):
    table = group.pivot(index='seed', columns='arm', values='mse')
    delta = 100*(table.ordinary8-table.protected8).to_numpy()
    ci = np.quantile(np.random.default_rng(22).choice(delta,(5000,len(delta))).mean(1),[.025,.975])
    rows.append(dict(distribution=dist,optimizer=opt,ordinary8=100*table.ordinary8.mean(),
                     ordinary32=100*table.ordinary32.mean(),protected8=100*table.protected8.mean(),
                     improvement_pp=delta.mean(),low_pp=ci[0],high_pp=ci[1],wins=int((delta>0).sum())))
summary=pd.DataFrame(rows)
summary.to_csv(folder/'summary.csv',index=False)
pd.DataFrame(checks).to_csv(folder/'audit.csv',index=False)
sources=[root/'scripts/studies/probe_general_competition.py',Path(__file__),
         root/'src/neural_competition/sign_survival.py',root/'src/neural_competition/training.py']
(folder/'sources').mkdir(exist_ok=True)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for p in sources:
    (folder/'sources'/(hashes[str(p.relative_to(root))]+'.txt')).write_bytes(p.read_bytes())
(folder/'provenance.json').write_text(json.dumps(hashes,indent=2))
print(summary.to_string(index=False))
