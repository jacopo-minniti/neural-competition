#!/usr/bin/env python3
"""Check allocation provenance, convex certificates, budgets, and cohort scope."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'docs/tables/allocation'


def main():
    checked = set()
    def verify(mapping):
        for name, expected in mapping.items():
            if (name, expected) in checked:
                continue
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
            checked.add((name, expected))
    for label in ['teacher', 'teacher_fullcal', 'teacher_matched']:
        for item in json.loads((OUTPUT/f'{label}_provenance.json').read_text()):
            verify(item['source_sha256'])
            verify(item.get('input_sha256', {}))
    item = json.loads((OUTPUT/'mnist_provenance.json').read_text())
    verify(item['source_sha256'])
    verify(item['checkpoint_sha256'])
    verify({'data/mnist/prepared.pt': item['data_sha256']})
    item = json.loads((OUTPUT/'readout_provenance.json').read_text())
    verify(item['input_sha256'])
    verify({'scripts/studies/check_allocation_features.py': item['source_sha256']})
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    verify(manifest['raw_sha256'])
    verify({'scripts/studies/analyze_allocation.py': manifest['analyzer_sha256']})
    assert manifest['teacher']['trajectories'] == 245
    assert manifest['teacher']['conditions'] == 15
    assert manifest['teacher_matched']['intended_targets'] == 490
    assert manifest['mnist']['checkpoints'] == 119
    error = 0.
    rows = 0
    for name in ['teacher', 'teacher_fullcal', 'teacher_matched', 'mnist']:
        frame = pd.read_csv(OUTPUT/f'{name}_episodes.csv.gz')
        assert (frame.oracle_allocation_gain >= -1e-12).all()
        assert frame.qp_relative_gap.max() < 1e-7
        assert frame.group_relative_gap.max() < 1e-7
        budget = frame.positive_work-frame.negative_work-frame.curvature-frame.progress
        error = max(error,float(np.max(np.abs(budget))))
        assert np.max(np.abs(budget)) < 1e-12
        assert frame.corrective_cosine.between(-1e-12,1+1e-12).all()
        rows += len(frame)
    coverage = pd.read_csv(OUTPUT/'teacher_matched_coverage.csv')
    assert len(coverage) == 490
    result = dict(verified_hashes=len(checked), episode_rows=rows, max_work_budget_error=error,
                  matched_targets_found=int(coverage.found.sum()), intended_targets=len(coverage),
                  producer_sources_match=True)
    (OUTPUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
