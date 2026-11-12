#!/usr/bin/env python3
"""Apply card 32's selection and validation to BOTTOM's read-only FP8 capture; never relax its coverage gate."""
import json
from pathlib import Path
import sys

import numpy as np

import encode
from encode import ce


def prepare(pilot, out):
    capture = pilot / 'eq/runs/fp8-calib-rows'
    checkpoint = pilot / 'checkpoint'
    ce.NVFP4 = pilot / 'models/drama-3-NVFP4-Pruned-5900'
    provenance = ce.NVFP4 / 'nvfp4-provenance.json'
    kept = ce.kept_experts(ce.NVFP4 / 'model.safetensors.index.json')
    if kept != {(layer, expert) for layer, experts in encode.kept_experts().items() for expert in experts}:
        raise ValueError('BOTTOM checkpoint does not contain the pinned 5,900 experts')
    target = json.loads(provenance.read_text())['stats_pairs']
    manifest = json.loads((capture / 'rows.json').read_text())
    count = manifest['rows']
    classes = np.fromfile(capture / 'class.u8', np.uint8)
    picks = {layer: ce.layer_rows(capture, layer, count, np)[1] for layer in range(ce.LAYERS)}
    selected = ce.select(classes, picks, kept, target, np)
    identity = np.array_equal(selected, np.arange(count))
    del picks, selected
    out.mkdir(parents=True, exist_ok=True)
    rows = capture if identity else out / 'rows'
    if not identity:
        ce.write_subset(capture, rows, kept, target)
    report = ce.validate(rows, checkpoint, kept, ce.nvfp4_routed(provenance), target)
    report.update(producer='BOTTOM FP8 capture + card 32 selection and validation',
                  capture=str(capture), selection_identity=identity, selection_seed=0)
    report['failures'] = ce.failures(report)
    report['inputs_sha256'] = {str(path.relative_to(pilot)): encode.ex.sha256(path) for path in (
        capture / 'rows.json', ce.NVFP4 / 'model.safetensors.index.json', provenance)}
    (out / 'rows-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['failures']:
        raise ValueError('; '.join(report['failures']))
    return rows


if __name__ == '__main__':
    print(prepare(*map(Path, sys.argv[1:])))
