#!/usr/bin/env python3
"""EXL3 with exllamav3's LDLQ rounding against NVFP4 and int3, on card 35's six experts.

  measure.py probe ENCODER INT3_DIR LAYER/EXPERT OUT_DIR   # one expert into OUT_DIR/LAYER-EXPERT.json
  measure.py report OUT_DIR OUT.json                       # card 35's six experts and the verdict

ENCODER is tf-exl3-encode built with ldlq.patch; INT3_DIR holds the int3 build's layer-LL.safetensors (int3.sha256 in
../smaller-builds). Each expert's error is trace((W - Q) H (W - Q)^T) over gate, up and down, W the FP8 original and H
its Hessians from the NVFP4-mix calibration rows (recipe/exl3_experts.py's recipe), for EXL3 at 3.0 and 2.5 bits, the
published NVFP4 and int3. EXL3 beats int3 at a bitrate when it is smaller and its summed error is lower.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe'))
import calibrate_experts as ce  # noqa: E402
import exl3_experts as ex  # noqa: E402

PROBES = ((0, 237), (0, 16), (0, 117), (2, 42), (2, 222), (2, 5))  # card 35: fewest, median, most rows in layers 0 and 2


class Int3(ce.Tensors):
    """The int3 job's per-layer files, which have no index."""

    def __init__(self, root, layers):
        self.root, self.np, self.headers, self.files, self.sizes = root, np, {}, {}, {}
        for layer in layers:
            file = f'layer-{layer:02d}.safetensors'
            with (root / file).open('rb') as f:
                header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
            header.pop('__metadata__', None)
            self.files.update(dict.fromkeys(header, file))
            self.sizes.update({k: v['data_offsets'][1] - v['data_offsets'][0] for k, v in header.items()})


def int3(tensors, name):
    """Per 16 weights, u16 planes sign, hi, lo (bit i is weight i): ±2·(2 hi + lo) times its E4M3 scale and the global."""
    packed = tensors(name + '.weight')
    planes = packed.reshape(packed.shape[0], -1, 3, 2).astype(np.uint16)
    planes = planes[..., 0] | planes[..., 1] << 8
    bits = (planes[..., None] >> np.arange(16, dtype=np.uint16)) & 1
    sign, hi, lo = bits[:, :, 0], bits[:, :, 1], bits[:, :, 2]
    values = (2.0 * (2 * hi + lo) * (1 - 2.0 * sign)).reshape(packed.shape[0], -1)
    return values * np.repeat(tensors(name + '.weight_scale'), 16, 1) * tensors(name + '.weight_scale_2')


def measure_expert(encoder, source, nvfp4, int3s, layer, expert, x):
    """Errors per projection and bytes of one expert: EXL3 at each bitrate, NVFP4 and int3, on the same Hessians."""
    w = {p: ce.original(source, ex.name(layer, expert, p), np) for p in ex.PROJECTIONS}
    with np.errstate(all='ignore'):
        h_in, h_down = ex.hessians(x, w['gate_proj'], w['up_proj'])
        h = {p: h_down if p == 'down_proj' else h_in for p in ex.PROJECTIONS}
        out = {'nvfp4': {p: ex.hessian_error(w[p], ce.nvfp4(nvfp4, ex.name(layer, expert, p), np), h[p])
                         for p in ex.PROJECTIONS},
               'int3': {p: ex.hessian_error(w[p], int3(int3s, ex.name(layer, expert, p)), h[p]) for p in ex.PROJECTIONS}}
    stored = ('.weight', '.weight_scale', '.weight_scale_2')
    out['bytes'] = {'nvfp4': ex.nvfp4_bytes(ce.NVFP4, layer, [expert]),
                    'int3': sum(int3s.sizes[ex.name(layer, expert, p) + s] for p in ex.PROJECTIONS for s in stored)}
    for bits in ex.BITS:
        out[f'exl3-{bits}'], size = {}, 0
        for p in ex.PROJECTIONS:
            out[f'exl3-{bits}'][p], tensors = ex.encode(encoder, w[p], h[p], bits)
            size += sum(len(raw) for _, _, raw in tensors.values())
        out['bytes'][f'exl3-{bits}'] = size
    return out


def verdict(probes):
    """Per EXL3 bitrate: bytes and summed error over int3's, and whether it beats int3 (smaller and lower error)."""
    total = {q: sum(sum(r[q].values()) for r in probes.values()) for q in ('int3', 'nvfp4', *(f'exl3-{b}' for b in ex.BITS))}
    size = {q: sum(r['bytes'][q] for r in probes.values()) for q in total}
    out = {'error_sum': total, 'bytes': size}
    for bits in ex.BITS:
        q = f'exl3-{bits}'
        out[q] = {'bytes_over_int3': size[q] / size['int3'], 'error_over_int3': total[q] / total['int3'],
                  'error_over_nvfp4': total[q] / total['nvfp4'],
                  'beats_int3': size[q] < size['int3'] and total[q] < total['int3']}
    return out


def probe(encoder, int3_dir, layer, expert, out_dir):
    rows = json.loads((ce.OUTPUT / 'rows.json').read_text())['rows']
    stem, picks, _ = ce.layer_rows(ce.OUTPUT, layer, rows, np)
    routed = (picks == expert).any(1)
    x = ce.bf16(f'{stem}.x.bf16', rows, np)[routed]
    result = measure_expert(encoder, ce.Tensors(ce.CHECKPOINT, np), ce.Tensors(ce.NVFP4, np),
                            Int3(int3_dir, [layer]), layer, expert, x)
    record = {'encoder_sha256': ex.sha256(encoder), 'rows': int(routed.sum()), **result}
    (out_dir / f'{layer}-{expert}.json').write_text(json.dumps(record, indent=1) + '\n')


def report(out_dir, out):
    probes = {f'{l}/{e}': json.loads((out_dir / f'{l}-{e}.json').read_text()) for l, e in PROBES}
    out.write_text(json.dumps({'rows': str(ce.OUTPUT), 'probes': probes, 'verdict': verdict(probes)}, indent=1) + '\n')


def main():
    if sys.argv[1] == 'probe':
        layer, expert = map(int, sys.argv[4].split('/'))
        probe(Path(sys.argv[2]).resolve(), Path(sys.argv[3]), layer, expert, Path(sys.argv[5]))
    else:
        report(Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == '__main__':
    main()
