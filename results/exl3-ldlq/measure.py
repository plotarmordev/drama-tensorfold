#!/usr/bin/env python3
"""EXL3 with exllamav3's LDLQ rounding against NVFP4 and int3, on card 35's six experts.

  measure.py probe ENCODER INT3_DIR LAYER/EXPERT OUT_DIR   # one expert into OUT_DIR/LAYER-EXPERT.json
  measure.py report OUT_DIR OUT.json                       # card 35's six experts and the verdict
  measure.py hessians OUT_DIR                              # the six experts' Hessians for exllamav3_job.py
  measure.py parity PROBES.json EXLLAMAV3.json OUT.json    # ours against exllamav3's own encodes

ENCODER is tf-exl3-encode built with ldlq.patch; INT3_DIR holds the int3 build's layer-LL.safetensors (int3.sha256 in
../smaller-builds). Each expert's error is trace((W - Q) H (W - Q)^T) over gate, up and down, W the FP8 original and H
its Hessians from the NVFP4-mix calibration rows (recipe/exl3_experts.py's recipe), for EXL3 at 3.0 and 2.5 bits, the
published NVFP4 and int3. EXL3 beats int3 at a bitrate when it is smaller and its summed error is lower. Ours matches
exllamav3 when every projection's error is within PARITY of exllamav3's on the same W and H.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe'))
import calibrate_experts as ce  # noqa: E402
import exl3_experts as ex  # noqa: E402

PROBES = ((0, 237), (0, 16), (0, 117), (2, 42), (2, 222), (2, 5))  # card 35: fewest, median, most rows in layers 0 and 2
PARITY = 1.1


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


def routed_rows(layer, expert):
    rows = json.loads((ce.OUTPUT / 'rows.json').read_text())['rows']
    stem, picks, _ = ce.layer_rows(ce.OUTPUT, layer, rows, np)
    routed = (picks == expert).any(1)
    return ce.bf16(f'{stem}.x.bf16', rows, np)[routed]


def probe(encoder, int3_dir, layer, expert, out_dir):
    x = routed_rows(layer, expert)
    result = measure_expert(encoder, ce.Tensors(ce.CHECKPOINT, np), ce.Tensors(ce.NVFP4, np),
                            Int3(int3_dir, [layer]), layer, expert, x)
    record = {'encoder_sha256': ex.sha256(encoder), 'rows': len(x), **result}
    (out_dir / f'{layer}-{expert}.json').write_text(json.dumps(record, indent=1) + '\n')


def report(out_dir, out):
    probes = {f'{l}/{e}': json.loads((out_dir / f'{l}-{e}.json').read_text()) for l, e in PROBES}
    out.write_text(json.dumps({'rows': str(ce.OUTPUT), 'probes': probes, 'verdict': verdict(probes)}, indent=1) + '\n')


def weight_sha256(w):
    return hashlib.sha256(np.ascontiguousarray(w, np.float32).tobytes()).hexdigest()


def hessians(out_dir):
    """Per expert, OUT_DIR/LAYER-EXPERT.npz: the upper triangles of its gate/up and down Hessians in float32 and the
    sha256 of each projection's float32 W, which exllamav3_job.py rebuilds from the FP8 checkpoint and checks."""
    source = ce.Tensors(ce.CHECKPOINT, np)
    for layer, expert in PROBES:
        w = {p: ce.original(source, ex.name(layer, expert, p), np) for p in ex.PROJECTIONS}
        with np.errstate(all='ignore'):
            h_in, h_down = ex.hessians(routed_rows(layer, expert), w['gate_proj'], w['up_proj'])
        np.savez_compressed(out_dir / f'{layer}-{expert}.npz', h_in=h_in[np.triu_indices(len(h_in))].astype(np.float32),
                            h_down=h_down[np.triu_indices(len(h_down))].astype(np.float32),
                            **{f'{p}_sha256': weight_sha256(w[p]) for p in ex.PROJECTIONS})


def parity(probes, exllamav3):
    """Per bitrate and probe, each projection's error over exllamav3's, whether all of them are within PARITY, and
    exllamav3's own verdict against int3 and NVFP4 (its EXL3 takes the same bytes as ours at each bitrate)."""
    ratios = {bits: {k: {p: probes[k][f'exl3-{bits}'][p] / ref[bits][p] for p in ex.PROJECTIONS}
                     for k, ref in exllamav3.items()} for bits in ex.BITS}
    worst = max(r for per_bits in ratios.values() for per_probe in per_bits.values() for r in per_probe.values())
    theirs = {k: {**probes[k], **{f'exl3-{bits}': ref[bits] for bits in ex.BITS}} for k, ref in exllamav3.items()}
    return {'ours_over_exllamav3': ratios, 'worst': worst, 'matches': worst <= PARITY,
            'exllamav3_verdict': verdict(theirs)}


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == 'probe':
        layer, expert = map(int, args[2].split('/'))
        probe(Path(args[0]).resolve(), Path(args[1]), layer, expert, Path(args[3]))
    elif command == 'report':
        report(Path(args[0]), Path(args[1]))
    elif command == 'hessians':
        hessians(Path(args[0]))
    else:
        probes = json.loads(Path(args[0]).read_text())['probes']
        result = parity(probes, json.loads(Path(args[1]).read_text()))
        Path(args[2]).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
