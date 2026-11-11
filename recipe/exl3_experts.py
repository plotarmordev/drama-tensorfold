#!/usr/bin/env python3
"""Quantize Drama 3's 5,900 kept routed experts to EXL3 on the Mac CPU, from the original FP8 weights.

Each kept expert's gate, up and down projections go through TensorFold's tf-exl3-encode at every bitrate, against
Hessians from the Mac calibration rows (recipe/calibrate_experts.py). The encoder reloads what it wrote and requires
dequantize() to reproduce its reconstruction bit for bit. Worker processes convert several layers at once, each into
its own work folder; a finished expert leaves its tensors and a record there, so an interrupted run resumes where it
stopped. Once every worker is done, each converted layer becomes one shard per bitrate, and each bitrate's directory
gets an index and exl3-provenance.json: the inputs, the settings, the sizes and, per layer, the Hessian error of EXL3
against the published NVFP4 experts on the same Hessians.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from itertools import repeat
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time

import numpy as np

import calibrate_experts as ce

BITS = ('3.0', '2.5')
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
OUTPUT = ce.ASSETS / 'drama3-exl3-experts'
RECEIPT = re.compile(r'hessian_error exl3=(\S+)')


def name(layer, expert, projection):
    return f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}'


def read_safetensors(path):
    """{name: (dtype, shape, bytes)} and the metadata of one safetensors file."""
    data = Path(path).read_bytes()
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + size])
    metadata = header.pop('__metadata__', {})
    return {k: (v['dtype'], v['shape'], data[8 + size + v['data_offsets'][0]:8 + size + v['data_offsets'][1]])
            for k, v in header.items()}, metadata


def write_safetensors(path, tensors, metadata):
    """Writes {name: (dtype, shape, bytes)} through a temporary file, so a reader never sees half a file."""
    header, offset = {'__metadata__': metadata}, 0
    for k, (dtype, shape, raw) in tensors.items():
        header[k] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    partial = Path(f'{path}.partial')
    with partial.open('wb') as f:
        f.write(struct.pack('<Q', len(encoded)) + encoded)
        for _, _, raw in tensors.values():
            f.write(raw)
    partial.rename(path)


def hessians(x, gate, up):
    """The gate/up and down Hessians of an expert from its rows x [n, 2048] and its original gate and up [512, 2048]."""
    x = x.astype(np.float64)
    g, u = x @ gate.T, x @ up.T
    a = g / (1 + np.exp(-g)) * u
    return x.T @ x / len(x), a.T @ a / len(a)


def hessian_error(w, q, h):
    d = w.astype(np.float64) - q
    return float(((d @ h) * d).sum())


def encode(encoder, weight, h, bits):
    """One [N, K] matrix through tf-exl3-encode: its EXL3 Hessian error and its tensors, named by role."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / 'w.npy', weight.astype(np.float64))
        np.save(tmp / 'h.npy', h)
        run = subprocess.run([str(encoder), str(tmp / 'w.npy'), str(tmp / 'h.npy'), bits, str(tmp / 'out')],
                             check=True, capture_output=True, text=True)
        tensors, _ = read_safetensors(tmp / 'out/model.safetensors')
    return float(RECEIPT.search(run.stdout)[1]), {k[len('weight.'):]: v for k, v in tensors.items()}


def convert_expert(encoder, source, quant, layer, expert, x, rows, work):
    """Encodes one expert, with x its calibration rows (ROWS of them routed), at every bitrate into WORK; the record,
    written last, marks it done."""
    weights = {p: ce.original(source, name(layer, expert, p), np) for p in PROJECTIONS}
    record = {'rows': rows, 'seconds': {}, 'nvfp4': {}, 'exl3': {}}
    # Accelerate's matmul raises FP flags on finite inputs and still returns the right values.
    with np.errstate(all='ignore'):
        h_in, h_down = hessians(x, weights['gate_proj'], weights['up_proj'])
        for p in PROJECTIONS:
            h = h_down if p == 'down_proj' else h_in
            record['nvfp4'][p] = hessian_error(weights[p], ce.nvfp4(quant, name(layer, expert, p), np), h)
    for bits in BITS:
        started, tensors, record['exl3'][bits] = time.monotonic(), {}, {}
        for p in PROJECTIONS:
            error, parts = encode(encoder, weights[p], h_down if p == 'down_proj' else h_in, bits)
            record['exl3'][bits][p] = error
            tensors.update({f'{name(layer, expert, p)}.{k}': v for k, v in parts.items()})
        record['seconds'][bits] = time.monotonic() - started
        write_safetensors(work / f'{expert:03d}-{bits}.safetensors', tensors, {'format': 'exl3'})
    # Through a temporary file, like the shards: a run killed mid-write must not leave a record that reads as done.
    partial = work / f'{expert:03d}.json.partial'
    partial.write_text(json.dumps(record))
    partial.rename(work / f'{expert:03d}.json')


def convert_layer(args, layer, experts, rows):
    """Converts the layer's unfinished experts into its own work folder; a worker process runs one layer at a time."""
    work = args.output / 'work' / f'layer-{layer:02d}'
    work.mkdir(parents=True, exist_ok=True)
    todo = [e for e in experts if not (work / f'{e:03d}.json').exists()]
    if todo:
        source, quant = ce.Tensors(args.checkpoint, np), ce.Tensors(args.nvfp4, np)
        stem, picks, _ = ce.layer_rows(args.rows, layer, rows, np)
        x = ce.bf16(f'{stem}.x.bf16', rows, np)
        for expert in todo:
            routed = (picks == expert).any(1)
            count = int(routed.sum())
            convert_expert(args.encoder, source, quant, layer, expert, x[routed] if count else x, count, work)
            print(f'layer {layer} expert {expert}: {count} rows', flush=True)


def write_shards(output, layer, experts):
    """The layer's shard at every bitrate from its finished experts, the last bitrate's last."""
    work = output / 'work' / f'layer-{layer:02d}'
    for bits in BITS:
        tensors = {}
        for expert in experts:
            tensors.update(read_safetensors(work / f'{expert:03d}-{bits}.safetensors')[0])
        write_safetensors(output / f'exl3-{bits}' / f'model-{layer:02d}.safetensors', tensors, {'format': 'exl3'})


def layer_report(records, bits, shard, nvfp4_bytes):
    """Sizes and per-projection Hessian errors of one layer's experts at one bitrate, beside NVFP4's."""
    report = {'experts': len(records), 'no_rows': sorted(e for e, r in records.items() if r['rows'] == 0),
              'bytes': shard.stat().st_size, 'nvfp4_bytes': nvfp4_bytes,
              'seconds': sum(r['seconds'][bits] for r in records.values())}
    for p in PROJECTIONS:
        exl3 = np.array([r['exl3'][bits][p] for r in records.values()])
        nvfp4 = np.array([r['nvfp4'][p] for r in records.values()])
        report[p] = {'exl3_error': float(exl3.sum()), 'nvfp4_error': float(nvfp4.sum()),
                     'exl3_over_nvfp4': float(exl3.sum() / nvfp4.sum()),
                     'median_expert_exl3_over_nvfp4': float(np.median(exl3 / nvfp4))}
    return report


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_sha256(rows):
    """One hash over every file of the rows directory but its report: the Hessians depend on all of their bytes."""
    digest = hashlib.sha256()
    for path in sorted(p for p in rows.iterdir() if p.name != 'report.json'):
        digest.update(path.name.encode())
        with path.open('rb') as f:
            while block := f.read(1 << 24):
                digest.update(block)
    return digest.hexdigest()


def nvfp4_bytes(nvfp4, layer, experts):
    """On-disk bytes of the published NVFP4 tensors of these experts."""
    index = json.loads((nvfp4 / 'model.safetensors.index.json').read_text())['weight_map']
    prefixes = tuple(f'{name(layer, e, p)}.' for e in experts for p in PROJECTIONS)
    total, headers = 0, {}
    for tensor, file in index.items():
        if tensor.startswith(prefixes):
            if file not in headers:
                with (nvfp4 / file).open('rb') as f:
                    headers[file] = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
            begin, end = headers[file][tensor]['data_offsets']
            total += end - begin
    return total


def write_checkpoint(out, bits, inputs, layers):
    """The index and exl3-provenance.json of one bitrate's directory, over every finished layer."""
    weight_map = {}
    for layer in layers:
        shard = f'model-{layer:02d}.safetensors'
        weight_map.update({k: shard for k in read_safetensors(out / shard)[0]})
    total = sum(r['bytes'] for r in layers.values())
    (out / 'model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total, 'format': 'exl3'}, 'weight_map': weight_map}, indent=1) + '\n')
    provenance = {
        'schema': 'drama-exl3-experts/1',
        'producer': 'recipe/exl3_experts.py + TensorFold tf-exl3-encode',
        'inputs': inputs,
        'settings': {'bits_per_weight': float(bits), 'codebook': 'mul1', 'hessian': ce.HESSIAN},
        'complete': len(layers) == ce.LAYERS,
        'experts': sum(r['experts'] for r in layers.values()),
        'bytes': total,
        'nvfp4_bytes': sum(r['nvfp4_bytes'] for r in layers.values()),
        'layers': {str(layer): r for layer, r in sorted(layers.items())},
    }
    (out / 'exl3-provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True, help='tf-exl3-encode from a TensorFold build')
    parser.add_argument('--checkpoint', type=Path, default=ce.CHECKPOINT, help='the original FP8 checkpoint')
    parser.add_argument('--nvfp4', type=Path, default=ce.NVFP4, help='the published 5,900-expert NVFP4 checkpoint')
    parser.add_argument('--rows', type=Path, default=ce.OUTPUT, help='calibration rows from calibrate_experts.py')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--layers', type=int, nargs='+', choices=range(ce.LAYERS), default=range(ce.LAYERS),
                        metavar='LAYER')
    parser.add_argument('--workers', type=int, default=8,
                        help='layers converted at once, one process each holding its layer\'s rows (~1.5 GB)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    manifest = json.loads((args.rows / 'rows.json').read_text())
    config = sha256(args.checkpoint / 'config.json')
    if manifest['identity']['config_sha256'] != config:
        raise SystemExit(f'{args.rows} was captured from another checkpoint than {args.checkpoint}')
    commit = subprocess.run(['git', '-C', str(args.encoder.parent), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    inputs = {
        'checkpoint': str(args.checkpoint), 'config_sha256': config,
        'nvfp4': str(args.nvfp4), 'nvfp4_index_sha256': sha256(args.nvfp4 / 'model.safetensors.index.json'),
        'rows': str(args.rows), 'rows_sha256': rows_sha256(args.rows), 'row_count': manifest['rows'],
        'encoder': str(args.encoder), 'encoder_sha256': sha256(args.encoder),
        'encoder_commit': commit.stdout.strip() or None,
    }
    # One recipe for every record, so a work folder encoded from other rows or with other Hessians is never resumed.
    recipe = {'hessian': ce.HESSIAN, 'rows_sha256': inputs['rows_sha256']}
    stamp = args.output / 'work' / 'hessian.json'
    if (args.output / 'work').exists() and (not stamp.exists() or json.loads(stamp.read_text()) != recipe):
        raise SystemExit(f'{args.output / "work"} holds experts encoded with other Hessians; use a new --output')
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(recipe))
    kept = ce.kept_experts(args.nvfp4 / 'model.safetensors.index.json')
    experts = {layer: sorted(e for l, e in kept if l == layer) for layer in range(ce.LAYERS)}
    for bits in BITS:
        (args.output / f'exl3-{bits}').mkdir(parents=True, exist_ok=True)
    layers = sorted(set(args.layers))
    # Each worker writes only its layer's work folder; a failed layer stops the run here, before any shard is written.
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(convert_layer, repeat(args), layers, [experts[l] for l in layers], repeat(manifest['rows'])))
    for layer in layers:
        write_shards(args.output, layer, experts[layer])
    # A run over some layers still reports every layer an earlier run finished.
    finished = [l for l in range(ce.LAYERS) if (args.output / f'exl3-{BITS[-1]}' / f'model-{l:02d}.safetensors').exists()]
    for bits in BITS:
        reports = {}
        for layer in finished:
            work = args.output / 'work' / f'layer-{layer:02d}'
            records = {e: json.loads((work / f'{e:03d}.json').read_text()) for e in experts[layer]}
            shard = args.output / f'exl3-{bits}' / f'model-{layer:02d}.safetensors'
            reports[layer] = layer_report(records, bits, shard, nvfp4_bytes(args.nvfp4, layer, experts[layer]))
            print(f'layer {layer} {bits} bits: ' + json.dumps({p: reports[layer][p]['exl3_over_nvfp4'] for p in PROJECTIONS}))
        write_checkpoint(args.output / f'exl3-{bits}', bits, inputs, reports)


if __name__ == '__main__':
    main()
