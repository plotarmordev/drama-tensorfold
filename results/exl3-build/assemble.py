#!/usr/bin/env python3
"""Splice completed EXL3 layers into trim-bf16 without changing dense weights, codec or pruning metadata.

  assemble.py SRC EXL3_DIR DST

Requires every kept layer's checksum record and recipe.json. DST must not exist; its index is written last.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import encode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'smaller-builds'))
from splice_int3 import read_header, tensor_bytes, write_shard

PREFIX = 'model.language_model.layers.'


def is_expert(name):
    return name.startswith(PREFIX) and '.mlp.experts.' in name


def validate_layer(path, layer, experts):
    header, start = read_header(path)
    expected = {}
    for expert in experts:
        for projection in encode.ex.PROJECTIONS:
            k, n = (512, 2048) if projection == 'down_proj' else (2048, 512)
            shapes = {'trellis': ('I16', [k // 16, n // 16, 48]), 'suh': ('F16', [k]),
                      'svh': ('F16', [n]), 'mul1': ('I32', [])}
            for role, (dtype, shape) in shapes.items():
                name = f'{encode.ex.name(layer, expert, projection)}.{role}'
                expected[name] = (dtype, shape)
    if set(header) - {'__metadata__'} != set(expected):
        raise ValueError(f'{path}: incomplete or unexpected expert tensors')
    if header.get('__metadata__') != {'format': 'exl3', 'bits': '3.0'}:
        raise ValueError(f'{path}: expected EXL3 3.0')
    offset = 0
    for name, meta in sorted(((n, header[n]) for n in expected), key=lambda pair: pair[1]['data_offsets'][0]):
        dtype, shape = expected[name]
        size = (4 if dtype == 'I32' else 2)
        for dimension in shape:
            size *= dimension
        if (meta['dtype'], meta['shape'], meta['data_offsets']) != (dtype, shape, [offset, offset + size]):
            raise ValueError(f'{path}: invalid tensor {name}')
        if name.endswith('.mul1') and tensor_bytes(path, header, start, name) != struct.pack('<I', 0x83DCD12D):
            raise ValueError(f'{path}: not the mul1 codebook')
        offset += size
    if path.stat().st_size != start + offset:
        raise ValueError(f'{path}: truncated or trailing tensor data')
    return expected, offset


def assemble(src, layers, dst):
    kept = encode.kept_experts()
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    wanted = {encode.ex.name(layer, expert, p) for layer, experts in kept.items()
              for expert in experts for p in encode.ex.PROJECTIONS}
    actual = {name.removesuffix('.weight') for name in index['weight_map']
              if is_expert(name) and name.endswith('.weight')}
    if actual != wanted:
        raise ValueError('source checkpoint expert set differs from kept.json')
    recipe = json.loads((layers / 'recipe.json').read_text())
    if recipe['kept'] != encode.ex.sha256(encode.KEPT):
        raise ValueError('encoded recipe expert set differs from kept.json')
    validated = {}
    for layer, experts in kept.items():
        if not encode.finished_layer(layers, layer, experts):
            raise ValueError(f'layer {layer}: unfinished')
        validated[layer] = validate_layer(layers / f'layer-{layer:02d}.safetensors', layer, experts)
    # All source and encoded admission checks happen before creating the destination.
    dst.mkdir(parents=True, exist_ok=False)
    weight_map, total = {}, 0
    for shard in sorted(set(index['weight_map'].values())):
        header, start = read_header(src / shard)
        tensors = {name: (meta['dtype'], meta['shape'], tensor_bytes(src / shard, header, start, name))
                   for name, meta in header.items() if name != '__metadata__' and not is_expert(name)}
        if tensors:
            total += write_shard(dst / shard, tensors, header.get('__metadata__'))
            weight_map.update({name: shard for name in tensors})
    for layer, (names, size) in validated.items():
        shard = f'exl3-layer-{layer:02d}.safetensors'
        if (dst / shard).exists():
            raise ValueError(f'output shard collision: {shard}')
        shutil.copyfile(layers / f'layer-{layer:02d}.safetensors', dst / shard)
        weight_map.update({name: shard for name in names})
        total += size
    for item in src.iterdir():
        if item.is_file() and item.suffix != '.safetensors' and item.name not in (
                'model.safetensors.index.json', 'MANIFEST.sha256', 'SHA256SUMS'):
            shutil.copy2(item, dst / item.name)
    index['weight_map'] = weight_map
    index.setdefault('metadata', {})['total_size'] = total
    (dst / 'exl3-provenance.json').write_text(json.dumps({'recipe': recipe, 'bits': 3.0,
        'experts': sum(map(len, kept.values())), 'shards': {f'exl3-layer-{l:02d}.safetensors':
        json.loads((layers / f'layer-{l:02d}.json').read_text())['sha256'] for l in kept}}, indent=1) + '\n')
    (dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=1) + '\n')
    size = sum(p.stat().st_size for p in dst.iterdir() if p.suffix == '.safetensors' or p.name == 'codec.pth')
    print(f'{dst}: {sum(map(len, kept.values()))} experts, {size / 2**30:.4f} GiB weights + codec')


if __name__ == '__main__':
    assemble(*map(Path, sys.argv[1:]))
