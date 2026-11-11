#!/usr/bin/env python3
"""A pruned NVFP4 checkpoint with its kept routed experts replaced by drama_nvfp4_gptq.py --mode int3's.

  splice_int3.py SRC INT3_DIR DST

SRC is a t5900-cover checkpoint (trimmed head and BF16 codec allowed); INT3_DIR holds layer-LL.safetensors with all
256 experts of each layer. Every expert tensor of SRC (weight, weight_scale, weight_scale_2) is taken from its int3
file; everything else is copied byte for byte. Shard names and index keys are unchanged.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

EXPERT = 'model.language_model.layers.{}.mlp.experts.'


def read_header(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def tensor_bytes(path, header, start, name):
    begin, end = header[name]['data_offsets']
    with path.open('rb') as f:
        f.seek(start + begin)
        return f.read(end - begin)


def write_shard(path, tensors, metadata):
    """tensors: name -> (dtype, shape, bytes), written in order."""
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
        offset += len(data)
    if metadata:
        header['__metadata__'] = metadata
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw)
        for _, _, data in tensors.values():
            f.write(data)
    return offset


def splice(src, int3_dir, dst):
    dst.mkdir(parents=True, exist_ok=False)
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    int3 = {}  # layer -> (path, header, data start); a layer's experts can span two shards
    total = 0
    for shard in sorted(set(index['weight_map'].values())):
        header, start = read_header(src / shard)
        metadata = header.pop('__metadata__', None)
        tensors = {}
        for name in sorted(header, key=lambda n: header[n]['data_offsets'][0]):
            layer = name.split('.')[3] if name.startswith('model.language_model.layers.') else None
            if layer is not None and name.startswith(EXPERT.format(layer)):
                if layer not in int3:
                    path = int3_dir / f'layer-{int(layer):02d}.safetensors'
                    int3[layer] = (path, *read_header(path))
                source = int3[layer]
            else:
                source = (src / shard, header, start)
            meta = source[1][name]
            tensors[name] = (meta['dtype'], meta['shape'], tensor_bytes(*source, name))
        total += write_shard(dst / shard, tensors, metadata)
    index['metadata']['total_size'] = total
    (dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    for item in src.iterdir():
        if not (dst / item.name).exists() and item.name != 'MANIFEST.sha256':
            shutil.copy2(item, dst / item.name)


if __name__ == '__main__':
    splice(*map(Path, sys.argv[1:]))
