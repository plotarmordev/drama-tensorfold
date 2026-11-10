#!/usr/bin/env python3
"""Write a Drama 3 checkpoint whose lm_head holds only the rows the engine samples.

Usage: trim_head.py SRC DST. DST must not exist. The frame step scores only EOS and the semantic band
(config.json's semantic_start..semantic_end), so lm_head keeps those rows, in that order, and the index's metadata
lists their vocabulary ids under drama_head_rows; the loaders of both engines read the list. The shard holding
lm_head is rewritten with every other tensor's bytes unchanged; every other file is copied as it is.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct

HEAD = 'lm_head.weight'
HEAD_ROWS = 'drama_head_rows'
INDEX = 'model.safetensors.index.json'
CHUNK = 64 << 20


def head_rows(config):
    start, end = config['semantic_start_token_id'], config['semantic_end_token_id']
    return [config['eos_token_id'], *range(start, end + 1)]


def read_header(f):
    (size,) = struct.unpack('<Q', f.read(8))
    return json.loads(f.read(size)), 8 + size


def copy_range(src, dst, begin, end):
    src.seek(begin)
    while begin < end:
        chunk = src.read(min(CHUNK, end - begin))
        if not chunk:
            raise SystemExit(f'{src.name}: ends before byte {end}')
        dst.write(chunk)
        begin += len(chunk)


def write_trimmed_shard(src_path, dst_path, rows):
    """Rewrite one shard with lm_head cut to `rows`, keeping tensor order, other tensors' bytes and __metadata__.

    Returns the tensor bytes dropped."""
    with open(src_path, 'rb') as src:
        header, data = read_header(src)
        head = header[HEAD]
        vocab, hidden = head['shape']
        if head['dtype'] != 'BF16' or max(rows) >= vocab:
            raise SystemExit(f'{src_path}: {HEAD} is {head["dtype"]} {head["shape"]}, not a BF16 head over the rows')
        row = hidden * 2
        tensors = sorted((k for k in header if k != '__metadata__'), key=lambda k: header[k]['data_offsets'][0])
        out, at, pieces = {}, 0, []
        if '__metadata__' in header:
            out['__metadata__'] = header['__metadata__']
        for name in tensors:
            t = dict(header[name])
            begin, end = t['data_offsets']
            if name == HEAD:
                t['shape'] = [len(rows), hidden]
                spans = [(data + begin + r * row, data + begin + (r + 1) * row) for r in rows]
            else:
                spans = [(data + begin, data + end)]
            size = sum(e - b for b, e in spans)
            t['data_offsets'] = [at, at + size]
            out[name] = t
            pieces += spans
            at += size
        text = json.dumps(out, separators=(',', ':')).encode()
        text += b' ' * (-len(text) % 8)
        with open(dst_path, 'xb') as dst:
            dst.write(struct.pack('<Q', len(text)) + text)
            for begin, end in pieces:
                copy_range(src, dst, begin, end)
    return (vocab - len(rows)) * row


def trim(src, dst):
    index = json.loads((src / INDEX).read_text())
    if HEAD_ROWS in index.get('metadata', {}):
        raise SystemExit(f'{src}: {HEAD} is already trimmed')
    rows = head_rows(json.loads((src / 'config.json').read_text()))
    shard = index['weight_map'][HEAD]
    dst.mkdir(parents=True)
    for path in sorted(src.iterdir()):
        if path.name == shard:
            dropped = write_trimmed_shard(path, dst / shard, rows)
        elif path.name != INDEX and path.is_file():
            shutil.copy2(path, dst / path.name)
    index['metadata']['total_size'] -= dropped
    index['metadata'][HEAD_ROWS] = rows
    (dst / INDEX).write_text(json.dumps(index, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    args = parser.parse_args()
    trim(args.src, args.dst)


if __name__ == '__main__':
    main()
