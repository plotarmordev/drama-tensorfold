#!/usr/bin/env python3
"""Rewrites a fork `drama3-voice/2` voice as the `drama3-http-voice/1` catalog file the pinned main engine loads.

  http_voice.py SRC.json OUT_DIR

The fork stores each reference's codes codebook-major (`vq_codes`, 10 x frames) with the sha256 of those int32s;
main reads frame rows of ten (`codes`) and attests only `synthetic`, `self` or `consented` sources, so the fork's
model-invented voices are `synthetic`. Every reference's codes are checked against their sha256 before the file,
OUT_DIR/<name of SRC>, is written; an existing file is never replaced.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path


def reference(fork):
    codebooks = fork['vq_codes']
    raw = b''.join(struct.pack(f'<{len(row)}i', *row) for row in codebooks)
    if len(codebooks) != 10 or hashlib.sha256(raw).hexdigest() != fork['codes_sha256']:
        raise ValueError(f'codes do not match codes_sha256 {fork["codes_sha256"]}')
    return {'text': fork['text'], 'codes': [list(frame) for frame in zip(*codebooks)]}


def convert(fork):
    if fork['format'] != 'drama3-voice/2' or fork['source']['kind'] != 'model-invented':
        raise ValueError(f'expected a model-invented drama3-voice/2 voice, got {fork["format"]} {fork["source"]}')
    return {'format': 'drama3-http-voice/1', 'source': {'kind': 'synthetic', 'note': fork['source']['note']},
            'references': [reference(ref) for ref in fork['references']]}


def main():
    source, out_dir = map(Path, sys.argv[1:3])
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = convert(json.loads(source.read_text()))
    with (out_dir / source.name).open('x') as out:
        out.write(json.dumps(doc, separators=(',', ':')) + '\n')
    print(f'{out_dir / source.name}: {len(doc["references"])} references')


if __name__ == '__main__':
    main()
