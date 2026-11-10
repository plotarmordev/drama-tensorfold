"""trim_head.py on a small synthetic checkpoint: no model, GPU or network."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import trim_head

HIDDEN = 4
CONFIG = {'eos_token_id': 9, 'semantic_start_token_id': 3, 'semantic_end_token_id': 5}


def write_shard(path, tensors):
    """`tensors`: (name, dtype, shape, bytes) in data order, the head between two others as in the real shard."""
    header, data = {'__metadata__': {'format': 'pt'}}, b''
    for name, dtype, shape, raw in tensors:
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(data), len(data) + len(raw)]}
        data += raw
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(text)) + text + data)


def read_shard(path):
    raw = path.read_bytes()
    (size,) = struct.unpack('<Q', raw[:8])
    header = json.loads(raw[8:8 + size])
    meta = header.pop('__metadata__')
    data = raw[8 + size:]
    return meta, {k: (v['dtype'], v['shape'], data[slice(*v['data_offsets'])]) for k, v in header.items()}


def row(r):
    return bytes([r]) * (HIDDEN * 2)


class TrimHeadTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.src = self.root / 'src'
        self.src.mkdir()
        self.head = b''.join(row(r) for r in range(12))
        self.before = ('model.language_model.norm.weight', 'BF16', [HIDDEN], b'\x01' * 8)
        self.after = ('audio_decoder.x', 'F32', [3], b'\x02' * 12)
        write_shard(self.src / 'b.safetensors', [self.before, ('lm_head.weight', 'BF16', [12, HIDDEN], self.head), self.after])
        write_shard(self.src / 'a.safetensors', [('model.language_model.embed_tokens.weight', 'BF16', [2, HIDDEN], b'\x03' * 16)])
        weight_map = {'model.language_model.embed_tokens.weight': 'a.safetensors', 'lm_head.weight': 'b.safetensors',
                      self.before[0]: 'b.safetensors', self.after[0]: 'b.safetensors'}
        self.metadata = {'total_size': 16 + 8 + len(self.head) + 12, 'drama_experts': {'0': [1, 2]}}
        (self.src / trim_head.INDEX).write_text(json.dumps({'metadata': self.metadata, 'weight_map': weight_map}))
        (self.src / 'config.json').write_text(json.dumps(CONFIG))
        (self.src / 'codec.pth').write_bytes(b'codec')

    def test_head_keeps_eos_then_the_semantic_band_and_nothing_else_changes(self):
        dst = self.root / 'dst'
        trim_head.trim(self.src, dst)
        meta, tensors = read_shard(dst / 'b.safetensors')
        self.assertEqual(meta, {'format': 'pt'})
        self.assertEqual(tensors['lm_head.weight'], ('BF16', [4, HIDDEN], row(9) + row(3) + row(4) + row(5)))
        self.assertEqual(tensors[self.before[0]], self.before[1:])
        self.assertEqual(tensors[self.after[0]], self.after[1:])
        for name in ('a.safetensors', 'config.json', 'codec.pth'):
            self.assertEqual((dst / name).read_bytes(), (self.src / name).read_bytes())
        index = json.loads((dst / trim_head.INDEX).read_text())
        self.assertEqual(index['metadata'], {**self.metadata, 'drama_head_rows': [9, 3, 4, 5],
                                             'total_size': self.metadata['total_size'] - 8 * HIDDEN * 2})

    def test_refuses_an_existing_destination_and_a_trimmed_source(self):
        dst = self.root / 'dst'
        dst.mkdir()
        with self.assertRaises(FileExistsError):
            trim_head.trim(self.src, dst)
        trim_head.trim(self.src, self.root / 'once')
        with self.assertRaisesRegex(SystemExit, 'already trimmed'):
            trim_head.trim(self.root / 'once', self.root / 'twice')


if __name__ == '__main__':
    unittest.main()
