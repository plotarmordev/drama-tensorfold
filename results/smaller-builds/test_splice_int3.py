import json
import tempfile
import unittest
from pathlib import Path

from splice_int3 import read_header, splice, tensor_bytes, write_shard

EXPERT = 'model.language_model.layers.{}.mlp.experts.{}.gate_proj.{}'


def tensors_of(path):
    header, start = read_header(path)
    header.pop('__metadata__', None)
    return {name: (meta['dtype'], meta['shape'], tensor_bytes(path, header, start, name))
            for name, meta in header.items()}


class SpliceTests(unittest.TestCase):
    def test_kept_experts_take_int3_tensors_and_the_rest_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, int3, dst = Path(tmp, 'src'), Path(tmp, 'int3'), Path(tmp, 'dst')
            src.mkdir()
            int3.mkdir()
            norm = 'model.language_model.layers.0.input_layernorm.weight'
            # layer 0 keeps experts 1 and 3 across two shards; NVFP4 codes are k/2 = 8 bytes a row of 16 inputs
            shards = {'a.safetensors': {norm: ('BF16', [4], b'n' * 8),
                                        EXPERT.format(0, 1, 'weight'): ('U8', [1, 8], b'\x11' * 8)},
                      'b.safetensors': {EXPERT.format(0, 3, 'weight'): ('U8', [1, 8], b'\x33' * 8),
                                        EXPERT.format(0, 3, 'weight_scale_2'): ('F32', [], b'\0\0\x80?')}}
            for shard, tensors in shards.items():
                write_shard(src / shard, tensors, {'format': 'pt'})
            index = {'metadata': {'total_size': 0, 'drama_experts': {'0': [1, 3]}},
                     'weight_map': {name: shard for shard, tensors in shards.items() for name in tensors}}
            (src / 'model.safetensors.index.json').write_text(json.dumps(index))
            (src / 'config.json').write_text('{}')
            (src / 'MANIFEST.sha256').write_text('stale\n')
            # int3 codes are 3k/8 = 6 bytes a row; the file holds every expert, kept or not
            write_shard(int3 / 'layer-00.safetensors',
                        {EXPERT.format(0, e, p): (('U8', [1, 6], bytes([e]) * 6) if p == 'weight'
                                                  else ('F32', [], bytes([e, 0, 0, 64])))
                         for e in range(4) for p in ('weight', 'weight_scale_2')}, {'gptq': '{}'})

            splice(src, int3, dst)

            a, b = tensors_of(dst / 'a.safetensors'), tensors_of(dst / 'b.safetensors')
            self.assertEqual(a[norm], ('BF16', [4], b'n' * 8))
            self.assertEqual(a[EXPERT.format(0, 1, 'weight')], ('U8', [1, 6], b'\x01' * 6))
            self.assertEqual(b[EXPERT.format(0, 3, 'weight')], ('U8', [1, 6], b'\x03' * 6))
            self.assertEqual(b[EXPERT.format(0, 3, 'weight_scale_2')], ('F32', [], b'\x03\0\0@'))
            self.assertNotIn(EXPERT.format(0, 0, 'weight'), a | b)
            self.assertEqual(read_header(dst / 'a.safetensors')[0]['__metadata__'], {'format': 'pt'})
            out = json.loads((dst / 'model.safetensors.index.json').read_text())
            self.assertEqual(out['metadata']['total_size'], 8 + 6 + 6 + 4)
            self.assertEqual(out['weight_map'], index['weight_map'])
            self.assertTrue((dst / 'config.json').exists())
            self.assertFalse((dst / 'MANIFEST.sha256').exists())


if __name__ == '__main__':
    unittest.main()
