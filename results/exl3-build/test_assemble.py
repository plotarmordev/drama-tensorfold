import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import assemble
from assemble import encode, read_header, tensor_bytes, write_shard


class AssembleTest(unittest.TestCase):
    def fixture(self, root):
        src, layers, dst = (root / n for n in ('src', 'layers', 'dst'))
        src.mkdir()
        layers.mkdir()
        dense = 'model.language_model.layers.0.input_layernorm.weight'
        old = {dense: ('BF16', [4], b'n' * 8)}
        tensors = {}
        for p in encode.ex.PROJECTIONS:
            name = encode.ex.name(0, 5, p)
            old[name + '.weight'] = ('U8', [2], b'xx')
            old[name + '.weight_scale'] = ('U8', [1], b's')
            old[name + '.weight_scale_2'] = ('F32', [], b'g' * 4)
            k, n = (512, 2048) if p == 'down_proj' else (2048, 512)
            tensors[name + '.trellis'] = ('I16', [k // 16, n // 16, 48], bytes(k * n * 3 // 8))
            tensors[name + '.suh'] = ('F16', [k], b'\0<' * k)
            tensors[name + '.svh'] = ('F16', [n], b'\0<' * n)
            tensors[name + '.mul1'] = ('I32', [], struct.pack('<I', 0x83DCD12D))
        write_shard(src / 'model.safetensors', old, {'format': 'pt'})
        (src / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'drama_experts': {'0': [5]}, 'drama_head_rows': [1, 2]},
            'weight_map': {n: 'model.safetensors' for n in old}}))
        (src / 'codec.pth').write_bytes(b'bf16 codec')
        (src / 'config.json').write_text('{}')
        (src / 'MANIFEST.sha256').write_text('stale')
        shard = layers / 'layer-00.safetensors'
        encode.ex.write_safetensors(shard, tensors, {'format': 'exl3', 'bits': '3.0'})
        record = {'rows': {'5': 2}, 'error': {'5': {p: 0.0 for p in encode.ex.PROJECTIONS}},
                  'sha256': encode.ex.sha256(shard)}
        (layers / 'layer-00.json').write_text(json.dumps(record))
        (layers / 'recipe.json').write_text(json.dumps({'kept': encode.ex.sha256(encode.KEPT)}))
        return src, layers, dst, dense, tensors

    def test_preserves_dense_codec_and_pruning_while_replacing_expert_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, layers, dst, dense, tensors = self.fixture(Path(tmp))
            with patch.object(encode, 'kept_experts', return_value={0: [5]}):
                assemble.assemble(src, layers, dst)
            index = json.loads((dst / 'model.safetensors.index.json').read_text())
            self.assertEqual(set(index['weight_map']), {dense} | set(tensors))
            self.assertEqual(index['metadata']['drama_experts'], {'0': [5]})
            self.assertEqual(index['metadata']['drama_head_rows'], [1, 2])
            header, start = read_header(dst / 'model.safetensors')
            self.assertEqual(tensor_bytes(dst / 'model.safetensors', header, start, dense), b'n' * 8)
            self.assertEqual((dst / 'codec.pth').read_bytes(), b'bf16 codec')
            self.assertEqual((dst / 'exl3-layer-00.safetensors').read_bytes(),
                             (layers / 'layer-00.safetensors').read_bytes())
            self.assertFalse((dst / 'MANIFEST.sha256').exists())
            self.assertEqual(index['metadata']['total_size'], 8 + sum(len(t[2]) for t in tensors.values()))

    def test_missing_layer_cannot_produce_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, layers, dst, _, _ = self.fixture(Path(tmp))
            (layers / 'layer-00.json').unlink()
            with patch.object(encode, 'kept_experts', return_value={0: [5]}):
                with self.assertRaisesRegex(ValueError, 'unfinished'):
                    assemble.assemble(src, layers, dst)
            self.assertFalse(dst.exists())

    def test_invalid_codebook_rejected_even_with_matching_shard_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, layers, dst, _, tensors = self.fixture(Path(tmp))
            name = encode.ex.name(0, 5, 'gate_proj') + '.mul1'
            tensors[name] = ('I32', [], bytes(4))
            shard = layers / 'layer-00.safetensors'
            encode.ex.write_safetensors(shard, tensors, {'format': 'exl3', 'bits': '3.0'})
            record = json.loads((layers / 'layer-00.json').read_text())
            record['sha256'] = encode.ex.sha256(shard)
            (layers / 'layer-00.json').write_text(json.dumps(record))
            with patch.object(encode, 'kept_experts', return_value={0: [5]}):
                with self.assertRaisesRegex(ValueError, 'mul1'):
                    assemble.assemble(src, layers, dst)
            self.assertFalse(dst.exists())


if __name__ == '__main__':
    unittest.main()
