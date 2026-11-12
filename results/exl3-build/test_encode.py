import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import encode
from encode import ce, ex


def stored(w):
    return {'trellis': np.full((1, 3), 7, np.int16), 'suh': np.ones(w.shape[1], np.float16),
            'svh': np.ones(w.shape[0], np.float16), 'mul1': np.array(-1, np.int32)}


class EncodeTest(unittest.TestCase):
    def run_encode(self, out, kept):
        seen = []

        def hessians(x, gate, up):
            seen.append(len(x))
            return np.eye(4), np.eye(2)

        x = np.arange(12.0).reshape(3, 4)
        picks = np.array([[0, 5], [5, 1], [0, 1]])  # expert 5 has two rows, expert 9 none
        weights = {'gate_proj': np.ones((2, 4)), 'up_proj': np.ones((2, 4)), 'down_proj': np.ones((4, 2))}
        encode.encode(out, kept, lambda layer: (x, picks), lambda name: weights[name.rsplit('.', 1)[1]], hessians,
                      lambda w, h: (w - 0.5, stored(w)))
        return seen

    def test_layer_holds_every_projection_and_its_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen = self.run_encode(out, {3: [5, 9]})
            # the expert the rows never pick takes all of its layer's rows
            self.assertEqual(seen, [2, 3])
            tensors, metadata = ex.read_safetensors(out / 'layer-03.safetensors')
            self.assertEqual(metadata, {'format': 'exl3', 'bits': '3.0'})
            self.assertEqual(len(tensors), 2 * 3 * 4)
            self.assertEqual(tensors[ex.name(3, 9, 'down_proj') + '.trellis'], ('I16', [1, 3], b'\7\0' * 3))
            self.assertEqual(tensors[ex.name(3, 5, 'gate_proj') + '.suh'][:2], ('F16', [4]))
            record = json.loads((out / 'layer-03.json').read_text())
            self.assertEqual(record['rows'], {'5': 2, '9': 0})
            # W - Q is 0.5 everywhere: 0.25 times H's trace, 4 for gate/up and 2 for down, times W's 2 or 4 rows
            self.assertEqual(record['error']['5'], {'gate_proj': 2.0, 'up_proj': 2.0, 'down_proj': 2.0})

    def test_finished_layers_are_skipped_and_partial_ones_redone(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_encode(out, {0: [5]})
            (out / 'layer-01.safetensors').write_bytes(b'cut off')
            self.assertEqual(self.run_encode(out, {0: [5], 1: [5, 9]}), [2, 3])
            self.assertTrue((out / 'layer-01.json').exists())
            self.assertEqual(len(ex.read_safetensors(out / 'layer-01.safetensors')[0]), 2 * 3 * 4)

    def test_committed_layer_rejects_missing_or_changed_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_encode(out, {0: [5]})
            shard = out / 'layer-00.safetensors'
            original = shard.read_bytes()
            shard.unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete or damaged'):
                self.run_encode(out, {0: [5]})
            shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'incomplete or damaged'):
                self.run_encode(out, {0: [5]})

    def test_committed_layer_rejects_changed_expert_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_encode(out, {0: [5]})
            with self.assertRaisesRegex(ValueError, 'incomplete or damaged'):
                self.run_encode(out, {0: [5, 9]})

    def test_committed_layer_rejects_missing_projection_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_encode(out, {0: [5]})
            path = out / 'layer-00.json'
            record = json.loads(path.read_text())
            del record['error']['5']['down_proj']
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'incomplete or damaged'):
                self.run_encode(out, {0: [5]})

    def test_resume_rejects_changed_row_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, rows, out = (root / name for name in ('checkpoint', 'rows', 'out'))
            checkpoint.mkdir()
            rows.mkdir()
            (checkpoint / 'config.json').write_text('{}')
            (checkpoint / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
            (rows / 'rows.json').write_text(json.dumps({
                'identity': {'config_sha256': ex.sha256(checkpoint / 'config.json')}, 'rows': 1}))
            data = rows / 'layer-00.x.bf16'
            data.write_bytes(b'\x00\x00')
            quantizer = types.SimpleNamespace(
                __file__=str(Path(encode.__file__).parent.parent / 'exl3-ldlq/exllamav3_job.py'),
                import_quantize=lambda: None)
            with patch.dict('sys.modules', exllamav3_job=quantizer), patch.object(encode, 'kept_experts', return_value={}):
                encode.main(checkpoint, rows, out)
                encode.main(checkpoint, rows, out)
                data.write_bytes(b'\x00\x01')
                with self.assertRaisesRegex(ValueError, 'another encoder recipe'):
                    encode.main(checkpoint, rows, out)


class KeptTest(unittest.TestCase):
    def test_every_kept_expert_of_the_published_quant(self):
        kept = encode.kept_experts()
        self.assertEqual(sum(map(len, kept.values())), 5900)
        index = ce.NVFP4 / 'model.safetensors.index.json'
        if not index.exists():
            self.skipTest(f'{index} is on the Mac')
        self.assertEqual({(l, e) for l, experts in kept.items() for e in experts}, ce.kept_experts(index))


if __name__ == '__main__':
    unittest.main()
