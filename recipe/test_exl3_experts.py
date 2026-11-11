"""exl3_experts.py end to end on synthetic checkpoints and rows, with a stand-in encoder: no model, GPU or TensorFold."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import calibrate_experts as ce
import exl3_experts as ex

RECIPE = Path(__file__).resolve().parent
ROWS, ROUTED, UNROUTED, LAYERS = 6, 3, 7, (0, 1)
PICKS = [[ROUTED, *range(10, 17)], [*range(20, 28)], [ROUTED, *range(10, 17)], [*range(20, 28)], [*range(20, 28)],
         [ROUTED, *range(10, 17)]]
CLASSES = [0, 1, 2, 2, 1, 2]
WEIGHTS = {'prompt_text': 0.5, 'prompt_audio': 1.0, 'generated': 2.0}
# Reports trace(H) as its EXL3 error and writes one group of the real encoder's tensor names and shapes, with suh
# carrying diag(H) so a shard's bytes depend on the rows each expert was encoded against.
FAKE_ENCODER = f"""#!{sys.executable}
import os
import sys
import numpy as np
sys.path.insert(0, {str(RECIPE)!r})
import exl3_experts as ex
w, h = np.load(sys.argv[1]), np.load(sys.argv[2])
n, k = w.shape
halves = round(float(sys.argv[3]) * 2)
os.makedirs(sys.argv[4])
ex.write_safetensors(sys.argv[4] + '/model.safetensors', {{
    'weight.trellis': ('I16', [k // 16, n // 16, 16 * halves // 2], bytes(k * n * halves // 4)),
    'weight.suh': ('F16', [k], np.diag(h).astype('<f2').tobytes()), 'weight.svh': ('F16', [n], bytes(2 * n)),
    'weight.mul1': ('I32', [1], bytes(4))}}, {{'format': 'exl3'}})
print(f'hessian_error exl3={{float(np.trace(h))!r}} plain=0 nvfp4=0')
"""
ONE_E4M3 = 0x38


def projection_shapes(projection):
    return (ce.HIDDEN, 512) if projection == 'down_proj' else (512, ce.HIDDEN)


def checkpoints(root):
    """An FP8 original whose expert weights are all 1 and a published NVFP4 whose are all 0.5, for both experts of
    every layer."""
    source, nvfp4 = {}, {}
    for layer in LAYERS:
        for expert in (ROUTED, UNROUTED):
            for p in ex.PROJECTIONS:
                n, k = projection_shapes(p)
                prefix = ex.name(layer, expert, p)
                source[prefix + '.weight'] = ('F8_E4M3', [n, k], bytes([ONE_E4M3]) * (n * k))
                source[prefix + '.weight_scale_inv'] = ('F32', [n // 128, k // 128],
                                                        np.ones(n * k // 16384, '<f4').tobytes())
                nvfp4[prefix + '.weight'] = ('U8', [n, k // 2], bytes([0x11]) * (n * k // 2))
                nvfp4[prefix + '.weight_scale'] = ('F8_E4M3', [n, k // 16], bytes([ONE_E4M3]) * (n * k // 16))
                nvfp4[prefix + '.weight_scale_2'] = ('F32', [], np.ones(1, '<f4').tobytes())
    for directory, tensors in (('checkpoint', source), ('nvfp4', nvfp4)):
        (root / directory).mkdir()
        ex.write_safetensors(root / directory / 'model.safetensors', tensors, {})
        (root / directory / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {k: 'model.safetensors' for k in tensors}}))
    (root / 'checkpoint/config.json').write_bytes(b'{}')


def rows(root):
    """Each layer's calibration rows, other rows per layer, their classes and class weights; returns layer 0's x as
    the converter reads it back."""
    out = root / 'rows'
    out.mkdir()
    (out / 'rows.json').write_text(json.dumps({'rows': ROWS, 'identity': {'config_sha256': hashlib.sha256(b'{}').hexdigest()}}))
    (out / 'report.json').write_text(json.dumps({'class_weights': WEIGHTS}))
    np.array(CLASSES, np.uint8).tofile(out / 'class.u8')
    for layer in LAYERS:
        x = np.random.default_rng(layer).standard_normal((ROWS, ce.HIDDEN)).astype('<f4')
        (x.view('<u4') >> 16).astype('<u2').tofile(out / f'layer-{layer:02d}.x.bf16')
        np.array(PICKS, '<i4').tofile(out / f'layer-{layer:02d}.picks.i32')
        np.full((ROWS, ce.TOP_K), 1 / ce.TOP_K, '<f4').tofile(out / f'layer-{layer:02d}.w.f32')
    return ce.bf16(out / 'layer-00.x.bf16', ROWS, np).astype(np.float64)


def weighted(x, c):
    # Accelerate's matmul raises FP flags on finite inputs, as in exl3_experts.convert_expert.
    with np.errstate(all='ignore'):
        return (x.T * c) @ x / c.sum()


class Exl3ExpertsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        checkpoints(self.root)
        self.x = rows(self.root)
        self.encoder = self.root / 'tf-exl3-encode'
        self.encoder.write_text(FAKE_ENCODER)
        self.encoder.chmod(0o755)

    def run_converter(self, out, workers):
        r = self.root
        return subprocess.run([sys.executable, str(RECIPE / 'exl3_experts.py'), '--encoder', str(self.encoder),
                               '--checkpoint', str(r / 'checkpoint'), '--nvfp4', str(r / 'nvfp4'), '--rows',
                               str(r / 'rows'), '--output', str(r / out), '--layers', *map(str, LAYERS),
                               '--workers', str(workers)], capture_output=True)

    def convert(self, out='out', workers=2):
        self.run_converter(out, workers).check_returncode()
        return {bits: json.loads((self.root / f'{out}/exl3-{bits}/exl3-provenance.json').read_text()) for bits in ex.BITS}

    def test_each_expert_is_measured_on_its_own_class_weighted_rows_against_the_published_nvfp4(self):
        provenance = self.convert()
        c = np.array([WEIGHTS[ce.CLASSES[k]] for k in CLASSES])
        h_routed, h_all = weighted(self.x[[0, 2, 5]], c[[0, 2, 5]]), weighted(self.x, c)
        gate = provenance['3.0']['layers']['0']['gate_proj']
        # The stand-in reports trace(H); every weight is off its NVFP4 copy by 0.5 in each of 512 rows.
        self.assertAlmostEqual(gate['exl3_error'] / (np.trace(h_routed) + np.trace(h_all)), 1, places=9)
        self.assertAlmostEqual(gate['nvfp4_error'] / (512 * 0.25 * (h_routed.sum() + h_all.sum())), 1, places=9)
        self.assertEqual(provenance['3.0']['layers']['0']['no_rows'], [UNROUTED])
        self.assertEqual(provenance['3.0']['inputs']['config_sha256'], hashlib.sha256(b'{}').hexdigest())
        self.assertFalse(provenance['3.0']['complete'])

    def test_each_bitrate_is_a_checkpoint_of_every_expert_group_with_its_size(self):
        provenance = self.convert()
        for bits, halves in (('3.0', 6), ('2.5', 5)):
            out = self.root / f'out/exl3-{bits}'
            tensors, _ = ex.read_safetensors(out / 'model-00.safetensors')
            index = json.loads((out / 'model.safetensors.index.json').read_text())
            self.assertEqual({k for k, v in index['weight_map'].items() if v == 'model-00.safetensors'}, set(tensors))
            self.assertEqual(set(index['weight_map'].values()), {f'model-{l:02d}.safetensors' for l in LAYERS})
            self.assertEqual(provenance[bits]['bytes'],
                             sum((out / f'model-{l:02d}.safetensors').stat().st_size for l in LAYERS))
            self.assertEqual(provenance[bits]['settings']['bits_per_weight'], float(bits))
            self.assertGreater(provenance[bits]['nvfp4_bytes'], 0)
            for expert in (ROUTED, UNROUTED):
                group = ex.name(0, expert, 'down_proj')
                self.assertEqual(tensors[f'{group}.trellis'][:2], ('I16', [32, 128, 8 * halves]))
                self.assertEqual({k for k in tensors if k.startswith(group)},
                                 {f'{group}.{part}' for part in ('trellis', 'suh', 'svh', 'mul1')})

    def test_a_rerun_keeps_finished_experts_and_encodes_only_the_rest(self):
        first = self.convert()
        (self.root / f'out/work/layer-00/{UNROUTED:03d}.json').unlink()
        calls = self.root / 'calls'
        self.encoder.write_text(FAKE_ENCODER.replace('import sys\n', f'import sys\nopen({str(calls)!r}, "a").write("x")\n', 1))
        second = self.convert()
        self.assertEqual(calls.read_text(), 'x' * len(ex.PROJECTIONS) * len(ex.BITS))
        for bits in ex.BITS:
            self.assertEqual(first[bits]['layers']['0']['gate_proj'], second[bits]['layers']['0']['gate_proj'])

    def test_layers_converted_in_parallel_match_the_single_process_run_byte_for_byte(self):
        self.convert('single', workers=1)
        self.convert('parallel', workers=len(LAYERS))
        for bits in ex.BITS:
            for layer in LAYERS:
                shard = f'exl3-{bits}/model-{layer:02d}.safetensors'
                self.assertEqual((self.root / 'single' / shard).read_bytes(), (self.root / 'parallel' / shard).read_bytes())
        zero, one = (ex.read_safetensors(self.root / f'parallel/exl3-3.0/model-{l:02d}.safetensors')[0] for l in LAYERS)
        suh = f'{ex.name(0, ROUTED, "gate_proj")}.suh'
        self.assertNotEqual(zero[suh], one[suh.replace('layers.0.', 'layers.1.')])

    def test_a_work_folder_encoded_with_other_class_weights_is_not_resumed(self):
        self.convert()
        (self.root / 'rows/report.json').write_text(json.dumps({'class_weights': {**WEIGHTS, 'generated': 1.5}}))
        run = self.run_converter('out', workers=2)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(b'other Hessians', run.stderr)

    def test_a_failed_layer_leaves_the_others_work_but_writes_no_shard_index_or_provenance(self):
        (self.root / 'rows/layer-01.picks.i32').unlink()
        self.assertNotEqual(self.run_converter('out', workers=2).returncode, 0)
        out = self.root / 'out'
        self.assertTrue((out / f'work/layer-00/{UNROUTED:03d}.json').exists())
        for bits in ex.BITS:
            self.assertEqual(list((out / f'exl3-{bits}').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
