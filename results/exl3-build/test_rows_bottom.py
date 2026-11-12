import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import rows_bottom
from rows_bottom import ce, encode


class BottomRowsTest(unittest.TestCase):
    def test_unapproved_missing_expert_blocks_preparation_and_saves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = Path(tmp)
            capture = pilot / 'eq/runs/fp8-calib-rows'
            checkpoint = pilot / 'checkpoint'
            quant = pilot / 'models/drama-3-NVFP4-Pruned-5900'
            for folder in (capture, checkpoint, quant):
                folder.mkdir(parents=True)
            (checkpoint / 'config.json').write_text('{}')
            manifest = {'schema': 'drama-probe-rows/1', 'layers': 1, 'rows': 3,
                        'identity': {'config_sha256': encode.ex.sha256(checkpoint / 'config.json')}}
            (capture / 'rows.json').write_text(json.dumps(manifest))
            np.array([0, 1, 2], np.uint8).tofile(capture / 'class.u8')
            np.ones((3, 2), dtype='<u2').tofile(capture / 'layer-00.x.bf16')
            np.array([[0, 1]] * 3, dtype='<i4').tofile(capture / 'layer-00.picks.i32')
            np.full((3, 2), 0.5, dtype='<f4').tofile(capture / 'layer-00.w.f32')
            (quant / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
                encode.ex.name(0, e, 'gate_proj') + '.weight': 'unused' for e in range(3)}}))
            (quant / 'nvfp4-provenance.json').write_text(json.dumps({
                'stats_pairs': dict.fromkeys(ce.CLASSES, 1),
                'layer_sources': [{'layer': 0, 'metadata': {'gptq': json.dumps({'rows': [3, 3, 0]})}}]}))
            out = pilot / 'out'
            with patch.multiple(ce, LAYERS=1, EXPERTS=3, TOP_K=2, HIDDEN=2, NVFP4=quant), \
                    patch.object(encode, 'kept_experts', return_value={0: [0, 1, 2]}), \
                    patch.object(ce, 'Tensors', return_value=lambda name: np.zeros((3, 2))), \
                    patch.object(ce, 'stats_check', return_value={'ratio_p05_p95': [1.0, 1.0]}):
                with self.assertRaisesRegex(ValueError, "kept experts without rows: .*2"):
                    rows_bottom.prepare(pilot, out)
            report = json.loads((out / 'rows-report.json').read_text())
            self.assertEqual(report['covered_kept_experts'], 2)
            self.assertEqual(report['missing_kept_by_layer'], {'0': [2]})
            self.assertTrue(report['selection_identity'])
            self.assertFalse((out / 'rows').exists())
            self.assertEqual(report['failures'], ["kept experts without rows: {'0': [2]}"])


if __name__ == '__main__':
    unittest.main()
