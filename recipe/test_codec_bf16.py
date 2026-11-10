import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

try:
    import torch
    import codec_bf16
except ModuleNotFoundError:
    torch = None
    codec_bf16 = None


@unittest.skipIf(torch is None, 'PyTorch is only needed for codec conversion')
class CodecBf16Test(unittest.TestCase):
    def test_converts_only_fp32_and_saved_checkpoint_round_trips(self):
        state = OrderedDict([
            ('weight', torch.tensor([0.0, -0.0, 1.0, 3.1415927], dtype=torch.float32)),
            ('existing', torch.tensor([2.0], dtype=torch.bfloat16)),
            ('mask', torch.tensor([True, False], dtype=torch.bool)),
        ])
        state._metadata = OrderedDict([('', {'version': 1})])
        counts = codec_bf16.convert(state)
        self.assertEqual(counts, {'torch.float32': 1, 'torch.bfloat16': 1, 'torch.bool': 1})
        self.assertEqual(state['weight'].dtype, torch.bfloat16)
        self.assertEqual(state['weight'].tolist(), [0.0, -0.0, 1.0, 3.140625])
        self.assertEqual(state['existing'].tolist(), [2.0])
        self.assertEqual(state['mask'].tolist(), [True, False])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'codec.pth'
            torch.save(state, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(loaded._metadata, state._metadata)
            self.assertTrue(torch.equal(loaded['weight'], state['weight']))
            self.assertTrue(torch.equal(loaded['mask'], state['mask']))

    def test_rejects_non_tensor_state_dict_entries(self):
        with self.assertRaisesRegex(ValueError, 'flat tensor state dict'):
            codec_bf16.convert({'not-a-tensor': 1})


if __name__ == '__main__':
    unittest.main()
