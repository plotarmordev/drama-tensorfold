import unittest

import numpy as np

import measure


class Int3Test(unittest.TestCase):
    def test_bit_planes_decode_to_signed_levels_times_scales(self):
        # one row of 16 weights: weight 0 is +6 (hi, lo), weight 3 is -2 (sign, lo), weight 15 is -4 (sign, hi)
        sign, hi, lo = 1 << 3 | 1 << 15, 1 << 0 | 1 << 15, 1 << 0 | 1 << 3
        packed = np.array([[sign & 255, sign >> 8, hi & 255, hi >> 8, lo & 255, lo >> 8]], np.uint8)
        tensors = {'w.weight': packed, 'w.weight_scale': np.array([[0.5]], np.float32),
                   'w.weight_scale_2': np.float32(0.25)}
        expected = np.zeros((1, 16))
        expected[0, 0], expected[0, 3], expected[0, 15] = 6 * 0.125, -2 * 0.125, -4 * 0.125
        np.testing.assert_array_equal(measure.int3(tensors.__getitem__, 'w'), expected)


def probe(int3, exl3_30, exl3_25):
    errors = lambda e: {'gate_proj': e, 'up_proj': 0.0, 'down_proj': 0.0}  # noqa: E731
    return {'int3': errors(int3), 'nvfp4': errors(1.0), 'exl3-3.0': errors(exl3_30), 'exl3-2.5': errors(exl3_25),
            'bytes': {'int3': 100, 'nvfp4': 130, 'exl3-3.0': 87, 'exl3-2.5': 73}}


class VerdictTest(unittest.TestCase):
    def test_beats_int3_only_with_lower_summed_error(self):
        v = measure.verdict({'0/1': probe(2.0, 1.5, 3.0), '0/2': probe(2.0, 2.0, 2.0)})
        self.assertTrue(v['exl3-3.0']['beats_int3'])
        self.assertFalse(v['exl3-2.5']['beats_int3'])
        self.assertAlmostEqual(v['exl3-3.0']['error_over_int3'], 3.5 / 4)
        self.assertAlmostEqual(v['exl3-2.5']['bytes_over_int3'], 0.73)


class ParityTest(unittest.TestCase):
    def test_matches_only_when_every_projection_is_within_parity(self):
        ours = {'0/1': probe(2.0, 1.0, 2.0)}
        ours['0/1']['exl3-3.0']['down_proj'] = 3.0
        reference = {'0/1': {'3.0': {'gate_proj': 1.0, 'up_proj': 1.0, 'down_proj': 3.0},
                             '2.5': {'gate_proj': 2.0, 'up_proj': 1.0, 'down_proj': 1.0}}}
        # zero error over a nonzero reference is a ratio of 0, within parity
        self.assertTrue(measure.parity(ours, reference)['matches'])
        # one projection 1.2x worse than exllamav3 fails it, though the sum over the expert is within 1.1x
        ours['0/1']['exl3-2.5']['gate_proj'] = 2.4
        result = measure.parity(ours, reference)
        self.assertFalse(result['matches'])
        self.assertAlmostEqual(result['worst'], 1.2)
        self.assertAlmostEqual(result['ours_over_exllamav3']['2.5']['0/1']['gate_proj'], 1.2)

    def test_exllamav3_verdict_scores_its_errors_on_our_bytes(self):
        reference = {'0/1': {'3.0': {'gate_proj': 0.3, 'up_proj': 0.1, 'down_proj': 0.1},
                             '2.5': {'gate_proj': 2.8, 'up_proj': 0.1, 'down_proj': 0.1}}}
        v = measure.parity({'0/1': probe(2.0, 1.5, 1.5)}, reference)['exllamav3_verdict']
        self.assertAlmostEqual(v['exl3-3.0']['error_over_int3'], 0.25)
        self.assertAlmostEqual(v['exl3-3.0']['bytes_over_int3'], 0.87)
        self.assertFalse(v['exl3-2.5']['beats_int3'])


if __name__ == '__main__':
    unittest.main()
