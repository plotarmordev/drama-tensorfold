"""calibrate_experts.py on a small synthetic rows directory: no model, GPU or network."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import calibrate_experts as ce

ROWS = 4
# The NVFP4 build's mix: rows 0-3 are one text, two audio and one generated row, against its 30 / 40 / 30 %.
TARGET = {'prompt_text': 3, 'prompt_audio': 4, 'generated': 3}
SHARES = {'prompt_text': 0.3, 'prompt_audio': 0.4, 'generated': 0.3}


def write_rows(out, config, picks_for):
    out.mkdir()
    manifest = {'schema': 'drama-probe-rows/1', 'layers': ce.LAYERS, 'rows': ROWS, 'chunks': [{'offset': 0, 'rows': ROWS}],
                'identity': {'config_sha256': hashlib.sha256(config).hexdigest()}}
    (out / 'rows.json').write_text(json.dumps(manifest))
    np.array([0, 1, 1, 2], np.uint8).tofile(out / 'class.u8')
    for layer in range(ce.LAYERS):
        stem = out / f'layer-{layer:02d}'
        np.zeros((ROWS, ce.HIDDEN), np.uint16).tofile(f'{stem}.x.bf16')
        np.array(picks_for(layer), '<i4').tofile(f'{stem}.picks.i32')
        np.full((ROWS, ce.TOP_K), 1 / ce.TOP_K, '<f4').tofile(f'{stem}.w.f32')


class CalibrateExpertsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.checkpoint = self.tmp / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'config.json').write_bytes(b'{}')

    def test_reports_kept_experts_without_rows_by_layer_and_the_class_mix(self):
        # Rows cover experts 0-31 in every layer; layer 5 keeps 40 and 41, which have none. Only 40 had NVFP4 rows.
        write_rows(self.tmp / 'rows', b'{}', lambda layer: [list(range(r * 8, r * 8 + 8)) for r in range(ROWS)])
        kept = {(layer, 0) for layer in range(ce.LAYERS)} | {(5, 40), (5, 41)}
        reference = {(layer, expert): 1 for layer in range(ce.LAYERS) for expert in range(ce.EXPERTS)} | {(5, 41): 0}
        report = ce.validate(self.tmp / 'rows', self.checkpoint, kept, reference, TARGET, weight_checks=False)
        self.assertEqual(report['class_rows'], {'prompt_text': 1, 'prompt_audio': 2, 'generated': 1})
        self.assertEqual(report['class_shares'], {'prompt_text': 0.25, 'prompt_audio': 0.5, 'generated': 0.25})
        self.assertEqual(report['nvfp4_class_shares'], SHARES)
        self.assertEqual(report['missing_kept_by_layer'], {'5': [40, 41]})
        self.assertEqual(report['missing_kept_nvfp4_routed'], {'5': [40]})
        self.assertEqual(report['covered_kept_experts'], len(kept) - 2)
        self.assertEqual(report['covered_experts'], ce.LAYERS * 32)
        self.assertEqual(report['kept_rows_min_median'], [0, 1])

    def test_usable_rows_cover_every_kept_expert_but_the_named_exception_in_the_nvfp4_mix_with_matching_statistics(self):
        usable = {'missing_kept_by_layer': {}, 'class_shares': {'prompt_text': 0.309, 'prompt_audio': 0.391, 'generated': 0.3},
                  'nvfp4_class_shares': SHARES, 'stats_check': {'ratio_p05_p95': [0.96, 1.04]}}
        self.assertEqual(ce.failures(usable), [])
        # The approved layer-36 experts may lack rows; any other kept expert without rows fails.
        self.assertEqual(ce.failures({**usable, 'missing_kept_by_layer': {'36': list(ce.NEVER_ROUTED[36])}}), [])
        for change, reason in (
                ({'missing_kept_by_layer': {'36': [0, 1]}}, "kept experts without rows: {'36': [1]}"),
                ({'missing_kept_by_layer': {'35': [0]}}, "kept experts without rows: {'35': [0]}"),
                # The 11 Nov capture's own mix against the NVFP4 build's 14.67 / 43.45 / 41.88 %.
                ({'class_shares': {'prompt_text': 0.1950, 'prompt_audio': 0.5131, 'generated': 0.2918},
                  'nvfp4_class_shares': {'prompt_text': 0.1467, 'prompt_audio': 0.4345, 'generated': 0.4188}},
                 'class shares'),
                ({'class_shares': {**usable['class_shares'], 'generated': 0.311}}, 'class shares'),
                ({'stats_check': {'ratio_p05_p95': [0.94, 1.0]}}, 'err_h ratio'),
                ({'stats_check': {'ratio_p05_p95': [1.0, 1.06]}}, 'err_h ratio')):
            with self.subTest(change=change):
                problems = ce.failures({**usable, **change})
                self.assertEqual(len(problems), 1)
                self.assertIn(reason, problems[0])

    def test_rejects_a_row_that_picks_an_expert_twice(self):
        write_rows(self.tmp / 'rows', b'{}', lambda layer: [[0] * 8] + [list(range(8))] * (ROWS - 1))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            ce.validate(self.tmp / 'rows', self.checkpoint, set(), {}, TARGET, weight_checks=False)

    def test_rejects_rows_from_another_checkpoint(self):
        write_rows(self.tmp / 'rows', b'{"other": 1}', lambda layer: [list(range(8))] * ROWS)
        with self.assertRaisesRegex(ValueError, 'identity'):
            ce.validate(self.tmp / 'rows', self.checkpoint, set(), {}, TARGET, weight_checks=False)

    def test_column_squares_count_each_routed_row_once_per_expert(self):
        x = np.array([[1, 2], [3, 4]], np.float32)
        picks = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10, 11, 12, 13, 14]])
        squares = ce.column_squares(x, picks, np)
        np.testing.assert_array_equal(squares[0], [10, 20])
        np.testing.assert_array_equal(squares[1], [1, 4])
        np.testing.assert_array_equal(squares[8], [9, 16])
        np.testing.assert_array_equal(squares[15], [0, 0])

    def test_selection_keeps_the_nvfp4_mix_and_every_kept_experts_rows(self):
        # 10 text, 20 audio and 15 generated rows for a 20 / 30 / 50 % mix: all 15 generated rows, 6 text, 9 audio.
        # Expert 99, kept in layer 2, is routed only in two text rows, which the floor keeps before the random fill.
        classes = np.array([0] * 10 + [1] * 20 + [2] * 15, np.uint8)
        picks = {layer: np.tile(np.arange(8, dtype='<i4'), (len(classes), 1)) for layer in range(ce.LAYERS)}
        picks[2][[3, 7], 0] = 99
        keep = ce.select(classes, picks, {(2, 99)}, {'prompt_text': 2, 'prompt_audio': 3, 'generated': 5}, np)
        self.assertEqual(np.bincount(classes[keep]).tolist(), [6, 9, 15])
        self.assertTrue({3, 7} <= set(keep.tolist()))

    def test_best_rank_is_the_highest_place_an_expert_reaches_in_any_row(self):
        # Logits per row: expert 0 = x0, expert 1 = x1, expert 2 = -x0 - x1. Expert 2 is never above third.
        x = np.array([[3, 1], [1, 2], [0.5, 0.25]], np.float32)
        gate = np.array([[1, 0], [0, 1], [-1, -1]], np.float32)
        self.assertEqual(ce.best_ranks(x, gate, [0, 1, 2], np), {0: 0, 1: 0, 2: 2})
        # A tie ranks the higher expert ID lower: expert 1 tied with expert 0 places second.
        self.assertEqual(ce.best_ranks(np.array([[1, 1]], np.float32), gate, [0, 1], np), {0: 0, 1: 1})

    def test_e4m3_and_e2m1_decoding(self):
        e4m3 = ce.e4m3(np.array([0x38, 0xB8, 0x7E, 0x01, 0x00]), np)
        np.testing.assert_array_equal(e4m3, [1, -1, 448, 2.0 ** -9, 0])
        self.assertTrue(np.isnan(ce.e4m3(np.array([0x7F, 0xFF]), np)).all())

        class Packed:
            def __call__(self, name):
                return {'.weight': np.array([[0x21, 0xF7] + [0] * 6], np.uint8), '.weight_scale': np.array([[2]], np.float32),
                        '.weight_scale_2': np.float32(0.5)}[name[name.index('.'):]]

        # Low nibble first: 0x21 -> 0.5, 1.0; 0xF7 -> 6, -6; times 2 * 0.5; one scale per 16 values.
        np.testing.assert_array_equal(ce.nvfp4(Packed(), 'w', np), [[0.5, 1, 6, -6] + [0] * 12])

    def test_requests_mix_every_line_alone_and_after_every_reference(self):
        clips = [{'ref_text': 'a', 'ref_codes': 'a.codes'}, {'ref_text': 'b', 'ref_codes': 'b.codes'}]
        out = ce.requests(['one', 'two'], clips)
        self.assertEqual(len(out), 2 * len(ce.SEEDS) * (1 + len(clips)) + 1)
        self.assertEqual(sum('ref_codes' in r for r in out), 2 * len(ce.SEEDS) * len(clips))
        self.assertEqual(len({json.dumps(r, sort_keys=True) for r in out}), len(out))
        self.assertEqual(out[-1]['text'], '\n\n'.join(['one', 'two'] * 3))


if __name__ == '__main__':
    unittest.main()
