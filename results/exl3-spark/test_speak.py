import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('speak', Path(__file__).with_name('speak.py'))
speak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speak)


def take(seconds, finish='stop'):
    return {'arm': 'ref', 'salt': 0, 'item': 'short-01', 'seed': 0, 'seconds': seconds,
            'bytes': 1, 'finish': finish}


class Summary(unittest.TestCase):
    def test_rows_aggregate_load_peak_and_seconds(self):
        rows = speak.summary([
            {'arm': 'ref', 'load_s': 30.0, 'peak_bytes': 15 * 2**30,
             'takes': [take(10.0), take(20.0), take(30.0), take(40.0), take(50.0, 'length')]}])
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(30.0, row['load_s'])
        self.assertEqual(15.0, row['peak_gib'])
        self.assertEqual(5, row['takes'])
        self.assertEqual(30.0, row['seconds_mean'])
        self.assertEqual(30.0, row['seconds_p50'])
        self.assertEqual(150.0, row['seconds_total'])
        self.assertEqual(1, row['capped'])

    def test_seconds_are_sorted_before_the_median(self):
        row = speak.summary([{'arm': 'builtin', 'load_s': 1.0, 'peak_bytes': 0,
                              'takes': [take(9.0), take(1.0), take(5.0)]}])[0]
        self.assertEqual(5.0, row['seconds_p50'])
        self.assertEqual(3, row['takes'])

    def test_estimate_window_scales_each_arm_by_its_rtf(self):
        self.assertEqual(2 * 60 + 100 * 2.0 + 50 * 1.0,
                         speak.estimate_window_seconds(100, 50, 2.0, 1.0))


class Corpus(unittest.TestCase):
    def test_card_37_corpus_is_the_24_texts_with_ids(self):
        self.assertEqual(24, len(speak.CORPUS))
        self.assertEqual({'short-01', 'medium-05', 'long-06'},
                         {item['id'] for item in speak.CORPUS} & {'short-01', 'medium-05', 'long-06'})


if __name__ == '__main__':
    unittest.main()
