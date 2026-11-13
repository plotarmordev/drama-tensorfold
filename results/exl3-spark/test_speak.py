import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('speak', Path(__file__).with_name('speak.py'))
speak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speak)


def take(seconds, finish='stop'):
    return {'arm': 'ref', 'salt': 0, 'item': 'short-01', 'seconds': seconds, 'bytes': 1, 'finish': finish}


class Summary(unittest.TestCase):
    def test_rows_aggregate_both_salts_of_an_arm(self):
        rows = speak.summary([
            {'arm': 'ref', 'salt': 0, 'load_s': 20.0, 'peak_bytes': 15 * 2**30, 'repeat_identical': True,
             'takes': [take(10.0), take(20.0), take(30.0), take(50.0, 'length')]},
            {'arm': 'ref', 'salt': 1, 'load_s': 30.0, 'peak_bytes': 16 * 2**30, 'repeat_identical': False,
             'takes': [take(40.0), take(60.0), take(70.0), take(80.0)]}])
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(25.0, row['load_s'])
        self.assertEqual(16.0, row['peak_gib'])
        self.assertEqual(8, row['takes'])
        self.assertEqual(45.0, row['seconds_mean'])
        self.assertEqual(50.0, row['seconds_p50'])
        self.assertEqual(360.0, row['seconds_total'])
        self.assertEqual(1, row['capped'])
        self.assertFalse(row['repeat_identical'], 'one salt whose repeat differed fails the arm')

    def test_seconds_are_sorted_before_the_median(self):
        row = speak.summary([{'arm': 'builtin', 'salt': 0, 'load_s': 1.0, 'peak_bytes': 0, 'repeat_identical': True,
                              'takes': [take(9.0), take(1.0), take(5.0)]}])[0]
        self.assertEqual(5.0, row['seconds_p50'])
        self.assertEqual(3, row['takes'])


class Corpus(unittest.TestCase):
    def test_card_37_corpus_is_the_24_texts_with_ids(self):
        self.assertEqual(24, len(speak.CORPUS))
        self.assertEqual({'short-01', 'medium-05', 'long-06'},
                         {item['id'] for item in speak.CORPUS} & {'short-01', 'medium-05', 'long-06'})


if __name__ == '__main__':
    unittest.main()
