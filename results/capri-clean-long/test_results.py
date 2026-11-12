#!/usr/bin/env python3
"""CPU-only checks of the recorded 20-line results; no model, audio or catalog input."""
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results.json'
KNOWN = [
    'Yes. Since late this afternoon I am speaking on the new Drama model, with the cleaned long version of my voice.',
    'Good evening, Net. Four cards finished today and two are waiting for a review in the morning. Nothing needs you tonight.',
]


class CapriCleanLongTests(unittest.TestCase):
    def test_every_line_speaks_to_the_end_in_both_voices(self):
        rows = json.loads(RESULTS.read_text())['rows']
        for voice in ('capri-clean-long', 'shell-nolaugh'):
            lines = [row for row in rows if row['voice'] == voice]
            self.assertEqual(len(lines), 20)
            for row in lines:
                with self.subTest(voice=voice, line=row['line']):
                    self.assertEqual(row['status'], 200)
                    self.assertEqual(row['finish'], 'stop')
                    self.assertGreater(row['seconds'], 2.0)
                    self.assertLess(row['trailing_silence_s'], 1.0)

    def test_the_two_lines_that_dropped_out_on_12_november_are_covered(self):
        rows = json.loads(RESULTS.read_text())['rows']
        for text in KNOWN:
            for voice in ('capri-clean-long', 'shell-nolaugh'):
                row = next(row for row in rows if row['voice'] == voice and row['text'] == text)
                self.assertEqual(row['finish'], 'stop')

    def test_the_live_catalog_dropped_out_on_the_same_two_lines(self):
        rows = json.loads(RESULTS.read_text())['rows']
        before = [row for row in rows if row['voice'] == 'capri-clean-long-47s']
        self.assertEqual(len(before), 2)
        for row in before:
            self.assertEqual(row['finish'], 'silence')


if __name__ == '__main__':
    unittest.main()
