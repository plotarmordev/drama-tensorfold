import json
import unittest
from pathlib import Path

from pick import winner


def build(size, ref, builtin):
    return {'bytes': size, 'voices': {'ref': {'wer_mean': ref[0], 'failed': ref[1]},
                                      'builtin': {'wer_mean': builtin[0], 'failed': builtin[1]}}}


BASE = build(100, (0.02, 0), (0.05, 2))


class WinnerTests(unittest.TestCase):
    def test_smallest_qualifying_build_wins(self):
        builds = {'base': BASE, 'mid': build(90, (0.03, 0), (0.06, 2)), 'small': build(80, (0.01, 0), (0.04, 1))}
        self.assertEqual(winner(builds, 'base'), 'small')

    def test_exactly_double_wer_and_equal_failures_qualify(self):
        builds = {'base': BASE, 'small': build(80, (0.04, 0), (0.10, 2))}
        self.assertEqual(winner(builds, 'base'), 'small')

    def test_more_than_double_wer_in_one_voice_disqualifies(self):
        builds = {'base': BASE, 'mid': build(90, (0.02, 0), (0.05, 2)), 'small': build(80, (0.0401, 0), (0.05, 2))}
        self.assertEqual(winner(builds, 'base'), 'mid')

    def test_one_more_failed_take_in_one_voice_disqualifies(self):
        builds = {'base': BASE, 'small': build(80, (0.02, 1), (0.05, 2))}
        self.assertEqual(winner(builds, 'base'), 'base')

    def test_recorded_winner_follows_the_rule_over_every_measured_build(self):
        recorded = json.loads((Path(__file__).parent / 'builds.json').read_text())
        self.assertIn('trim-bf16-exl3-3.0', recorded['builds'])
        self.assertEqual(recorded['winner'], winner(recorded['builds'], recorded['base']))


if __name__ == '__main__':
    unittest.main()
