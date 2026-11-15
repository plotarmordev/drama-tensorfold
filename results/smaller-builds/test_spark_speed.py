import unittest

from spark_speed import p50


class P50Tests(unittest.TestCase):
    def test_even_count_takes_the_lower_middle_value(self):
        self.assertEqual(p50([0.9, 0.5, 0.7, 0.6]), 0.6)

    def test_odd_count_takes_the_middle_value(self):
        self.assertEqual(p50([3, 1, 2]), 2)


if __name__ == '__main__':
    unittest.main()
