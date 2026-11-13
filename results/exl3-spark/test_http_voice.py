import hashlib
import importlib.util
import struct
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('http_voice', Path(__file__).with_name('http_voice.py'))
http_voice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http_voice)

CODEBOOKS = [[4095, 7, 9]] + [[c, c + 1, 1023] for c in range(9)]  # 10 codebooks x 3 frames


def fork_voice(codebooks=CODEBOOKS):
    raw = b''.join(struct.pack(f'<{len(row)}i', *row) for row in CODEBOOKS)
    return {'format': 'drama3-voice/2', 'source': {'kind': 'model-invented', 'note': 'three passages'},
            'references': [{'text': 'A lighthouse.', 'codes_sha256': hashlib.sha256(raw).hexdigest(),
                            'vq_codes': codebooks}]}


class Convert(unittest.TestCase):
    def test_codes_become_frame_rows_of_ten_and_the_source_synthetic(self):
        doc = http_voice.convert(fork_voice())
        self.assertEqual({'kind': 'synthetic', 'note': 'three passages'}, doc['source'])
        codes = doc['references'][0]['codes']
        self.assertEqual(3, len(codes))
        self.assertEqual([4095] + list(range(9)), codes[0])
        self.assertEqual([9] + [1023] * 9, codes[2])

    def test_codes_that_miss_their_sha256_are_refused(self):
        swapped = [row[::-1] for row in CODEBOOKS]
        with self.assertRaises(ValueError):
            http_voice.convert(fork_voice(swapped))


if __name__ == '__main__':
    unittest.main()
