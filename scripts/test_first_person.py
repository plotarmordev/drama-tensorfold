"""The public prose (every markdown file, the NOTICE and the Hugging Face cards kept here) speaks as I, never we."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLURAL = re.compile(r"\b(we|we've|we're|we'll|we'd|us|our|ours|ourselves|let's)\b", re.IGNORECASE)


def plural_lines(text):
    return [n for n, line in enumerate(text.splitlines(), 1) if PLURAL.search(line)]


class FirstPersonTests(unittest.TestCase):
    def test_finds_every_plural_form(self):
        self.assertEqual(plural_lines('I ran it.\nWe found none.\nnot ours\nlet us see\nUS spelling\nusing trust'),
                         [2, 3, 4, 5])

    def test_public_prose_says_i(self):
        files = subprocess.run(['git', 'ls-files', '*.md', 'NOTICE'], cwd=ROOT, check=True, capture_output=True,
                               text=True).stdout.split()
        self.assertIn('results/full-vs-quant-01/hub-card.md', files)
        found = {f: lines for f in files if (lines := plural_lines((ROOT / f).read_text(encoding='utf-8')))}
        self.assertEqual(found, {})


if __name__ == '__main__':
    unittest.main()
