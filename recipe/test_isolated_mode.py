"""Both launchers start under python3 -I, which leaves the script's own directory off sys.path."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

RECIPE = Path(__file__).resolve().parent


class IsolatedModeTests(unittest.TestCase):
    def test_each_launcher_imports_its_own_modules_in_isolated_mode(self):
        for launcher in ('pilot.py', 'pilot_macos.py'):
            with self.subTest(launcher=launcher):
                result = subprocess.run([sys.executable, '-I', str(RECIPE / launcher), '--help'], cwd='/',
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--gpu-lock', result.stdout)


if __name__ == '__main__':
    unittest.main()
