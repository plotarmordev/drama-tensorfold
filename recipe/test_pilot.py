"""recipe/pilot.py sizing: faked nvidia-smi and /proc/meminfo, no GPU, model or child process."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pilot

GIB = 1024**3
INT3_WEIGHTS = 14_500_000_000  # the 14.5 GB int3 build's checkpoint files


class SizingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meminfo = self.root / 'meminfo'
        self.bin = self.root / 'bin'
        self.bin.mkdir()

    def machine(self, host_gib: int, name: str, device_mib: int) -> pilot.Sizing:
        self.meminfo.write_text(f'MemTotal: {host_gib * 1024**2} kB\nMemAvailable: {host_gib * 1024**2} kB\n')
        smi = self.bin / 'nvidia-smi'
        smi.write_text(f'#!/bin/sh\nprintf "%s\\n" "{name}, {device_mib}"\n')
        smi.chmod(0o755)
        with patch.object(pilot, 'MEMINFO', self.meminfo), patch.dict(os.environ, {'PATH': str(self.bin)}):
            return pilot.sizing()

    def test_a_gb10_keeps_its_measured_figures(self):
        machine = self.machine(128, 'NVIDIA GB10', 122880)
        self.assertTrue(machine.unified)
        self.assertEqual((machine.cgroup_cap, machine.headroom, machine.admission_floor, machine.abort_below),
                         (54 * GIB, 30 * GIB, 88 * GIB, 34 * GIB))
        self.assertEqual((machine.reserve_gib, machine.cuda_limit_gib), (34, 54))
        self.assertEqual(pilot.guard(INT3_WEIGHTS, machine, False), (88 * GIB, 34 * GIB))

    def test_a_16_gb_discrete_card_loads_the_int3_build(self):
        machine = self.machine(32, 'NVIDIA GeForce RTX 5060 Ti', 16380)
        self.assertFalse(machine.unified)
        self.assertEqual(machine.reserve_gib, 0.5)
        self.assertEqual(machine.cuda_limit_gib, (16380 * 1024**2 - GIB) / GIB)
        self.assertEqual((machine.cgroup_cap, machine.headroom, machine.admission_floor, machine.abort_below),
                         (8 * GIB, 4 * GIB, 14 * GIB, 6 * GIB))
        # The cap covers issue #68's 14.77 GiB peak, and the reserve leaves room for the 14.5 GB weights.
        self.assertGreater(machine.cuda_limit_gib, 14.77)
        self.assertGreaterEqual(machine.cuda_limit_gib - machine.reserve_gib, INT3_WEIGHTS / GIB)
        self.assertGreater(34 + INT3_WEIGHTS / GIB, 16380 * 1024**2 / GIB)
        self.assertEqual(pilot.guard(INT3_WEIGHTS, machine, False), (14 * GIB, 6 * GIB))
        self.assertLessEqual(pilot.guard(INT3_WEIGHTS, machine, False)[0], 32 * GIB)

    def test_the_host_floor_follows_host_ram(self):
        machine = self.machine(64, 'NVIDIA GeForce RTX 4090', 24576)
        self.assertEqual((machine.cgroup_cap, machine.admission_floor, machine.abort_below),
                         (16 * GIB, 22 * GIB, 6 * GIB))
        self.assertEqual(machine.cuda_limit_gib, 23)

    def test_a_gpu_without_usable_memory_or_nvidia_smi_is_refused(self):
        self.meminfo.write_text('MemTotal: 33554432 kB\n')
        smi = self.bin / 'nvidia-smi'
        for output, message in (('NVIDIA GeForce RTX 5060 Ti, N/A', 'did not report'),
                                ('NVIDIA GeForce RTX 5060 Ti, 1024', 'driver reserve')):
            with self.subTest(output=output):
                smi.write_text(f'#!/bin/sh\nprintf "%s\\n" "{output}"\n')
                smi.chmod(0o755)
                with patch.object(pilot, 'MEMINFO', self.meminfo), patch.dict(os.environ, {'PATH': str(self.bin)}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        pilot.sizing()
        smi.unlink()
        with patch.object(pilot, 'MEMINFO', self.meminfo), patch.dict(os.environ, {'PATH': str(self.bin)}):
            with self.assertRaisesRegex(RuntimeError, 'nvidia-smi is unavailable'):
                pilot.sizing()


if __name__ == '__main__':
    unittest.main()
