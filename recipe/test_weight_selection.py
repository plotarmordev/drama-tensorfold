"""Deterministic checkpoint/launcher tests: no network, model, GPU or child process."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pilot
import pilot_macos
import weight_selection as selection

GIB = 1024**3


class WeightSelectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.recipe = self.root / 'recipe'
        (self.recipe / 'checksums').mkdir(parents=True)
        self.sources = json.loads((selection.RECIPE / 'sources.json').read_text())
        (self.recipe / 'sources.json').write_text(json.dumps(self.sources))
        self.stack.enter_context(patch.object(selection, 'REPO', self.root))
        self.stack.enter_context(patch.object(selection, 'RECIPE', self.recipe))
        # Real files and real SHA-256, distinct per selection so using the wrong
        # manifest cannot accidentally pass. No production weights are opened.
        for mode, spec in selection.WEIGHTS.items():
            target = self.root / spec['dir']
            target.mkdir()
            entries = []
            for name in ('model.safetensors', 'codec.pth', 'config.json'):
                content = f'{mode}:{name}'.encode()
                (target / name).write_bytes(content)
                entries.append(f'{hashlib.sha256(content).hexdigest()}  {name}\n')
            (self.recipe / 'checksums' / spec['pins']).write_text(''.join(entries))
            executable = selection.native_path(mode, None)
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b'not an executable model')
            executable.chmod(0o700)
        self.download = self.stack.enter_context(
            patch.object(selection.subprocess, 'run', side_effect=AssertionError('unexpected download')))


    def test_guidance_override_replaces_all_defaults(self):
        self.assertEqual(selection.serve_arguments('nvfp4', ['--max-frames', '8']),
                         ['--cfg-gamma', '1.5', '--cfg-batched', '--max-frames', '8'])
        for flag in ('--cfg-gamma', '--cfg-gamma=2', '--cfg-batched',
                     '--cfg-allow-request-gamma', '--cfg-allow-request-gamma=true'):
            arguments = [flag, '2'] if flag == '--cfg-gamma' else [flag]
            self.assertEqual(selection.serve_arguments('nvfp4', arguments), arguments)
        for mode in ('full', None):
            arguments = ['--max-frames', '8']
            self.assertEqual(selection.serve_arguments(mode, arguments), arguments)
            self.assertEqual(arguments, ['--max-frames', '8'])

    def test_native_environment_preserves_full_model_prefill(self):
        for mode in ('full', None):
            with self.subTest(mode=mode):
                self.assertEqual(selection.native_environment(mode), ['TF_DRAMA_PREFILL_ROWS=128'])
        self.assertEqual(selection.native_environment('nvfp4'), [])

    def test_explicit_paths_override_defaults_but_not_manifest(self):
        override = self.root / 'custom-checkpoint'
        selection.checkpoint_path('nvfp4', None).rename(override)
        binary = self.root / 'custom-native'
        self.assertEqual(selection.native_path('nvfp4', binary), binary)
        provenance = selection.prepare_checkpoint('nvfp4', override)
        self.assertEqual(provenance['checkpoint_path'], str(override))
        self.assertTrue(provenance['checkpoint_verified'])
        with self.assertRaises(ValueError):
            selection.prepare_checkpoint('full', override)
        self.download.assert_not_called()

    def test_missing_override_never_downloads(self):
        with self.assertRaises(ValueError):
            selection.prepare_checkpoint('nvfp4', self.root / 'missing')
        self.download.assert_not_called()
        self.assertFalse((self.root / 'missing').exists())

    def test_existing_missing_or_corrupt_input_is_not_repaired(self):
        for mode in selection.WEIGHTS:
            target = selection.checkpoint_path(mode, None)
            (target / 'model.safetensors').unlink()
            with self.subTest(mode=mode, damage='missing'):
                with self.assertRaises(ValueError):
                    selection.prepare_checkpoint(mode, None)
            (target / 'model.safetensors').write_bytes(b'corrupt')
            with self.subTest(mode=mode, damage='corrupt'):
                with self.assertRaises(ValueError):
                    selection.prepare_checkpoint(mode, None)
            self.assertEqual((target / 'model.safetensors').read_bytes(), b'corrupt')
        self.download.assert_not_called()

    def test_downloaded_content_is_verified_before_acceptance(self):
        target = selection.checkpoint_path('nvfp4', None)
        saved = self.root / 'download-fixture'
        target.rename(saved)

        def download(_command, **_kwargs):
            saved.rename(target)
            (target / 'config.json').write_bytes(b'bad download')

        self.download.side_effect = download
        with patch.object(selection.shutil, 'which', return_value='/fixture/hf'):
            with self.assertRaises(ValueError):
                selection.prepare_checkpoint('nvfp4', None)
        self.download.assert_called_once()

    def test_legacy_has_no_claim_of_verified_pinned_source(self):
        target = self.root / 'operator-checkpoint'
        provenance = selection.prepare_checkpoint(None, target)
        self.assertEqual(provenance, {'weights_mode': 'explicit', 'checkpoint_path': str(target),
                                     'checkpoint_source': None, 'checkpoint_manifest': None,
                                     'checkpoint_verified': False})
        with self.assertRaises(ValueError):
            selection.checkpoint_path(None, None)
        with self.assertRaises(ValueError):
            selection.native_path(None, None)
        self.download.assert_not_called()

    def launcher_arguments(self, mode, override=None, binary=None):
        lock = self.root / 'gpu.lock'
        lock.touch()
        args = ['pilot', '--runtime-dir', str(self.root / 'runtime'),
                '--evidence-dir', str(self.root / 'evidence'), '--gpu-lock', str(lock)]
        if mode is not None:
            args += ['--weights', mode]
        if override is not None:
            args += ['--checkpoint', str(override)]
        if binary is not None:
            args += ['--native-bin', str(binary)]
        return args

    def test_both_launchers_refuse_wrong_selected_override_before_spawn(self):
        for launcher in (pilot, pilot_macos):
            with self.subTest(launcher=launcher.__name__):
                arguments = self.launcher_arguments('nvfp4', selection.checkpoint_path('full', None))
                with patch.object(sys, 'argv', arguments), \
                        patch.object(launcher, 'memory_available', return_value=100 * GIB), \
                        patch.object(launcher.subprocess, 'Popen') as spawn:
                    if launcher is pilot:
                        with self.assertRaises(SystemExit):
                            launcher.main()
                    else:
                        self.assertEqual(launcher.main(), 1)
                        completion = json.loads((self.root / 'evidence/completion.json').read_text())
                        self.assertTrue(completion['group_settled'])
                    spawn.assert_not_called()
                self.download.assert_not_called()

    def test_mac_missing_override_fails_before_spawn_and_invalidates_reused_evidence(self):
        arguments = self.launcher_arguments('nvfp4', self.root / 'missing')
        evidence = self.root / 'evidence'
        evidence.mkdir()
        (evidence / 'ready.json').write_text('old readiness')
        arguments.append('--evidence-reuse')
        with patch.object(sys, 'argv', arguments), \
                patch.object(pilot_macos, 'memory_available', return_value=100 * GIB), \
                patch.object(pilot_macos.subprocess, 'Popen') as spawn:
            self.assertEqual(pilot_macos.main(), 1)
            spawn.assert_not_called()
        self.assertFalse((evidence / 'ready.json').exists())
        completion = json.loads((evidence / 'completion.json').read_text())
        self.assertIsNone(completion['server_pid'])
        self.download.assert_not_called()

    def test_linux_guard_is_the_documented_floor_unless_admission_is_external(self):
        self.assertEqual(pilot.guard(20 * GIB, False), (88 * GIB, 34 * GIB))
        self.assertEqual(pilot.guard(70 * GIB, False), (100 * GIB, 34 * GIB))
        self.assertEqual(pilot.guard(70 * GIB, True), (None, None))

    def test_both_launchers_admit_externally_below_the_floor(self):
        systemctl = subprocess.CompletedProcess([], 0, 'LoadState=not-found\nActiveState=inactive\n')
        for launcher in (pilot, pilot_macos):
            for external in (False, True):
                with self.subTest(launcher=launcher.__name__, external=external):
                    evidence = self.root / f'evidence-{launcher.__name__}-{external}'
                    arguments = self.launcher_arguments('nvfp4')
                    arguments[arguments.index('--evidence-dir') + 1] = str(evidence)
                    arguments += ['--port', str(free_port()), '--reserve-gib', '2']
                    if external:
                        arguments.append('--external-admission')
                    (self.root / 'runtime').mkdir(exist_ok=True)
                    with patch.object(sys, 'argv', arguments), \
                            patch.object(launcher, 'memory_available', return_value=10 * GIB), \
                            patch.object(launcher.subprocess, 'run', return_value=systemctl), \
                            patch.object(launcher.subprocess, 'Popen', side_effect=Spawned) as spawn:
                        if launcher is pilot and external:
                            with self.assertRaises(Spawned):
                                launcher.main()
                        elif launcher is pilot:
                            with self.assertRaises(SystemExit):
                                launcher.main()
                        else:
                            self.assertEqual(launcher.main(), 1)
                    self.assertEqual(spawn.call_count, 1 if external else 0)
                    if not external:
                        continue
                    session = json.loads((evidence / 'session.json').read_text())
                    self.assertTrue(session['admission'].startswith('external'))
                    self.assertIsNone(session['required_memavailable_bytes'])
                    self.assertIsNone(session['abort_threshold_bytes'])
                    self.assertEqual(session['engine_memory_reserve_gib'], 2)
                    command = spawn.call_args.args[0]
                    environment = spawn.call_args.kwargs.get('env') or {}
                    self.assertTrue('TENSORFOLD_MEMORY_RESERVE_GIB=2' in command
                                    or environment.get('TENSORFOLD_MEMORY_RESERVE_GIB') == '2')
        self.download.assert_not_called()

    def test_external_admission_refuses_an_admission_override_or_a_bad_reserve(self):
        arguments = self.launcher_arguments('nvfp4') + ['--external-admission', '--min-available-gib', '40']
        with patch.object(sys, 'argv', arguments), \
                patch.object(pilot_macos, 'memory_available', return_value=100 * GIB), \
                patch.object(pilot_macos.subprocess, 'Popen') as spawn:
            self.assertEqual(pilot_macos.main(), 1)
            spawn.assert_not_called()
        completion = json.loads((self.root / 'evidence/completion.json').read_text())
        self.assertIn('--external-admission skips admission', '\n'.join(completion['failures']))
        for reserve in ('nan', '-1', 'inf'):
            with self.subTest(reserve=reserve), \
                    patch.object(sys, 'argv', self.launcher_arguments('nvfp4') + [f'--reserve-gib={reserve}']), \
                    patch.object(pilot.subprocess, 'Popen') as spawn:
                with self.assertRaises(SystemExit):
                    pilot.main()
                spawn.assert_not_called()


class Spawned(Exception):
    """Raised by a mocked Popen: the launcher got past admission and started the server."""


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]



if __name__ == '__main__':
    unittest.main()
