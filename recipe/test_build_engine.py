"""Exercise platform selection and real git-am/checksum handling without a compiler or network."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'recipe'
        self.source = self.root / 'source'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.source.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        self.env.update(GIT_AUTHOR_NAME='Fixture Contributor', GIT_COMMITTER_NAME='Fixture Contributor',
                        GIT_AUTHOR_EMAIL='fixture@example.invalid', GIT_COMMITTER_EMAIL='fixture@example.invalid',
                        GIT_AUTHOR_DATE='2026-01-01T00:00:00Z', GIT_COMMITTER_DATE='2026-01-01T00:00:00Z')
        self.git('init', '-q')
        self.git('config', 'commit.gpgsign', 'false')
        (self.source / '.zig-version').write_text('0.17.0\n')
        self.git('add', '.zig-version')
        self.git('commit', '-qm', 'Fixture engine base')
        self.base = self.git('rev-parse', 'HEAD').stdout.strip()
        script = self.repo / 'scripts' / 'build-engine.sh'
        script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'scripts' / 'build-engine.sh', script)
        self.series = {'full': 'engine', 'cuda': 'engine/nvfp4', 'metal': 'engine/nvfp4/macos'}
        for mode, relative in self.series.items():
            target = self.repo / relative
            patches = target / 'patches'
            patches.mkdir(parents=True)
            (target / 'UPSTREAM').write_text(
                f'TENSORFOLD_REPOSITORY={shlex.quote(str(self.source))}\n'
                f'TENSORFOLD_COMMIT={self.base}\n')
            filename = f'applied-{mode}'
            (self.source / filename).write_text(mode + '\n')
            self.git('add', filename)
            self.git('commit', '-qm', f'Fixture {mode} patch')
            content = self.git('format-patch', '-1', '--stdout').stdout.encode()
            (patches / '0001-fixture.patch').write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            (patches / 'SHA256SUMS').write_text(f'{digest}  0001-fixture.patch\n')
        self.executable('uname', '#!/bin/sh\nprintf "%s\\n" "$TEST_PLATFORM"\n')
        self.executable('nvcc', '#!/bin/sh\nexit 0\n')
        self.executable('zig', '''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
if sys.argv[1:] == ['version']:
    print('0.17.0')
else:
    Path('build-args.json').write_text(json.dumps(sys.argv[1:]))
    for target in ('native', 'drama3-generate', 'drama3-voice'):
        if target in sys.argv[1:]:
            name = 'native/bin/tensorfold-native' if target == 'native' else 'bin/' + target
            path = Path('zig-out') / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(target)
''')

    def git(self, *arguments):
        return subprocess.run(['git', *arguments], cwd=self.source, env=self.env,
                              capture_output=True, text=True, check=True, timeout=20)

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def build(self, platform, weights, name='build'):
        target = self.root / name
        result = subprocess.run(
            ['bash', str(self.repo / 'scripts/build-engine.sh'), '--weights', weights, str(target)],
            env=dict(self.env, TEST_PLATFORM=platform, ZIG=str(self.bin / 'zig'), NVCC=str(self.bin / 'nvcc')),
            capture_output=True, text=True, timeout=30)
        return result, target

    def test_each_platform_applies_only_its_selected_patch_series(self):
        for platform, weights, selected in [('Linux', 'nvfp4', 'cuda'), ('Darwin', 'nvfp4', 'metal'),
                                             ('Linux', 'full', 'full'), ('Darwin', 'full', 'full')]:
            with self.subTest(platform=platform, weights=weights):
                result, target = self.build(platform, weights, platform + weights)
                self.assertEqual(result.returncode, 0, result.stderr)
                for mode in self.series:
                    self.assertEqual((target / f'applied-{mode}').exists(), mode == selected)
                arguments = json.loads((target / 'build-args.json').read_text())
                expected = ['native'] if selected == 'cuda' else ['native', 'drama3-generate']
                if selected == 'metal':
                    expected.append('drama3-voice')
                self.assertEqual([a for a in arguments if a in ('native', 'drama3-generate', 'drama3-voice')], expected)
                self.assertEqual(any(a.startswith('-Dnvcc=') for a in arguments), platform == 'Linux')

    def test_corrupt_cuda_patch_is_refused_before_checkout_or_build(self):
        path = self.repo / self.series['cuda'] / 'patches/0001-fixture.patch'
        path.write_bytes(path.read_bytes() + b'changed\n')
        result, target = self.build('Linux', 'nvfp4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('patch checksum verification failed', result.stderr)
        self.assertFalse(target.exists())

    def test_missing_cuda_series_is_refused(self):
        (self.repo / self.series['cuda'] / 'patches/0001-fixture.patch').unlink()
        result, target = self.build('Linux', 'nvfp4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('has no *.patch files', result.stderr)
        self.assertFalse(target.exists())

    def test_manifest_covers_exactly_the_selected_patches(self):
        directory = self.repo / self.series['cuda'] / 'patches'
        manifest = directory / 'SHA256SUMS'
        original = manifest.read_text()
        unlisted = directory / '0002-unlisted.patch'
        unlisted.write_bytes((self.repo / self.series['metal'] / 'patches/0001-fixture.patch').read_bytes())
        result, target = self.build('Linux', 'nvfp4')
        self.assertNotEqual(result.returncode, 0, 'an unlisted patch must never be applied')
        self.assertFalse(target.exists())
        unlisted.unlink()
        for content in (original + original, original.replace('0001-fixture.patch', 'missing.patch'),
                        original + hashlib.sha256(b'notes').hexdigest() + '  README.md\n'):
            with self.subTest(manifest=content):
                manifest.write_text(content)
                result, target = self.build('Linux', 'nvfp4')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())

    def test_rebuild_reapplies_the_same_cuda_patch(self):
        first, target = self.build('Linux', 'nvfp4')
        self.assertEqual(first.returncode, 0, first.stderr)
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        second, _ = self.build('Linux', 'nvfp4')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip(), head)
        self.assertEqual((target / 'applied-cuda').read_text(), 'cuda\n')

    def test_modified_checkout_is_preserved_and_refused(self):
        first, target = self.build('Linux', 'nvfp4')
        self.assertEqual(first.returncode, 0, first.stderr)
        (target / '.zig-version').write_text('operator edit\n')
        second, _ = self.build('Linux', 'nvfp4')
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('local changes', second.stderr)
        self.assertEqual((target / '.zig-version').read_text(), 'operator edit\n')


if __name__ == '__main__':
    unittest.main()
