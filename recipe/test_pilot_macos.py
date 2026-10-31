#!/usr/bin/env python3
"""Pure unit tests, plus explicitly enabled macOS CPU lifecycle smoke tests.

Unit: python3 -m unittest discover -s recipe -p test_pilot_macos.py
Smoke: PILOT_MACOS_INTEGRATION=1 python3 -m unittest discover -s recipe -p test_pilot_macos.py

The smoke cases use temporary dummy weights, a harmless loopback HTTP child,
controlled VM observations, and only their own finite-lived subprocesses. They
never invoke a model, contact an external endpoint, or use the operator's lock.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pilot_macos as pilot


def snapshot(size=16384, free=10, inactive=20):
    return (f'Mach Virtual Memory Statistics: (page size of {size} bytes)\n'
            f'Pages free: {free}.\nPages inactive: {inactive}.\n'
            'Pages speculative: 100.\nPages purgeable: 200.\n')


class MemorySnapshotTests(unittest.TestCase):
    def test_counts_only_free_and_inactive_at_each_page_size(self):
        for size in (4096, 16384):
            with self.subTest(size=size):
                self.assertEqual(pilot.parse_memory_available(snapshot(size)), 30 * size)
        self.assertEqual(pilot.parse_memory_available(snapshot(free=0, inactive=0)), 0)

    def test_missing_duplicate_or_invalid_counts_fail_closed(self):
        valid = snapshot()
        invalid = [
            valid.replace('page size of 16384 bytes', 'unknown page size'),
            snapshot(size=0),
            valid + 'page size of 4096 bytes\n',
            valid.replace('Pages free: 10.\n', ''),
            valid.replace('Pages inactive: 20.\n', ''),
            snapshot(free=-1), snapshot(inactive='unknown'),
            valid + 'Pages free: 30.\n',
            valid.replace('Pages inactive: 20.', 'Pages inactive: 2.5.'),
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                pilot.parse_memory_available(text)

    def test_one_snapshot_supplies_both_counts_and_page_size(self):
        observations = [subprocess.CompletedProcess([], 0, snapshot(4096)),
                        subprocess.CompletedProcess([], 0, snapshot(16384, 1000, 2000))]
        with patch.object(pilot.subprocess, 'run', side_effect=observations) as observe:
            self.assertEqual(pilot.memory_available(), 30 * 4096)
            self.assertEqual(observe.call_count, 1)
            self.assertGreater(observe.call_args.kwargs['timeout'], 0)

    def test_observation_error_is_not_available_memory(self):
        for error in (subprocess.CalledProcessError(7, 'vm_stat'),
                      subprocess.TimeoutExpired('vm_stat', 2)):
            with self.subTest(error=error):
                with patch.object(pilot.subprocess, 'run', side_effect=error):
                    with self.assertRaises(type(error)):
                        pilot.memory_available()

    def test_nonfinite_and_negative_values_are_refused(self):
        for value in (float('nan'), float('inf'), float('-inf'), -0.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.finite_nonnegative(value)
        for value in (0, 0.01, 34):
            self.assertEqual(pilot.finite_nonnegative(value), value)


class HealthDeadlineTests(unittest.TestCase):
    def test_partial_status_line_respects_one_total_deadline(self):
        with patch.object(pilot.socket, 'socket') as factory, \
                patch.object(pilot.time, 'monotonic', side_effect=(0, 0.1, 0.2, 0.8, 1.0)):
            connection = factory.return_value.__enter__.return_value
            connection.recv.side_effect = (b'H', b'T')
            with self.assertRaises(TimeoutError):
                pilot.health_ready('127.0.0.1', 1234, 1)

    def test_only_a_complete_success_status_is_ready(self):
        for chunks, expected in (((b'HTTP/1.1 2', b'00 OK\r\n'), True),
                                 ((b'HTTP/1.0 503 Unavailable\r\n',), False),
                                 ((b'not HTTP\r\n',), False),
                                 ((b'',), False)):
            with self.subTest(chunks=chunks):
                with patch.object(pilot.socket, 'socket') as factory, \
                        patch.object(pilot.time, 'monotonic', return_value=0):
                    factory.return_value.__enter__.return_value.recv.side_effect = chunks
                    self.assertEqual(pilot.health_ready('127.0.0.1', 1234, 1), expected)


# Kept separate from unit coverage: these cases deliberately exercise real process
# groups, file locks, signals, and loopback readiness with bounded smoke deadlines.
RUNNER = r'''
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
control = Path(sys.argv.pop(1))
import pilot_macos as pilot
original_run = subprocess.run

def observe(command, **kwargs):
    if command[0] == '/usr/bin/vm_stat':
        mode = (control / 'memory-mode').read_text()
        if mode == 'error':
            return original_run([sys.executable, '-c', 'raise SystemExit(7)'], **kwargs)
        if mode == 'timeout':
            return original_run([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        pages = int(mode) * (1024**3 // 16384)
        text = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                f'Pages free: {pages}.\nPages inactive: 0.\n'
                'Pages speculative: 99999999.\nPages purgeable: 99999999.\n')
        return subprocess.CompletedProcess(command, 0, text)
    if command[0] == '/bin/ps' and (control / 'process-error').exists():
        return original_run([sys.executable, '-c', 'raise SystemExit(9)'], **kwargs)
    return original_run(command, **kwargs)

pilot.subprocess.run = observe
pilot.TERM_TIMEOUT = 1.0
pilot.KILL_TIMEOUT = 2.0
raise SystemExit(pilot.main())
'''

SERVER = r'''
import http.server
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path.cwd()
control = Path(__file__).parent
mode = (control / 'server-mode').read_text()
port = int(sys.argv[sys.argv.index('--port') + 1])
host = sys.argv[sys.argv.index('--host') + 1]
deadline = time.monotonic() + 20

if mode in ('descendant', 'early-descendant'):
    child = os.fork()
    if child == 0:
        def ignore_term(signum, frame):
            (root / 'descendant-term').touch()
        signal.signal(signal.SIGTERM, ignore_term)
        (root / 'descendant.pid').write_text(str(os.getpid()))
        while time.monotonic() < deadline and not (control / 'stop-fixtures').exists():
            time.sleep(0.02)
        os._exit(0)
    # The fixture will not announce readiness before the child's handler exists.
    while not (root / 'descendant.pid').exists() and time.monotonic() < deadline:
        time.sleep(0.01)

(root / 'server.json').write_text(json.dumps({'pid': os.getpid(), 'host': host}))
if mode == 'early-descendant':
    os._exit(7)
if mode == 'ignore-term':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if mode == 'slow-health':
            (root / 'health-started').touch()
            while time.monotonic() < deadline and not (control / 'stop-fixtures').exists():
                try:
                    self.wfile.write(b'H')
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.2)
            return
        self.send_response(503 if mode == 'unready' else 200)
        self.send_header('Content-Length', '0')
        self.end_headers()
    def log_message(self, *args):
        pass

with http.server.HTTPServer((host, port), Health) as server:
    server.timeout = 0.05
    while time.monotonic() < deadline and not (control / 'stop-fixtures').exists():
        if (root / 'exit-now').exists():
            os._exit(int((root / 'exit-now').read_text()))
        server.handle_request()
'''


@unittest.skipUnless(sys.platform == 'darwin' and os.environ.get('PILOT_MACOS_INTEGRATION') == '1',
                     'explicit macOS subprocess smoke; not a unit test')
class MacLifecycleSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pilot-macos-smoke-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.checkpoint = self.base / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'tiny.safetensors').write_bytes(b'not a model')
        (self.checkpoint / 'codec.pth').write_bytes(b'not a codec')
        self.lock = self.base / 'gpu.lock'
        self.lock.touch()
        (self.base / 'memory-mode').write_text('80')
        (self.base / 'server-mode').write_text('normal')
        self.runner = self.base / 'run-pilot.py'
        self.runner.write_text(RUNNER)
        self.server = self.base / 'dummy-server'
        self.server.write_text(f'#!{sys.executable}\n' + SERVER)
        self.server.chmod(0o700)
        self.launches = []

    def tearDown(self):
        # Never signal a guessed/reaped PGID. Launchers are our direct unreaped
        # children; server fixtures also have their own absolute lifetime bound.
        (self.base / 'stop-fixtures').touch()
        for process, root, evidence, output, log in self.launches:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            log.close()
            for marker in ('server.json', 'descendant.pid'):
                path = root / marker
                if path.exists():
                    pid = (json.loads(path.read_text())['pid'] if marker.endswith('.json')
                           else int(path.read_text()))
                    self.wait_until(lambda: not self.is_live(pid), timeout=10)

    def wait_until(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        logs = '\n'.join(item[3].read_text() for item in self.launches)
        self.fail(f'Owned fixture deadline exceeded\n{logs}')

    @staticmethod
    def is_live(pid):
        result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='],
                                capture_output=True, text=True, timeout=2)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        return any(not state.startswith('Z') for state in result.stdout.split())

    def start(self, extra=(), *, root=None, evidence=None):
        index = len(self.launches)
        root = root or self.base / f'runtime-{index}'
        evidence = evidence or self.base / f'evidence-{index}'
        # Reserve only a loopback port for the fixture. The launcher still performs
        # its own availability check; no requests leave this host.
        with socket.socket() as port_socket:
            port_socket.bind(('127.0.0.1', 0))
            port = port_socket.getsockname()[1]
        output = self.base / f'launcher-{index}.log'
        log = output.open('w')
        command = [sys.executable, str(self.runner), str(Path(pilot.__file__).parent), str(self.base),
                   '--native-bin', str(self.server), '--runtime-dir', str(root),
                   '--checkpoint', str(self.checkpoint), '--gpu-lock', str(self.lock),
                   '--evidence-dir', str(evidence), '--port', str(port), '--ready-timeout', '5',
                   *extra]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        attempt = (process, root, evidence, output, log)
        self.launches.append(attempt)
        return attempt

    def ready(self, attempt):
        process, root, evidence, output, _ = attempt
        self.wait_until(lambda: (evidence / 'ready.json').exists() or process.poll() is not None)
        self.assertIsNone(process.poll(), output.read_text())
        self.assertTrue((evidence / 'ready.json').exists())
        return json.loads((root / 'server.json').read_text())

    def completed(self, attempt, expected):
        process, _, evidence, output, _ = attempt
        self.assertEqual(process.wait(timeout=12), expected, output.read_text())
        return json.loads((evidence / 'completion.json').read_text())

    def assert_lock_held(self):
        with self.lock.open('r+') as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def assert_lock_free(self):
        with self.lock.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def set_memory(self, mode):
        pending = self.base / 'memory-mode.next'
        pending.write_text(mode)
        pending.replace(self.base / 'memory-mode')

    @staticmethod
    def exit_server(root, status):
        pending = root / 'exit-now.next'
        pending.write_text(str(status))
        pending.replace(root / 'exit-now')

    def test_default_loopback_runtime_creation_and_explicit_stop_at_floor(self):
        self.set_memory('34')
        attempt = self.start(('--min-available-gib', '34'))
        server = self.ready(attempt)
        self.assertEqual(server['host'], '127.0.0.1')
        self.assertTrue((attempt[1] / 'hf-cache').is_dir())
        self.assertTrue((attempt[1] / 'runtime-cache').is_dir())
        self.assert_lock_held()
        attempt[0].terminate()
        result = self.completed(attempt, 0)
        self.assertTrue(result['group_settled'])
        self.assertEqual(result['stop_signal'], signal.SIGTERM)
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['cleanup_failures'], [])
        self.assertFalse(self.is_live(server['pid']))
        self.assert_lock_free()

    def test_external_admission_starts_below_the_floor_and_never_aborts(self):
        self.set_memory('20')
        attempt = self.start(('--external-admission',))
        server = self.ready(attempt)
        session = json.loads((attempt[2] / 'session.json').read_text())
        self.assertIsNone(session['required_memavailable_bytes'])
        self.assertIsNone(session['abort_threshold_bytes'])
        samples = (attempt[2] / 'memory.jsonl').read_text().count('\n')
        self.set_memory('1')
        self.wait_until(lambda: (attempt[2] / 'memory.jsonl').read_text().count('\n') >= samples + 5)
        self.assertIsNone(attempt[0].poll(), attempt[3].read_text())
        attempt[0].terminate()
        result = self.completed(attempt, 0)
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['stop_signal'], signal.SIGTERM)
        self.assertFalse(self.is_live(server['pid']))
        self.assert_lock_free()

    def test_admission_failure_invalidates_reused_readiness(self):
        evidence = self.base / 'reused'
        evidence.mkdir()
        for name in ('ready.json', 'session.json', 'completion.json'):
            (evidence / name).write_text('{"old": true}')
        self.set_memory('33')
        attempt = self.start(('--evidence-reuse',), evidence=evidence)
        result = self.completed(attempt, 1)
        self.assertFalse((evidence / 'ready.json').exists())
        self.assertFalse((evidence / 'session.json').exists())
        self.assertFalse(attempt[1].exists())
        self.assertFalse(result['ready'])
        self.assertIsNone(result['server_returncode'])
        self.assertIn('Need 34.00 GiB', '\n'.join(result['failures']))
        self.assert_lock_free()

    def test_invalid_admission_arguments_do_not_preserve_old_readiness(self):
        cases = [('--min-available-gib', '33.99'), ('--min-available-gib', 'nan'),
                 ('--min-available-gib', 'inf'), ('--reserve-gib', '-1'),
                 ('--reserve-gib', 'nan'), ('--reserve-gib', '1e308'),
                 ('--ready-timeout', 'inf'), ('--ready-timeout', '-1')]
        for index, (flag, value) in enumerate(cases):
            with self.subTest(flag=flag, value=value):
                evidence = self.base / f'invalid-{index}'
                evidence.mkdir()
                (evidence / 'ready.json').write_text('old readiness')
                attempt = self.start(('--evidence-reuse', f'{flag}={value}'), evidence=evidence)
                result = self.completed(attempt, 1)
                self.assertFalse((evidence / 'ready.json').exists())
                self.assertFalse(attempt[1].exists())
                self.assertTrue(result['failures'])

    def test_existing_evidence_is_refused_without_reuse(self):
        evidence = self.base / 'existing'
        evidence.mkdir()
        (evidence / 'ready.json').write_text('untouched')
        attempt = self.start(evidence=evidence)
        self.assertEqual(attempt[0].wait(timeout=8), 2, attempt[3].read_text())
        self.assertEqual((evidence / 'ready.json').read_text(), 'untouched')
        self.assertFalse(attempt[1].exists())

    def test_runtime_file_fails_before_server_spawn(self):
        root = self.base / 'not-a-directory'
        root.write_text('untouched')
        attempt = self.start(root=root)
        result = self.completed(attempt, 1)
        self.assertEqual(root.read_text(), 'untouched')
        self.assertIsNone(result['server_returncode'])

    def test_contender_neither_spawns_nor_clears_active_evidence(self):
        owner = self.start()
        self.ready(owner)
        ready = (owner[2] / 'ready.json').read_text()
        contender = self.start(('--evidence-reuse',), evidence=owner[2])
        self.assertEqual(contender[0].wait(timeout=8), 2, contender[3].read_text())
        self.assertFalse(contender[1].exists())
        self.assertEqual((owner[2] / 'ready.json').read_text(), ready)
        owner[0].terminate()
        self.completed(owner, 0)

    def test_any_unexpected_post_ready_exit_is_failure_with_original_status(self):
        for status in (0, 7):
            with self.subTest(status=status):
                attempt = self.start()
                self.ready(attempt)
                self.exit_server(attempt[1], status)
                result = self.completed(attempt, 1)
                self.assertEqual(result['server_returncode'], status)
                self.assertTrue(result['group_settled'])
                self.assertIn(f'status {status} after readiness', '\n'.join(result['failures']))
                self.assert_lock_free()

    def test_dead_leader_does_not_skip_live_descendant_or_release_lock_early(self):
        (self.base / 'server-mode').write_text('descendant')
        attempt = self.start()
        self.ready(attempt)
        descendant = int((attempt[1] / 'descendant.pid').read_text())
        self.exit_server(attempt[1], 0)
        self.wait_until(lambda: (attempt[1] / 'descendant-term').exists())
        self.assertTrue(self.is_live(descendant))
        self.assert_lock_held()
        result = self.completed(attempt, 1)
        self.assertEqual(result['server_returncode'], 0)
        self.assertTrue(result['group_settled'])
        self.assertFalse(self.is_live(descendant))
        self.assert_lock_free()

    def test_pre_ready_exit_also_settles_descendants(self):
        (self.base / 'server-mode').write_text('early-descendant')
        attempt = self.start()
        result = self.completed(attempt, 1)
        self.assertEqual(result['server_returncode'], 7)
        self.assertFalse(result['ready'])
        self.assertTrue(result['group_settled'])
        descendant = int((attempt[1] / 'descendant.pid').read_text())
        self.assertFalse(self.is_live(descendant))

    def test_term_resistant_server_is_killed_with_bounded_cleanup(self):
        (self.base / 'server-mode').write_text('ignore-term')
        attempt = self.start()
        self.ready(attempt)
        attempt[0].terminate()
        result = self.completed(attempt, 0)
        self.assertEqual(result['server_returncode'], -signal.SIGKILL)
        self.assertTrue(result['group_settled'])

    def test_readiness_deadline_stops_real_unready_child(self):
        (self.base / 'server-mode').write_text('unready')
        attempt = self.start(('--ready-timeout', '0.5'))
        result = self.completed(attempt, 1)
        self.assertFalse(result['ready'])
        self.assertTrue(result['group_settled'])
        self.assertIn('Readiness deadline exceeded', '\n'.join(result['failures']))

    def test_slow_health_cannot_stall_memory_monitoring(self):
        (self.base / 'server-mode').write_text('slow-health')
        attempt = self.start()
        self.wait_until(lambda: (attempt[1] / 'health-started').exists())
        self.set_memory('33')
        result = self.completed(attempt, 1)
        self.assertFalse(result['ready'])
        self.assertTrue(result['group_settled'])
        self.assertIn('MemoryError', '\n'.join(result['failures']))

    def test_watchdog_low_memory_error_and_timeout_fail_closed(self):
        for mode, reason in (('33', 'MemoryError'), ('error', 'CalledProcessError'),
                             ('timeout', 'TimeoutExpired')):
            with self.subTest(mode=mode):
                self.set_memory('80')
                attempt = self.start()
                server = self.ready(attempt)
                self.set_memory(mode)
                result = self.completed(attempt, 1)
                self.assertTrue(result['group_settled'])
                self.assertIn(reason, '\n'.join(result['failures']))
                if mode != '33':
                    self.assertIsNone(result['final_memavailable_bytes'])
                self.assertFalse(self.is_live(server['pid']))
                self.assert_lock_free()

    def test_unobservable_cleanup_is_reported_as_failure_not_success(self):
        attempt = self.start()
        server = self.ready(attempt)
        (self.base / 'process-error').touch()
        result = self.completed(attempt, 1)
        self.assertFalse(result['group_settled'])
        self.assertIsNone(result['server_returncode'])
        self.assertIn('CalledProcessError', '\n'.join(result['failures']))
        self.assertIn('confirmed settlement', '\n'.join(result['cleanup_failures']))
        self.wait_until(lambda: not self.is_live(server['pid']))

    def test_uncatchable_launcher_death_leaves_child_holding_shared_lock(self):
        owner = self.start()
        server = self.ready(owner)
        owner[0].kill()
        self.assertEqual(owner[0].wait(timeout=3), -signal.SIGKILL)
        self.assertTrue(self.is_live(server['pid']))
        self.assert_lock_held()
        contender = self.start(('--evidence-reuse',), evidence=owner[2])
        self.assertEqual(contender[0].wait(timeout=8), 2, contender[3].read_text())
        self.assertFalse(contender[1].exists())
        self.exit_server(owner[1], 0)
        self.wait_until(lambda: not self.is_live(server['pid']))
        self.assert_lock_free()


if __name__ == '__main__':
    unittest.main()
