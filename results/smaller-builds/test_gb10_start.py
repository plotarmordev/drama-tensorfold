import os
import subprocess
import tempfile
import unittest
from pathlib import Path

START = Path(__file__).with_name('gb10-start.sh')
SHA = 'a' * 40
NO_WINDOW = 'refused: a GPU job runs on middle or bottom inside a window: `deepseek pause [hours]` first'
BUSY = 'busy: a test or another job is being set up; retry in a few minutes'
STARTED = '04:30:00 started bottom-0123456789ab: plotarmordev/drama-tensorfold@aaaaaaaaaaaa ./run.sh on bottom'


def run(*replies):
    """gb10-start.sh against a fake ssh that gives these TOP menu replies in turn; returns the result and every call."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, reply in enumerate(replies):
            (tmp / f'reply-{i}').write_text(reply)
        (tmp / 'ssh').write_text('#!/usr/bin/env bash\n'
                                 'n=$(wc -l < "$0.calls" 2>/dev/null || echo 0)\n'
                                 'echo "$*" >> "$0.calls"\n'
                                 'cat "$(dirname "$0")/reply-$((n))"\n')
        (tmp / 'ssh').chmod(0o755)
        env = {**os.environ, 'PATH': f'{tmp}:{os.environ["PATH"]}', 'POLL': '0'}
        done = subprocess.run([str(START), SHA, 'results/smaller-builds/gb10-int3.sh'], env=env,
                              capture_output=True, text=True, timeout=30)
        return done, (tmp / 'ssh.calls').read_text().splitlines()


class StartTests(unittest.TestCase):
    def test_waits_for_a_window_without_stopping_deepseek(self):
        done, calls = run(NO_WINDOW, BUSY, NO_WINDOW, STARTED)
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stdout.strip(), STARTED)
        self.assertEqual(calls, [f'top-menu job start gpu bottom plotarmordev/drama-tensorfold {SHA} '
                                 'results/smaller-builds/gb10-int3.sh'] * 4)

    def test_any_other_refusal_ends_the_wait(self):
        done, calls = run('refused: sha not found on origin')
        self.assertEqual(done.returncode, 1)
        self.assertIn('sha not found', done.stderr)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
