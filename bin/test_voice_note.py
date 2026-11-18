import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

VOICE_NOTE = Path(__file__).with_name('voice-note')
M4A = b'\x00\x00\x00\x1cftypM4A \x00\x00\x02\x00' + b'\x00' * 64
TEXT = ('Capri here. ' * 34)[:400]


def run(reply=b'', exit_code=0):
    """voice-note against a fake ssh that prints `reply` and exits `exit_code`; returns the result, every call and
    the JSON each call was sent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'reply').write_bytes(reply)
        (tmp / 'ssh').write_text('#!/usr/bin/env bash\n'
                                 'echo "$*" >> "$0.calls"\n'
                                 'cat >> "$0.stdin"\n'
                                 'cat "$(dirname "$0")/reply"\n'
                                 f'exit {exit_code}\n')
        (tmp / 'ssh').chmod(0o755)
        env = {**os.environ, 'PATH': f'{tmp}:{os.environ["PATH"]}', 'VOICE_NOTE_DIR': str(tmp / 'notes')}
        done = subprocess.run([str(VOICE_NOTE), TEXT], env=env, capture_output=True, text=True, timeout=30)
        note = Path(done.stdout.strip()).read_bytes() if done.returncode == 0 else None
        sent = [json.loads(line) for line in (tmp / 'ssh.stdin').read_text().splitlines()]
        return done, note, (tmp / 'ssh.calls').read_text().splitlines(), sent


class VoiceNoteTests(unittest.TestCase):
    def test_a_400_character_note_comes_back_as_an_m4a(self):
        done, note, calls, sent = run(base64.b64encode(M4A) + b'\n')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.strip().endswith('.m4a'))
        self.assertEqual(note, M4A)
        self.assertEqual(sent, [{'text': TEXT}])
        self.assertEqual(calls, ['-o BatchMode=yes -o ConnectTimeout=10 top-menu voice'])

    def test_a_timeout_says_timed_out_not_down_and_asks_once(self):
        done, _, calls, _ = run(exit_code=124)
        self.assertEqual(done.returncode, 124)
        self.assertIn('timed out', done.stderr)
        self.assertNotIn('is down', done.stderr)
        self.assertEqual(len(calls), 1)

    def test_any_other_failure_says_the_voice_is_down_and_asks_once(self):
        done, _, calls, _ = run(b'voice failed: HTTP 502\n', exit_code=70)
        self.assertEqual(done.returncode, 1)
        self.assertIn('is down', done.stderr)
        self.assertEqual(len(calls), 1)

    def test_a_reply_that_is_not_an_m4a_says_the_voice_is_down(self):
        done, _, _, _ = run(base64.b64encode(b'not audio') + b'\n')
        self.assertEqual(done.returncode, 1)
        self.assertIn('is down', done.stderr)


if __name__ == '__main__':
    unittest.main()
