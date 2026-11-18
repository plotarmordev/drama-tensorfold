import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

VOICE_NOTE = Path(__file__).with_name('voice-note')
M4A = b'\x00\x00\x00\x1cftypM4A \x00\x00\x02\x00' + b'\x00' * 64
# A natural passage, as Capri's notes are; the repeated phrase that used to fail has its own live test.
TEXT = ('Hi, it is Capri. I listened to the new take this morning and it finally sounds like one breath from start '
        'to end. The join that used to jump in pitch is gone, and the pauses land where I would take them. Tonight '
        'I want to try the longer passage from the second chapter, the one with the storm. If it holds up, we can '
        'send it to the others tomorrow and see what they all think of it. Talk soon, loves.')


def run(reply=b'', exit_code=0, notes=1, busy=0, folder_mode=None):
    """`notes` voice-notes at once against a fake ssh that answers busy (exit 75) to its first `busy` calls, then
    prints `reply` and exits `exit_code`, into a notes folder made beforehand with `folder_mode` (default: none yet);
    returns each result, the notes folder (name -> (bytes, mode)) and its mode, every call and the JSON each call was
    sent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'reply').write_bytes(reply)
        (tmp / 'ssh').write_text('#!/usr/bin/env bash\n'
                                 'echo "$*" >> "$0.calls"\n'
                                 'cat >> "$0.stdin"\n'
                                 f'if [ "$(wc -l < "$0.calls")" -le {busy} ]; then\n'
                                 '  echo "busy: another voice call is running; retry in a few seconds"; exit 75\n'
                                 'fi\n'
                                 'cat "$(dirname "$0")/reply"\n'
                                 f'exit {exit_code}\n')
        (tmp / 'ssh').chmod(0o755)
        if folder_mode is not None:
            (tmp / 'notes').mkdir()
            (tmp / 'notes').chmod(folder_mode)
        env = {**os.environ, 'PATH': f'{tmp}:{os.environ["PATH"]}', 'VOICE_NOTE_DIR': str(tmp / 'notes')}
        procs = [subprocess.Popen([str(VOICE_NOTE), TEXT], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for _ in range(notes)]
        outputs = [p.communicate(timeout=30) for p in procs]
        done = [subprocess.CompletedProcess(p.args, p.returncode, *out) for p, out in zip(procs, outputs)]
        folder = {f.name: (f.read_bytes(), f.stat().st_mode & 0o777) for f in (tmp / 'notes').iterdir()}
        sent = [json.loads(line) for line in (tmp / 'ssh.stdin').read_text().splitlines()]
        return (done, folder, (tmp / 'notes').stat().st_mode & 0o777, (tmp / 'ssh.calls').read_text().splitlines(),
                sent)


class VoiceNoteTests(unittest.TestCase):
    def test_a_400_character_note_comes_back_as_a_private_m4a(self):
        [done], folder, folder_mode, calls, sent = run(base64.b64encode(M4A) + b'\n')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(folder, {Path(done.stdout.strip()).name: (M4A, 0o600)})
        self.assertTrue(done.stdout.strip().endswith('.m4a'))
        self.assertEqual(folder_mode, 0o700)
        self.assertEqual(sent, [{'text': TEXT}])
        self.assertEqual(calls, ['-o BatchMode=yes -o ConnectTimeout=10 top-menu voice'])

    def test_an_existing_open_notes_folder_is_made_private(self):
        [done], _, folder_mode, _, _ = run(base64.b64encode(M4A) + b'\n', folder_mode=0o755)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(folder_mode, 0o700)

    def test_notes_made_in_the_same_second_each_keep_their_own_file(self):
        done, folder, _, _, _ = run(base64.b64encode(M4A) + b'\n', notes=4)
        self.assertEqual([d.returncode for d in done], [0] * 4)
        self.assertEqual(sorted(folder), sorted(Path(d.stdout.strip()).name for d in done))
        self.assertEqual(len(folder), 4)

    def test_a_note_turned_away_as_busy_waits_for_its_turn_with_the_same_text(self):
        [done], folder, _, calls, sent = run(base64.b64encode(M4A) + b'\n', busy=1)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(list(folder.values()), [(M4A, 0o600)])
        self.assertEqual(sent, [{'text': TEXT}] * 2)
        self.assertEqual(len(calls), 2)

    def test_a_timeout_says_timed_out_not_down_asks_once_and_leaves_nothing(self):
        [done], folder, _, calls, _ = run(exit_code=124)
        self.assertEqual(done.returncode, 124)
        self.assertIn('timed out', done.stderr)
        self.assertNotIn('is down', done.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(folder, {})

    def test_any_other_failure_says_the_voice_is_down_and_asks_once(self):
        [done], folder, _, calls, _ = run(b'voice failed: HTTP 502\n', exit_code=70)
        self.assertEqual(done.returncode, 1)
        self.assertIn('is down', done.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(folder, {})

    def test_a_reply_that_is_not_an_m4a_says_the_voice_is_down(self):
        [done], folder, _, _, _ = run(base64.b64encode(b'not audio') + b'\n')
        self.assertEqual(done.returncode, 1)
        self.assertIn('is down', done.stderr)
        self.assertEqual(folder, {})


@unittest.skipUnless(os.environ.get('VOICE_NOTE_LIVE') == '1', 'explicit live call to TOP voice; not a unit test')
class LiveVoiceNoteTests(unittest.TestCase):
    def note(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run([str(VOICE_NOTE), text], env={**os.environ, 'VOICE_NOTE_DIR': tmp},
                                  capture_output=True, text=True, timeout=180)
            return done, done.returncode == 0 and Path(done.stdout.strip()).read_bytes()[4:8] == b'ftyp'

    def test_a_400_character_note_comes_back_as_an_m4a_in_time(self):
        done, m4a = self.note(TEXT)
        self.assertTrue(m4a, done.stderr)

    def test_the_repeated_phrase_that_used_to_fail_is_an_m4a_or_a_timeout_never_voice_down(self):
        # The model draws this stress text slowly, near TOP's 57 s cap (set in TOP's voice shim, not here).
        done, m4a = self.note('Capri here. ' * 33)
        if not m4a:
            self.assertEqual(done.returncode, 124, done.stderr)
            self.assertIn('timed out', done.stderr)


if __name__ == '__main__':
    unittest.main()
