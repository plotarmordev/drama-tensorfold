#!/usr/bin/env python3
"""Speaks the card 37 corpus on a GB10 through the staged EXL3 engine and reports the job's numbers.

  speak.py BIN CHECKPOINT VOICES OUT

Two arms, `ref` (the eq-3ref voice, guidance γ 1.5 batched) and `builtin`, salts 0 and 1 as request seeds, the 24
texts of bench/corpus/drama-bench-en-v1.jsonl one at a time under `--max-frames 1024` — card 37's protocol. Per arm
it reports the load time (process start to `/health` 200), every take's wall seconds, and the peak device memory
from `GET /memory`. WAVs, `takes.json` and `summary.json` land in OUT; the WAVs of both salts make the listening
page's 96 takes should the Desk want them.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORPUS = [json.loads(line) for line in (REPO / 'bench/corpus/drama-bench-en-v1.jsonl').read_text().splitlines()]
PORT = 18841
ARMS = {
    'ref': lambda voices: ['--voices', voices, '--default-voice', 'eq-3ref', '--cfg-gamma', '1.5', '--cfg-batched'],
    'builtin': lambda voices: [],
}
SALTS = (0, 1)


def wait_ready(timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def speak_arm(binary, checkpoint, arm, out):
    """One arm: a server per salt, every corpus text spoken alone, timed from send to last byte."""
    log = out / 'server.log'
    with log.open('w') as sink:
        server = subprocess.Popen(
            [str(binary), 'serve', str(checkpoint), '--host', '127.0.0.1', '--port', str(PORT),
             '--name', 'fishaudio/drama-3', '--max-frames', '1024', *ARMS[arm](out.parent / 'voices')],
            stdout=sink, stderr=subprocess.STDOUT,
            env={**os.environ, 'TENSORFOLD_NO_UPDATE_CHECK': '1'})
    started = time.monotonic()
    try:
        wait_ready()
        load_s = time.monotonic() - started
        takes = []
        for salt in SALTS:
            for item in CORPUS:
                body = json.dumps({'model': 'fishaudio/drama-3', 'input': item['text'], 'voice': 'default',
                                   'response_format': 'wav', 'seed': salt}).encode()
                request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/audio/speech', body,
                                                 {'Content-Type': 'application/json'})
                before = time.monotonic()
                with urllib.request.urlopen(request, timeout=1800) as reply:
                    wav = reply.read()
                    takes.append({'arm': arm, 'salt': salt, 'item': item['id'], 'seed': salt,
                                  'seconds': round(time.monotonic() - before, 3), 'bytes': len(wav),
                                  'finish': reply.headers.get('X-Finish-Reason')})
                (out / f"{item['id']}-s{salt}.wav").write_bytes(wav)
        peak = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/memory', timeout=30))
    finally:
        server.terminate()
        server.wait(60)
    return {'arm': arm, 'load_s': round(load_s, 3), 'peak_bytes': peak['device']['peak_bytes'], 'takes': takes}


def summary(runs):
    """The report's rows: per arm the load, peak and per-take seconds, and the failed (capped) takes."""
    rows = []
    for run in runs:
        seconds = sorted(t['seconds'] for t in run['takes'])
        rows.append({'arm': run['arm'], 'load_s': run['load_s'], 'peak_gib': round(run['peak_bytes'] / 2**30, 2),
                     'takes': len(seconds), 'seconds_mean': round(sum(seconds) / len(seconds), 3),
                     'seconds_p50': seconds[len(seconds) // 2],
                     'seconds_total': round(sum(seconds), 1),
                     'capped': sum(1 for t in run['takes'] if t['finish'] == 'length')})
    return rows


def estimate_window_seconds(ref_audio_s, builtin_audio_s, rtf_ref, rtf_builtin, loads=2, load_s=60):
    """The window the job asks for: every take at its audio length times the Spark's RTF for that arm, plus loads."""
    speaking = ref_audio_s * rtf_ref + builtin_audio_s * rtf_builtin
    return loads * load_s + speaking


def main():
    binary, checkpoint, voices, out = map(Path, sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / 'voices').symlink_to(voices)
    runs = [speak_arm(binary, checkpoint, arm, out / arm) for arm in ARMS]
    (out / 'summary.json').write_text(json.dumps({'runs': runs, 'report': summary(runs)}, indent=1) + '\n')
    for row in summary(runs):
        print(row, flush=True)


if __name__ == '__main__':
    main()
