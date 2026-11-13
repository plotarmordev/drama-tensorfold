#!/usr/bin/env python3
"""Speaks the card 37 corpus on a GB10 through the staged EXL3 engine and reports the job's numbers.

  speak.py BIN CHECKPOINT VOICES OUT

Two arms, `ref` (the eq-3ref voice, guidance γ 1.5 batched) and `builtin`, each at salts 0 and 1 — the engine's
speech seed is its prompt keyed by `TENSORFOLD_SEED_SALT`, so one server per arm and salt — the 24 texts of
bench/corpus/drama-bench-en-v1.jsonl one at a time under `--max-frames 1024`, card 37's protocol. Per arm it
reports the load time (process start to `/health` 200), every take's wall seconds, and the peak device memory
from `GET /memory`. WAVs and `summary.json` land in OUT; the WAVs of both salts make the listening page's 96
takes should the Desk want them.
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
    'ref': lambda voices: ['--voices', str(voices), '--default-voice', 'eq-3ref', '--cfg-gamma', '1.5', '--cfg-batched'],
    'builtin': lambda voices: [],
}
SALTS = (0, 1)


def wait_ready(server, log, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2)
            return
        except OSError:
            if server.poll() is not None:
                raise RuntimeError(f'server exited during load: {log}')
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def say(text):
    """One take through the running server: its WAV bytes and finish reason."""
    body = json.dumps({'model': 'fishaudio/drama-3', 'input': text, 'voice': 'default',
                       'response_format': 'wav'}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/audio/speech', body,
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as reply:
        return reply.read(), reply.headers.get('X-Finish-Reason')


def speak_arm(binary, checkpoint, arm, salt, voices, out):
    """One arm and salt: a server seeded by TENSORFOLD_SEED_SALT, every corpus text spoken alone, timed, then the
    first text again, which must give the same bytes: the guide and the conditional stream share no scratch."""
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'server.log'
    with log.open('w') as sink:
        server = subprocess.Popen(
            [str(binary), 'serve', str(checkpoint), '--host', '127.0.0.1', '--port', str(PORT),
             '--name', 'fishaudio/drama-3', '--max-frames', '1024', *ARMS[arm](voices)],
            stdout=sink, stderr=subprocess.STDOUT,
            env={**os.environ, 'TENSORFOLD_SEED_SALT': str(salt), 'TENSORFOLD_NO_UPDATE_CHECK': '1'})
    started = time.monotonic()
    try:
        wait_ready(server, log)
        load_s = time.monotonic() - started
        takes = []
        for item in CORPUS:
            before = time.monotonic()
            wav, finish = say(item['text'])
            takes.append({'arm': arm, 'salt': salt, 'item': item['id'],
                          'seconds': round(time.monotonic() - before, 3), 'bytes': len(wav), 'finish': finish})
            (out / f"{item['id']}.wav").write_bytes(wav)
        repeat_identical = say(CORPUS[0]['text'])[0] == (out / f"{CORPUS[0]['id']}.wav").read_bytes()
        peak = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/memory', timeout=30))
    finally:
        server.terminate()
        server.wait(60)
    return {'arm': arm, 'salt': salt, 'load_s': round(load_s, 3), 'peak_bytes': peak['device']['peak_bytes'],
            'repeat_identical': repeat_identical, 'takes': takes}


def summary(runs):
    """The report's rows: per arm the load, peak and per-take seconds, the failed (capped) takes, and whether every
    salt's repeated first take matched its bytes."""
    rows = []
    for arm in ARMS:
        arm_runs = [run for run in runs if run['arm'] == arm]
        if not arm_runs:
            continue
        seconds = sorted(take['seconds'] for run in arm_runs for take in run['takes'])
        rows.append({'arm': arm,
                     'load_s': round(sum(run['load_s'] for run in arm_runs) / len(arm_runs), 3),
                     'peak_gib': round(max(run['peak_bytes'] for run in arm_runs) / 2**30, 2),
                     'takes': len(seconds), 'seconds_mean': round(sum(seconds) / len(seconds), 3),
                     'seconds_p50': seconds[len(seconds) // 2],
                     'seconds_total': round(sum(seconds), 1),
                     'capped': sum(1 for run in arm_runs for take in run['takes'] if take['finish'] == 'length'),
                     'repeat_identical': all(run['repeat_identical'] for run in arm_runs)})
    return rows


def main():
    binary, checkpoint, voices, out = map(Path, sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for arm in ARMS:
        for salt in SALTS:
            runs.append(speak_arm(binary, checkpoint, arm, salt, voices, out / arm / f's{salt}'))
            # a window that closes mid-run still leaves the arms it finished
            (out / 'summary.json').write_text(json.dumps({'runs': runs, 'report': summary(runs)}, indent=1) + '\n')
    for row in summary(runs):
        print(row, flush=True)


if __name__ == '__main__':
    main()
