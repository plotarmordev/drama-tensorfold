#!/usr/bin/env python3
"""The int3 build's and t5900's speed on a GB10, one engine binary: the Hugging Face card's Spark rows.

  spark_speed.py BIN INT3 T5900 VOICES OUT

Per build and voice (built-in, and the eq-3ref reference with guidance γ 1.5 batched), one server with
quant-15gb's arguments, then `bench/corpus_capture.py --passes 2`: the 24 corpus texts one request at a time, pass 0
warming the engine. Reports per arm the warm pass's real-time factor and whole reply at p50 nearest-rank (quant-15gb's
convention), process start to `/health` 200, and the peak device memory `/health` reported. OUT/speed.json holds the
rows, the captures their per-request records.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PORT = 18797
BASE = f'http://127.0.0.1:{PORT}'
SERVE = ('--context', '8192', '--parallel', '1', '--no-drafts', '--prompt-cache-gib', '0', '--keep-warm', '0',
         '--max-frames', '1024', '--cfg-gamma', '1.5', '--cfg-batched')
VOICES = {'builtin': lambda voices: (), 'ref': lambda voices: ('--voices', str(voices), '--default-voice', 'eq-3ref')}


def health():
    with urllib.request.urlopen(BASE + '/health', timeout=2) as reply:
        return json.load(reply)


def serve(binary, ckpt, voice_args, log):
    """Starts a server and waits for /health: the process and its start-to-ready seconds."""
    server = subprocess.Popen([str(binary), 'serve', str(ckpt), '--host', '127.0.0.1', '--port', str(PORT),
                               '--name', 'fishaudio/drama-3', *SERVE, *voice_args],
                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'TENSORFOLD_NO_UPDATE_CHECK': '1'})
    started = time.monotonic()
    while True:
        try:
            health()
            return server, time.monotonic() - started
        except OSError:
            if server.poll() is not None or time.monotonic() - started > 900:
                server.kill()
                raise RuntimeError(f'server not ready: {log.name}')
            time.sleep(0.5)


def p50(values):
    """Nearest rank: the sorted value at rank ceil(n/2), never interpolated."""
    ranked = sorted(values)
    return ranked[(len(ranked) + 1) // 2 - 1]


def measure(binary, build, ckpt, voice, voices, out):
    run = out / f'{build}-{voice}'
    run.parent.mkdir(parents=True, exist_ok=True)
    with open(f'{run}.log', 'w') as log:
        server, ready_s = serve(binary, ckpt, VOICES[voice](voices), log)
        try:
            subprocess.run([sys.executable, str(REPO / 'bench/corpus_capture.py'), BASE, str(run), '--passes', '2'],
                           check=True)
            peak = health()['memory']['peak']
        finally:
            server.terminate()
            server.wait(120)
    warm = [row for row in json.loads((run / 'capture.json').read_text())['rows'] if row['pass'] == 1]
    if any(row['status'] != 200 for row in warm):
        raise RuntimeError(f'{run}: a warm request failed')
    return {'build': build, 'voice': voice, 'ready_s': round(ready_s, 1), 'peak_gib': round(peak / 2**30, 2),
            'rtf_warm_p50': round(p50(row['rtf'] for row in warm), 3),
            'reply_warm_p50_s': round(p50(row['seconds'] for row in warm), 2),
            'audio_warm_s': round(sum(row['audio_s'] for row in warm), 1)}


def main():
    binary, int3, t5900, voices, out = map(Path, sys.argv[1:6])
    rows = []
    for build, ckpt in (('trim-bf16-int3', int3), ('t5900', t5900)):
        for voice in VOICES:
            rows.append(measure(binary, build, ckpt, voice, voices, out))
            print(json.dumps(rows[-1]), flush=True)
            # a window that closes mid-run still leaves the arms it finished
            (out / 'speed.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
