#!/usr/bin/env python3
"""Whole-PCM replies for every corpus item against a running server (stdlib only), for byte comparison and timing.

Usage: corpus_capture.py BASE OUT_DIR [--passes N] [--voice NAME]. OUT_DIR must not exist. Writes one body per item and pass
(pass-N/ITEM.pcm) and capture.json: per request status, bytes, sha256, seconds, audio seconds and RTF. A 503 is
retried for up to a second: the server frees a reply's output slot only after it has sent it, so the next request can
race it. Any other non-200 reply keeps its status and error text; it is a recorded outcome, not an exception. Pass 0
also warms the engine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

CORPUS = Path(__file__).resolve().parent / 'corpus/drama-bench-en-v1.jsonl'
SAMPLE_RATE = 44100
BUSY_RETRIES = 10


def request(base, text, voice=None):
    payload = {'model': 'fishaudio/drama-3', 'input': text, 'response_format': 'pcm'}
    if voice is not None:
        payload['voice'] = voice
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + '/v1/audio/speech', data=body, method='POST',
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as reply:
            return reply.status, reply.read(), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('out', type=Path)
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--voice', help='Explicit catalog voice for reference-cache identity gates')
    args = parser.parse_args()
    items = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for p in range(args.passes):
        directory = args.out / f'pass-{p}'
        directory.mkdir()
        for item in items:
            status, body, seconds = request(args.base, item['text'], args.voice)
            for _ in range(BUSY_RETRIES):
                if status != 503:
                    break
                time.sleep(0.1)
                status, body, seconds = request(args.base, item['text'], args.voice)
            row = {'pass': p, 'item': item['id'], 'status': status, 'seconds': seconds, 'bytes': len(body),
                   'sha256': hashlib.sha256(body).hexdigest()}
            if status == 200:
                (directory / f"{item['id']}.pcm").write_bytes(body)
                row['audio_s'] = len(body) / 2 / SAMPLE_RATE
                row['rtf'] = seconds / row['audio_s'] if row['audio_s'] else None
            else:
                row['error'] = body.decode('utf-8', 'replace')[:500]
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ('pass', 'item', 'status', 'seconds', 'rtf')}), flush=True)
    (args.out / 'capture.json').write_text(json.dumps(
        {'corpus': str(CORPUS), 'passes': args.passes, 'voice': args.voice, 'rows': rows}, indent=1) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
