#!/usr/bin/env python3
"""The default build's one spoken line on a GB10, without systemd: gb10-default.sh's last step.

  speak_once.py BIN CHECKPOINT

Checks CHECKPOINT against the int3 build's pins (recipe/checksums/trim-bf16-int3.sha256), starts BIN's server on it
with the launchers' arguments and the build's guidance, waits for /health, streams one line as PCM and prints the
weights revision, the HTTP status, the seconds of audio and the time to first audio. Any failure exits non-zero with
the error, the server's own output above it.
"""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe'))
import weight_selection  # noqa: E402

WEIGHTS = 'int3-5900'
PORT = 18798
TEXT = 'The train leaves at seven fifteen.'


def wait_ready(server, timeout):
    """Polls /health until it answers 200; fails once the server exits or `timeout` seconds pass."""
    started = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as reply:
                if reply.status == 200:
                    if server.poll() is not None:
                        raise RuntimeError(f'the server exited with status {server.returncode} after /health answered')
                    return
        except OSError:
            pass
        if server.poll() is not None:
            raise RuntimeError(f'the server exited with status {server.returncode} before /health answered')
        if time.monotonic() - started > timeout:
            raise RuntimeError(f'/health did not answer within {timeout:.0f} s')
        time.sleep(0.5)


def speak():
    """One streamed PCM request: (HTTP status, seconds of audio, seconds to the first audio bytes)."""
    connection = http.client.HTTPConnection('127.0.0.1', PORT, timeout=600)
    try:
        started = time.monotonic()
        connection.request('POST', '/v1/audio/speech', headers={'Content-Type': 'application/json'},
                           body=json.dumps({'model': 'fishaudio/drama-3', 'input': TEXT, 'response_format': 'pcm',
                                            'stream': True}).encode())
        reply = connection.getresponse()
        if reply.status != 200:
            raise RuntimeError(f'HTTP {reply.status}: {reply.read().decode(errors="replace")}')
        audio_bytes, first_audio = 0, None
        while block := reply.read1(1 << 16):
            if first_audio is None:
                first_audio = time.monotonic() - started
            audio_bytes += len(block)
    finally:
        connection.close()
    if not audio_bytes:
        raise RuntimeError('HTTP 200 with no audio')
    return reply.status, audio_bytes / 2 / int(reply.headers.get('X-Sample-Rate', 44100)), first_audio


def serve_and_speak(command, ready_timeout=900):
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', PORT))
        except OSError as error:
            raise RuntimeError(f'127.0.0.1:{PORT} is unavailable: {error}') from error
    server = subprocess.Popen(command, env={**os.environ, 'TENSORFOLD_NO_UPDATE_CHECK': '1'})
    try:
        wait_ready(server, ready_timeout)
        return speak()
    finally:
        server.terminate()
        server.wait(120)


def main():
    binary, checkpoint = map(Path, sys.argv[1:3])
    provenance = weight_selection.prepare_checkpoint(WEIGHTS, checkpoint)
    print(f'weights revision: {provenance["checkpoint_source"]["revision"]}', flush=True)
    status, audio_s, first_audio_s = serve_and_speak(weight_selection.server_command(
        binary, checkpoint, '127.0.0.1', PORT, weight_selection.serve_arguments(WEIGHTS, [])))
    print(f'HTTP status: {status}\naudio: {audio_s:.2f} s\ntime to first audio: {first_audio_s:.3f} s', flush=True)


if __name__ == '__main__':
    main()
