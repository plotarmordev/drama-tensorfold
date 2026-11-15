#!/usr/bin/env python3
"""Serve Drama 3 on this Mac with the speech API of the GB10 host, from one config file.

Usage: python3 recipe/speech_server.py CONFIG.ini   (settings: recipe/speech-server.example.ini)

Starts recipe/pilot_macos.py on the engine port and, once it is ready, serves the public port in front of it.
Every request is relayed unchanged except speech. The Metal engine has no long-form splitter, and one long take
can stop speaking yet run on in silence to --max-frames (4096 frames, 190 s). So a long input is spoken segment
by segment and the audio joined, and the engine's silence guard (-40 dBFS for 10 s,
results/silence-guard-floor.json) ends a take that falls silent, redrawing a whole one once. Cuts fall at
sentence ends, else at spaces, never inside a special token, tag, paired tag or bracketed cue. A segment that
ends in silence or at the frame cap does not end the reply; the reply's X-Finish-Reason is its last segment's.
A take with no audio at all is drawn again with another seed; if every draw is empty, the reply is a 502 error,
never an empty success. The engine has one lane, taken one segment at a time: a short note (under 400 characters)
waiting goes before a long reply's next segment, so it waits for at most one segment, never a whole long reply;
a streamed long reply pauses for that note's take. Each take's start (with its length and time queued) and end
(with its seconds on the engine and of audio) are logged. From arrival to its last byte a speech request holds a
shared lock on SPEECH_LOCK, the turn that TensorFold's batch tools (tf-h3-image, tf-h3-dit) give way to between GPU
blocks; the engine's own hold (Metal patch 0007) covers only its takes, not the gaps between them.
"""
from __future__ import annotations

import configparser
import contextlib
import fcntl
import http.client
import http.server
import itertools
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Iterator

from weight_selection import DEFAULT_WEIGHTS

RECIPE = Path(__file__).resolve().parent
SPEECH = '/v1/audio/speech'
READY = 'DRAMA_PILOT_READY'
ENGINE_FLAGS = ('--max-frames', '4096', '--silence-floor-dbfs', '-40', '--silence-seconds', '10',
                '--silence-redraws', '1')
# A take that falls silent loses at most the rest of one segment. 600 is bin/voice-note's cap, so a voice note is
# always one take: separately drawn segments restart the intonation at their join.
SEGMENT_CHARS = 600
# A note shorter than this is one segment and goes before the next segment of a long reply.
SHORT_CHARS = 400
# The engine's own input limit; a longer input is relayed for the engine to refuse.
MAX_INPUT_CHARS = 4096
WAV_HEADER = 44
# Mono signed 16-bit PCM at 44.1 kHz, whole or streamed.
AUDIO_BYTES_PER_S = 44100 * 2
SPEECH_LOCK = '/tmp/tensorfold-speech.lock'
# The engine can sample end of speech at a take's first frame. Without a request seed it draws from the prompt,
# so the same text repeats it; such a take is drawn again with another seed. Metal patch 0002 refuses it with a
# 500 naming NO_AUDIO; an engine without the patch answers a clean stop with no audio.
EMPTY_REDRAWS = 2
NO_AUDIO = b'the model ended speech before its first audio frame'
LANE_RETRIES = 50
LANE_WAIT_S = 0.1
HOP_BY_HOP = {'connection', 'content-length', 'host', 'keep-alive', 'trailer', 'transfer-encoding'}

PROTECTED = re.compile(r'<(\w[\w-]*)[^<>]*>.*?</\1>|<[^<>]*>|\[[^\[\]]*\]', re.S)
SENTENCE_END = re.compile(r'[.!?…]["\'”’)]*\s|[。！？；]')
SPACE = re.compile(r'\s')
SPEAKER = re.compile(r'<\|speaker:\d+\|>')


def read_config(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(defaults={'weights': DEFAULT_WEIGHTS, 'host': '127.0.0.1', 'port': '18793',
                                                 'engine_port': '18794'})
    with path.open() as config:
        parser.read_file(config)
    return parser['server']


def pilot_command(server: configparser.SectionProxy) -> list[str]:
    """The launcher for the engine; relative paths are taken from the working directory, as its own are."""
    def place(key: str) -> str:
        return str(Path(server[key]).resolve())

    # Without a checkpoint, pilot_macos.py serves the selected weights' own folder in the repository root.
    checkpoint = ['--checkpoint', place('checkpoint')] if 'checkpoint' in server else []
    command = [sys.executable, str(RECIPE / 'pilot_macos.py'), '--weights', server['weights'], *checkpoint,
               '--runtime-dir', place('runtime_dir'), '--gpu-lock', place('gpu_lock'),
               '--evidence-dir', place('evidence_dir'), '--evidence-reuse',
               '--port', server['engine_port'], '--', *ENGINE_FLAGS]
    if 'voices' in server:
        command += ['--voices', place('voices')]
    if 'default_voice' in server:
        command += ['--default-voice', server['default_voice']]
    if server.getboolean('external_admission', fallback=False):
        command.insert(command.index('--'), '--external-admission')
    return command


def segment_end(masked: str, start: int) -> int:
    limit = start + SEGMENT_CHARS
    for rule in (SENTENCE_END, SPACE):
        ends = [match.end() for match in rule.finditer(masked, start, limit)]
        if ends:
            return ends[-1]
    # one protected span or word longer than a segment stays whole
    after = SPACE.search(masked, limit)
    return after.end() if after else len(masked)


def segments(text: str) -> list[str]:
    """``text`` in segments of at most SEGMENT_CHARS opening with the speaker tag in force; one is as sent."""
    if len(text.strip()) <= SEGMENT_CHARS:
        return [text]
    masked = PROTECTED.sub(lambda span: '_' * len(span.group()), text)
    pieces: list[str] = []
    start, speaker = 0, ''
    while start < len(text):
        end = len(text) if len(text) - start <= SEGMENT_CHARS else segment_end(masked, start)
        piece = text[start:end].strip()
        if piece:
            pieces.append(f'{speaker} {piece}' if speaker and not SPEAKER.match(piece) else piece)
            speaker = (SPEAKER.findall(piece) or [speaker])[-1]
        start = end
    return pieces


def speech_request(body: bytes) -> dict | None:
    """The speech request this server speaks; anything else is relayed for the engine to refuse."""
    try:
        request = json.loads(body)
    except ValueError:
        return None
    text = request.get('input') if isinstance(request, dict) else None
    return request if isinstance(text, str) and len(text) <= MAX_INPUT_CHARS else None


def draws(request: dict, text: str) -> Iterator[bytes]:
    """The engine request for each take of ``text``: as asked, then redraw k at the request seed (or 0) + k."""
    take = {**request, 'input': text}
    yield json.dumps(take).encode()
    for k in range(1, EMPTY_REDRAWS + 1):
        yield json.dumps({**take, 'seed': ((take.get('seed') or 0) + k) % 2**64}).encode()


def chunks(reply: BinaryIO, trailer: dict[str, str]) -> Iterator[bytes]:
    """The data of a chunked body, filling ``trailer``; a body cut short raises ValueError."""
    while size := int(reply.readline().split(b';')[0], 16):
        yield reply.read(size)
        reply.readline()
    while (line := reply.readline().decode('latin-1')).strip():
        name, _, value = line.partition(':')
        trailer[name.strip().lower()] = value.strip()


def whole_audio(headers: http.client.HTTPMessage, reply: BinaryIO) -> bytes | None:
    """A whole take's body, or None when it holds no audio."""
    body = reply.read(int(headers['Content-Length']))
    return body if len(body) > (WAV_HEADER if headers['Content-Type'] == 'audio/wav' else 0) else None


def stream_audio(reply: BinaryIO, trailer: dict[str, str]) -> Iterator[bytes] | None:
    """A streamed take's chunks, filling ``trailer``, or None when it ends before any."""
    trailer.clear()
    audio = chunks(reply, trailer)
    first = next(audio, None)
    return None if first is None else itertools.chain([first], audio)


def joined_wav(parts: list[bytes]) -> bytes:
    data = b''.join(part[WAV_HEADER:] for part in parts)
    head = parts[0][:WAV_HEADER]
    return head[:4] + struct.pack('<I', 36 + len(data)) + head[8:40] + struct.pack('<I', len(data)) + data


@contextlib.contextmanager
def gpu_turn(path: str) -> Iterator[None]:
    """A shared hold on ``path`` while live speech needs the GPU; batch work probes it with a nonblocking exclusive
    flock and pauses until it is free. Read-only and best effort, as the engine's own hold is. A symlink at ``path``
    is not followed, so one planted in the shared /tmp cannot have its target's mode changed; speech then goes on
    without the hold."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    except OSError:
        yield
        return
    try:
        # past the creator's umask, as patch 0007 does, so batch work running as another user can open it
        with contextlib.suppress(OSError):
            os.fchmod(fd, 0o644)
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server: Proxy

    def do_GET(self) -> None:
        self.relay(b'')

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        request = speech_request(body) if self.path == SPEECH else None
        if request is None:
            return self.relay(body)
        # each take's queue time runs from the request's arrival, then from its previous take's end
        self.queued = time.monotonic()
        self.short = len(request['input']) < SHORT_CHARS
        with gpu_turn(self.server.speech_lock):
            if self.short:
                return self.reply(request)
            # long replies speak one at a time, so a stream never pauses for another long reply's segments
            with self.server.long:
                self.reply(request)

    def reply(self, request: dict) -> None:
        if request.get('stream'):
            self.stream(request)
        else:
            self.whole(request)

    @contextlib.contextmanager
    def turn(self, n: int, of: int, text: str) -> Iterator[None]:
        """The engine lane for take ``n`` of ``of``, logged so a slow queue shows; ``self.audio`` counts its bytes."""
        with self.server.lane.turn(self.short):
            started = time.monotonic()
            self.audio = 0
            self.log_message('take %d/%d start: %d chars, %.2f s queued', n, of, len(text), started - self.queued)
            try:
                yield
            finally:
                self.queued = time.monotonic()
                self.log_message('take %d/%d end: %.2f s on the engine, %.2f s of audio', n, of,
                                 self.queued - started, self.audio / AUDIO_BYTES_PER_S)

    def upstream(self, body: bytes) -> socket.socket:
        engine = socket.create_connection(self.server.engine)
        head = [f'{self.command} {self.path} HTTP/1.1', 'Host: {}:{}'.format(*self.server.engine),
                'Connection: close', f'Content-Length: {len(body)}']
        head += [f'{name}: {value}' for name, value in self.headers.items() if name.lower() not in HOP_BY_HOP]
        engine.sendall(('\r\n'.join(head) + '\r\n\r\n').encode('latin-1') + body)
        return engine

    def relay(self, body: bytes) -> None:
        """The engine's reply byte for byte, trailers included."""
        with self.upstream(body) as engine:
            while data := engine.recv(65536):
                self.wfile.write(data)
        self.close_connection = True

    def speak(self, body: bytes) -> tuple[int, http.client.HTTPMessage, BinaryIO]:
        # the engine frees its lane just after the previous segment's reply, so a 503 is brief
        for _ in range(LANE_RETRIES):
            with self.upstream(body) as engine:
                reply = engine.makefile('rb')
            status = int(reply.readline().split()[1])
            headers = http.client.parse_headers(reply)
            if status != 503:
                break
            reply.read(int(headers['Content-Length']))
            time.sleep(LANE_WAIT_S)
        return status, headers, reply

    def take(self, request: dict, text: str, listen) -> tuple[int, http.client.HTTPMessage, object] | None:
        """The first take of ``text`` that ``listen`` hears audio in, or a refusal; None if all are empty."""
        for k, body in enumerate(draws(request, text)):
            if k:
                self.log_message('the engine spoke no audio; redraw %d of %d', k, EMPTY_REDRAWS)
            status, headers, reply = self.speak(body)
            if status != 200:
                refusal = reply.read(int(headers['Content-Length']))
                if status == 500 and NO_AUDIO in refusal:
                    continue
                return status, headers, refusal
            audio = listen(headers, reply)
            if audio is not None:
                return status, headers, audio
        return None

    def send_head(self, status: int, headers: http.client.HTTPMessage) -> None:
        # the engine closes after every reply, and so does this server
        self.send_response_only(status)
        self.send_header('Connection', 'close')
        for name, value in headers.items():
            if name.lower() not in HOP_BY_HOP:
                self.send_header(name, value)

    def answer(self, status: int, headers: http.client.HTTPMessage, body: bytes) -> None:
        self.send_head(status, headers)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def no_audio(self) -> None:
        headers = http.client.HTTPMessage()
        headers['Content-Type'] = 'application/json'
        message = f'Drama 3 spoke no audio in {EMPTY_REDRAWS + 1} draws with different seeds'
        self.answer(502, headers, json.dumps({'error': {'message': message, 'type': 'server_error'}}).encode())

    def whole(self, request: dict) -> None:
        parts = []
        pieces = segments(request['input'])
        for n, text in enumerate(pieces, 1):
            with self.turn(n, len(pieces), text):
                take = self.take(request, text, whole_audio)
                if take is not None and take[0] == 200:
                    self.audio = len(take[2]) - (WAV_HEADER if take[1]['Content-Type'] == 'audio/wav' else 0)
            if take is None:
                return self.no_audio()
            status, headers, body = take
            if status != 200:
                return self.answer(status, headers, body)
            parts.append(body)
        audio = joined_wav(parts) if headers['Content-Type'] == 'audio/wav' else b''.join(parts)
        self.answer(200, headers, audio)

    def stream(self, request: dict) -> None:
        started = False
        trailer: dict[str, str] = {}
        pieces = segments(request['input'])
        for n, text in enumerate(pieces, 1):
            # the engine speaks while the audio is relayed, so the turn lasts until the take's last chunk
            with self.turn(n, len(pieces), text):
                take = self.take(request, text, lambda _, reply: stream_audio(reply, trailer))
                if take is None or take[0] != 200:
                    if started:
                        # audio is already out: no terminating chunk, so the client sees an incomplete body
                        self.close_connection = True
                        return
                    return self.no_audio() if take is None else self.answer(*take)
                _, headers, audio = take
                if not started:
                    self.send_head(200, headers)
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.send_header('Trailer', 'X-Finish-Reason')
                    self.end_headers()
                    started = True
                for data in audio:
                    self.audio += len(data)
                    self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
        reason = trailer.get('x-finish-reason')
        self.wfile.write(b'0\r\n' + (f'X-Finish-Reason: {reason}\r\n'.encode() if reason else b'') + b'\r\n')


class Lane:
    """The engine's one lane: one take at a time, and a short note waiting takes the next turn."""

    def __init__(self):
        self.free = threading.Condition()
        self.taken = False
        self.short_waiting = 0

    @contextlib.contextmanager
    def turn(self, short: bool) -> Iterator[None]:
        with self.free:
            self.short_waiting += short
            self.free.wait_for(lambda: not self.taken and (short or not self.short_waiting))
            self.short_waiting -= short
            self.taken = True
        try:
            yield
        finally:
            with self.free:
                self.taken = False
                self.free.notify_all()


class Proxy(http.server.ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], engine: tuple[str, int], speech_lock: str = SPEECH_LOCK):
        super().__init__(address, Handler)
        self.engine = engine
        self.speech_lock = speech_lock
        self.long = threading.Lock()
        self.lane = Lane()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    server = read_config(Path(sys.argv[1]))
    # bound before the model loads, so a taken port fails at once; requests wait until the engine is ready
    proxy = Proxy((server['host'], server.getint('port')), ('127.0.0.1', server.getint('engine_port')))
    pilot = subprocess.Popen(pilot_command(server), stdout=subprocess.PIPE, text=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: pilot.terminate())
    serving = None
    for line in pilot.stdout:
        print(line, end='', flush=True)
        if line.strip() == READY:
            serving = threading.Thread(target=proxy.serve_forever)
            serving.start()
    if serving is not None:
        proxy.shutdown()
    proxy.server_close()
    return pilot.wait()


if __name__ == '__main__':
    raise SystemExit(main())
