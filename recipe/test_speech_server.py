#!/usr/bin/env python3
"""recipe/speech_server.py in front of a loopback stand-in for the engine: no model, GPU or network.

The stand-in speaks a request's input as its UTF-8 bytes, so a joined reply shows every segment, in order.
"""
from __future__ import annotations

import http.client
import http.server
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest

import speech_server

LONG = ' '.join(f'Sentence {n} keeps the train on time, and the guard waves from the platform.' for n in range(40))[:2400]


def wav(data: bytes) -> bytes:
    return (b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 44100, 88200, 2, 16)
            + b'data' + struct.pack('<I', len(data)) + data)


class Engine(http.server.ThreadingHTTPServer):
    """Built-in voice plus the catalogs in ``voices``; finish reasons are taken from ``reasons`` in turn.

    The first ``empty`` takes end at frame 0 with no audio: a 500 naming it, as Metal patch 0008 answers, or with
    ``empty_ok`` a clean stop, as an engine without the patch does.

    As the real engine does, it answers 503 once right after a reply, before its lane is free again.
    """

    def __init__(self, voices: set[str], reasons: list[str], empty: int, empty_ok: bool):
        super().__init__(('127.0.0.1', 0), EngineHandler)
        self.voices = {'default'} | voices
        self.reasons = reasons
        self.empty = empty
        self.empty_ok = empty_ok
        self.requests: list[dict] = []
        self.busy = False


class EngineHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server: Engine

    def log_message(self, *_):
        pass

    def refuse(self, status: int, message: str):
        body = json.dumps({'error': {'message': message, 'type': 'invalid_request_error'}}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.close_connection = True
        if self.server.busy:
            self.server.busy = False
            return self.refuse(503, 'speech output capacity is full')
        self.server.requests.append(request)
        if request.get('voice', 'default') not in self.server.voices:
            return self.refuse(400, 'Drama 3 request: unknown voice')
        self.server.busy = True
        if request['input'] == 'The codec fails.':
            return self.refuse(500, 'the codec produced non-finite audio')
        audio = request['input'].encode()
        reason = self.server.reasons.pop(0) if self.server.reasons else 'stop'
        if self.server.empty:
            self.server.empty -= 1
            if not self.server.empty_ok:
                return self.refuse(500, speech_server.NO_AUDIO.decode())
            audio, reason = b'', 'stop'
        self.send_response(200)
        if request.get('stream'):
            self.send_header('Content-Type', 'audio/pcm')
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Trailer', 'X-Finish-Reason')
            self.end_headers()
            for start in range(0, len(audio), 100):
                piece = audio[start:start + 100]
                self.wfile.write(b'%x\r\n%s\r\n' % (len(piece), piece))
            self.wfile.write(f'0\r\nX-Finish-Reason: {reason}\r\n\r\n'.encode())
            return
        body = wav(audio) if request.get('response_format', 'wav') == 'wav' else audio
        self.send_header('Content-Type', 'audio/wav' if body is not audio else 'audio/pcm')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Finish-Reason', reason)
        self.end_headers()
        self.wfile.write(body)


class ServerTests(unittest.TestCase):
    def start(self, voices=frozenset(), reasons=(), empty=0, empty_ok=False):
        self.engine = Engine(set(voices), list(reasons), empty, empty_ok)
        self.proxy = speech_server.Proxy(('127.0.0.1', 0), self.engine.server_address)
        self.proxy.RequestHandlerClass.log_message = lambda *_: None
        for server in (self.engine, self.proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def post(self, **request) -> http.client.HTTPResponse:
        client = http.client.HTTPConnection(*self.proxy.server_address, timeout=10)
        self.addCleanup(client.close)
        client.request('POST', '/v1/audio/speech', json.dumps({'model': 'fishaudio/drama-3', **request}),
                       {'Content-Type': 'application/json'})
        return client.getresponse()

    def raw_stream(self, **request) -> bytes:
        body = json.dumps({'model': 'fishaudio/drama-3', 'response_format': 'pcm', 'stream': True, **request})
        with socket.create_connection(self.proxy.server_address, timeout=10) as client:
            client.sendall(f'POST /v1/audio/speech HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n'
                           f'Connection: close\r\n\r\n{body}'.encode())
            return b''.join(iter(lambda: client.recv(65536), b''))

    def test_default_voice_is_relayed_unchanged(self):
        self.start()
        reply = self.post(input='The train leaves at seven fifteen.')
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.read(), wav(b'The train leaves at seven fifteen.'))
        self.assertEqual(reply.headers['X-Finish-Reason'], 'stop')
        self.assertEqual(self.engine.requests, [{'model': 'fishaudio/drama-3', 'input': 'The train leaves at seven fifteen.'}])

    def test_cloned_voice_from_the_config_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            voices = Path(folder, 'voices')
            voices.mkdir()
            (voices / 'synthetic-a.json').write_text('{}')
            config = Path(folder, 'speech-server.ini')
            config.write_text(f'[server]\ncheckpoint = model\nvoices = {voices}\ndefault_voice = synthetic-a\n'
                              f'gpu_lock = gpu.lock\nruntime_dir = runtime\nevidence_dir = runs\n')
            command = speech_server.pilot_command(speech_server.read_config(config))
        self.assertEqual(command[command.index('--voices') + 1], str(voices.resolve()))
        self.assertEqual(command[command.index('--default-voice') + 1], 'synthetic-a')
        self.assertEqual(command[command.index('--port') + 1], '18794')
        self.start(voices={'synthetic-a'})
        self.assertEqual(self.post(input=LONG, voice='synthetic-a').status, 200)
        self.assertEqual({request['voice'] for request in self.engine.requests}, {'synthetic-a'})

    def test_external_admission_is_left_to_the_machine_only_when_set(self):
        base = '[server]\ncheckpoint = model\ngpu_lock = gpu.lock\nruntime_dir = runtime\nevidence_dir = runs\n'
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder, 'speech-server.ini')
            config.write_text(base)
            default = speech_server.pilot_command(speech_server.read_config(config))
            config.write_text(base + 'external_admission = true\n')
            external = speech_server.pilot_command(speech_server.read_config(config))
        self.assertNotIn('--external-admission', default)
        # a launcher flag, before the serve flags
        self.assertEqual(external, default[:default.index('--')] + ['--external-admission']
                         + default[default.index('--'):])

    def test_long_form_does_not_stop_at_a_silent_segment(self):
        self.assertEqual(len(LONG), 2400)
        pieces = speech_server.segments(LONG)
        self.assertGreater(len(pieces), 4)
        self.assertTrue(all(len(piece) <= speech_server.SEGMENT_CHARS for piece in pieces))
        self.assertTrue(all(piece.endswith('.') for piece in pieces[:-1]))
        self.start(reasons=['silence'])
        reply = self.post(input=LONG)
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.read(), wav(''.join(pieces).encode()))
        self.assertEqual(reply.headers['X-Finish-Reason'], 'stop')
        self.assertEqual([request['input'] for request in self.engine.requests], pieces)

    def test_long_form_stream_carries_every_segment_and_the_last_finish_reason(self):
        self.start(reasons=['silence', 'stop', 'stop', 'stop', 'stop', 'length'])
        head, _, body = self.raw_stream(input=LONG).partition(b'\r\n\r\n')
        self.assertIn(b'Transfer-Encoding: chunked', head)
        self.assertTrue(body.endswith(b'0\r\nX-Finish-Reason: length\r\n\r\n'))
        audio, rest = b'', body
        while size := int(rest.split(b'\r\n', 1)[0], 16):
            data_start = rest.index(b'\r\n') + 2
            audio += rest[data_start:data_start + size]
            rest = rest[data_start + size + 2:]
        self.assertEqual(audio, ''.join(speech_server.segments(LONG)).encode())

    def test_an_empty_take_is_redrawn_with_another_seed(self):
        text = 'The train leaves at seven fifteen.'
        for empty_ok in (False, True):
            with self.subTest(empty_ok=empty_ok):
                self.start(empty=1, empty_ok=empty_ok)
                reply = self.post(input=text)
                self.assertEqual(reply.status, 200)
                self.assertEqual(reply.read(), wav(text.encode()))
                self.assertEqual([request.get('seed') for request in self.engine.requests], [None, 1])
                self.start(empty=2, empty_ok=empty_ok)
                self.assertIn(b'\r\n\r\n22\r\nThe train', self.raw_stream(input=text, seed=2**64 - 1))
                self.assertEqual([request['seed'] for request in self.engine.requests], [2**64 - 1, 0, 1])

    def test_only_empty_takes_answer_a_clear_error_not_success(self):
        for empty_ok in (False, True):
            with self.subTest(empty_ok=empty_ok):
                self.start(empty=3, empty_ok=empty_ok)
                reply = self.post(input='The train leaves at seven fifteen.')
                self.assertEqual(reply.status, 502)
                self.assertEqual(json.loads(reply.read())['error']['type'], 'server_error')
                self.assertEqual(len(self.engine.requests), 3)
                self.start(empty=3, empty_ok=empty_ok)
                head = self.raw_stream(input='The train leaves at seven fifteen.')
                self.assertTrue(head.startswith(b'HTTP/1.1 502'))
                self.assertIn(b'"server_error"', head)

    def test_another_engine_failure_is_passed_on_without_a_redraw(self):
        self.start()
        reply = self.post(input='The codec fails.')
        self.assertEqual(reply.status, 500)
        self.assertEqual(len(self.engine.requests), 1)

    def test_bad_request_returns_the_engine_refusal_and_speaks_nothing_more(self):
        self.start()
        reply = self.post(input=LONG, voice='nobody')
        self.assertEqual(reply.status, 400)
        self.assertEqual(json.loads(reply.read())['error']['type'], 'invalid_request_error')
        self.assertEqual(len(self.engine.requests), 1)
        self.assertIn(b'HTTP/1.1 400', self.raw_stream(input=LONG, voice='nobody'))

    def test_cuts_keep_tags_cues_and_the_speaker(self):
        cue = '<whisper>' + 'hush. ' * 100 + '</whisper>'
        text = '<|speaker:1|> ' + 'A plain line of speech. ' * 20 + cue + ' [laughs, then sighs] The end.'
        pieces = speech_server.segments(text)
        self.assertEqual(pieces[1], '<|speaker:1|> ' + cue)
        self.assertEqual(pieces[2], '<|speaker:1|> [laughs, then sighs] The end.')
        self.assertTrue(pieces[0].startswith('<|speaker:1|> A plain'))


if __name__ == '__main__':
    unittest.main()
