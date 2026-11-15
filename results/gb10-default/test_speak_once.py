import socket
import subprocess
import sys
import unittest
from unittest import mock

import speak_once

# A stand-in for tensorfold-native serve: argv is MODE PORT. ok streams one second of 8 kHz PCM in two chunks,
# error answers the speech request 500, exit dies before /health answers.
FAKE = r'''
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
mode, port = sys.argv[1], int(sys.argv[2])
if mode == 'exit':
    sys.exit(3)
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'{}')
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        if mode == 'error':
            body = b'{"error": "voice not found"}'
            self.send_response(500)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('X-Sample-Rate', '8000')
        self.end_headers()
        for _ in range(2):
            self.wfile.write(b'1f40\r\n' + bytes(8000) + b'\r\n')
        self.wfile.write(b'0\r\n\r\n')
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class SpeakOnceTest(unittest.TestCase):
    def run_fake(self, mode):
        port = free_port()
        with mock.patch.object(speak_once, 'PORT', port):
            return speak_once.serve_and_speak([sys.executable, '-c', FAKE, mode, str(port)], ready_timeout=30)

    def test_reports_status_audio_seconds_and_first_audio(self):
        status, audio_s, first_audio_s = self.run_fake('ok')
        self.assertEqual((status, audio_s), (200, 1.0))
        self.assertGreater(first_audio_s, 0)

    def test_an_http_error_fails_with_the_servers_reply(self):
        with self.assertRaisesRegex(RuntimeError, r'^HTTP 500: \{"error": "voice not found"\}$'):
            self.run_fake('error')

    def test_a_server_that_exits_before_health_fails_with_its_status(self):
        with self.assertRaisesRegex(RuntimeError, 'exited with status 3 before /health answered'):
            self.run_fake('exit')

    def test_an_occupied_port_fails_instead_of_using_its_server(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            with mock.patch.object(speak_once, 'PORT', port):
                with self.assertRaisesRegex(RuntimeError, f'127.0.0.1:{port} is unavailable: .*Address already in use'):
                    speak_once.serve_and_speak([sys.executable, '-c', FAKE, 'exit', str(port)])

    def test_a_port_taken_between_check_and_launch_fails(self):
        port = free_port()
        other_servers = []
        real_popen = subprocess.Popen

        def collide(command, **kwargs):
            if command[0] == 'lsof':
                return real_popen(command, **kwargs)
            other_servers.append(real_popen([sys.executable, '-c', FAKE, 'ok', str(port)]))
            return real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)

        try:
            with mock.patch.object(speak_once, 'PORT', port):
                with mock.patch.object(speak_once.subprocess, 'Popen', side_effect=collide):
                    with self.assertRaisesRegex(RuntimeError, f'port {port} is not owned by launched process'):
                        speak_once.serve_and_speak(['ignored'], ready_timeout=30)
        finally:
            for server in other_servers:
                server.terminate()
                server.wait(30)


if __name__ == '__main__':
    unittest.main()
