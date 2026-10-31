#!/usr/bin/env python3
"""Run a guarded macOS pilot without systemd, cgroups, or a hard memory cap.

Admission and monitoring use one conservative vm_stat free+inactive snapshot.
Speculative and purgeable pages are not added again. The default admission floor
is weights + 30 GiB, but never below the 34 GiB abort threshold.

The server owns a new process group and inherits the shared GPU lock. Cleanup
signals only that group, keeping its leader unreaped until all members stop.
This is not containment: a server must not daemonize, escape the group, or close
its inherited lock. Observation and TERM/KILL cleanup have bounded deadlines;
inability to confirm settlement is reported as failure, not successful cleanup.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

# python3 -I leaves the script's own directory off sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_selection  # noqa: E402

GIB = 1024**3
HEADROOM = 30 * GIB
STOP_MARGIN = 4 * GIB
ABORT_AVAILABLE = HEADROOM + STOP_MARGIN
OBSERVATION_TIMEOUT = 2.0
TERM_TIMEOUT = 5.0
KILL_TIMEOUT = 5.0


def parse_memory_available(text: str) -> int:
    sizes = re.findall(r'page size of (\d+) bytes', text)
    if len(sizes) != 1 or int(sizes[0]) <= 0:
        raise ValueError('vm_stat did not report one positive page size')
    counts = []
    for label in ('Pages free', 'Pages inactive'):
        matches = re.findall(r'^' + re.escape(label) + r':[ \t]+(\d+)\.?[ \t]*$',
                             text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f'vm_stat did not report one valid {label} count')
        counts.append(int(matches[0]))
    return int(sizes[0]) * sum(counts)


def memory_available() -> int:
    snapshot = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True,
                              check=True, timeout=OBSERVATION_TIMEOUT)
    return parse_memory_available(snapshot.stdout)


def health_ready(host: str, port: int, timeout: float) -> bool:
    # Read only the HTTP status line, under one total deadline. A trickling
    # response must not indefinitely postpone memory checks; no proxies or DNS.
    deadline = time.monotonic() + timeout
    with socket.socket() as connection:
        connection.settimeout(timeout)
        connection.connect((host, port))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Health observation deadline exceeded')
        connection.settimeout(remaining)
        connection.sendall(b'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n')
        response = bytearray()
        while len(response) < 1024:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Health observation deadline exceeded')
            connection.settimeout(remaining)
            chunk = connection.recv(1024 - len(response))
            if not chunk:
                return False
            response.extend(chunk)
            if b'\n' in response:
                status = response.split(b'\n', 1)[0].split()
                return (len(status) >= 2 and status[0] in (b'HTTP/1.0', b'HTTP/1.1')
                        and status[1] == b'200')
    return False


def group_states(pgid: int, timeout: float = OBSERVATION_TIMEOUT) -> dict[int, str]:
    # Do not poll()/wait() the leader: its zombie pins our PGID until cleanup ends.
    snapshot = subprocess.run(['/bin/ps', '-x', '-g', str(pgid), '-o', 'pid=,pgid=,stat='],
                              capture_output=True, text=True, check=True, timeout=timeout)
    states = {}
    for line in snapshot.stdout.splitlines():
        pid, group, state = line.split()
        if int(group) == pgid:
            states[int(pid)] = state
    if pgid not in states:
        raise RuntimeError('Owned process-group leader disappeared before it was reaped')
    return states


def settle_group(worker: subprocess.Popen, errors: list[str]) -> bool:
    for sig, duration in ((signal.SIGTERM, TERM_TIMEOUT), (signal.SIGKILL, KILL_TIMEOUT)):
        try:
            os.killpg(worker.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f'Group signal {sig.name}: {type(exc).__name__}: {exc}')
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            try:
                states = group_states(worker.pid, min(OBSERVATION_TIMEOUT,
                                                      max(0.001, deadline - time.monotonic())))
                if all(state.startswith('Z') for state in states.values()):
                    # No group signal is permitted after this reap.
                    worker.wait(timeout=OBSERVATION_TIMEOUT)
                    return True
            except Exception as exc:
                errors.append(f'Group observation: {type(exc).__name__}: {exc}')
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    errors.append('Owned process group did not reach confirmed settlement after SIGKILL')
    return False


def finite_nonnegative(number: float) -> float:
    if not math.isfinite(number) or number < 0:
        raise ValueError('must be finite and nonnegative')
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=('native', 'sglang-omni'), default='native')
    parser.add_argument('--weights', choices=tuple(weight_selection.WEIGHTS),
                        help='Download (first run), verify and serve pinned weights; nvfp4 adds '
                             '--cfg-gamma 1.5 --cfg-batched unless guidance follows --')
    parser.add_argument('--native-bin', type=Path,
                        help='Patched executable (default with --weights: its engine in build/)')
    parser.add_argument('--runtime-dir', type=Path, required=True,
                        help='Isolated runtime and cache directory')
    parser.add_argument('--checkpoint', type=Path,
                        help='Checkpoint directory (with --weights: verified against the selected pins)')
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Existing lock file that every GPU workload on this host takes')
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--evidence-reuse', action='store_true',
                        help='Replace prior attempt evidence after acquiring the shared GPU lock')
    parser.add_argument('--port', type=int, default=18793)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IPv4 bind address for the server; the port check uses it too')
    parser.add_argument('--ready-timeout', type=float, default=900)
    parser.add_argument('--min-available-gib', type=float, default=None,
                        help='Override admission, never below 34 GiB (default: weights + 30 GiB)')
    parser.add_argument('--reserve-gib', type=float, default=34,
                        help='TENSORFOLD_MEMORY_RESERVE_GIB passed to the server (not a hard cap)')
    parser.add_argument('serve_args', nargs='*', metavar='SERVE_ARG',
                        help='Extra tensorfold-native serve flags, after --')
    args = parser.parse_args()
    if not args.gpu_lock.is_file():
        parser.error('The operator-specified shared GPU lock must already exist')
    if args.weights is None and args.checkpoint is None:
        parser.error('Pass --weights nvfp4 or --weights full, or --checkpoint DIR')

    root = args.runtime_dir.resolve()
    checkpoint = weight_selection.checkpoint_path(args.weights, args.checkpoint)
    evidence = args.evidence_dir.resolve()

    with args.gpu_lock.open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another workload owns the GPU slot; no model was started')

        try:
            if root == evidence:
                raise ValueError('Runtime and evidence directories must be distinct')
            for path in (root, evidence):
                if path == checkpoint or checkpoint in path.parents:
                    raise ValueError('Runtime and evidence must be outside the checkpoint directory')
            evidence.mkdir(parents=True, exist_ok=args.evidence_reuse)
            # Invalidate a reused attempt before any admission check can fail. Never
            # touch another live attempt's evidence before acquiring the GPU lock.
            for name in ('ready.json', 'ready.json.tmp', 'session.json', 'completion.json',
                         'memory.jsonl', 'server.log'):
                (evidence / name).unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot prepare evidence: {exc}; no model was started')

        failure: list[str] = []
        cleanup_failure: list[str] = []
        minimum_available = None
        final_available = None
        started = time.monotonic()
        started_at_epoch = time.time()
        ready_at = None
        worker = None
        group_settled = True
        unexpected_exit = False
        stop_signal = None

        def request_stop(signum: int, _frame: object) -> None:
            nonlocal stop_signal
            stop_signal = signum

        previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if not 1 <= args.port <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            # Validate after invalidating reused evidence, including NaN/Inf that
            # argparse's float conversion accepts.
            for name in ('ready_timeout', 'min_available_gib', 'reserve_gib'):
                value = getattr(args, name)
                if value is not None:
                    try:
                        finite_nonnegative(value)
                    except ValueError as exc:
                        raise ValueError(f'--{name.replace("_", "-")} {exc}') from exc
            socket.inet_pton(socket.AF_INET, args.host)
            if args.backend != 'native':
                raise ValueError('sglang-omni needs the CUDA reference stack; this launcher is native-only')
            if args.min_available_gib is not None and args.min_available_gib < ABORT_AVAILABLE / GIB:
                raise ValueError('Admission override cannot be below the 34 GiB abort threshold')
            for value in (args.reserve_gib, args.min_available_gib):
                if value is not None and not math.isfinite(value * GIB):
                    raise ValueError('GiB argument is too large')
            executable = weight_selection.native_path(args.weights, args.native_bin)
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f'Missing executable server: {executable}')
            provenance = weight_selection.prepare_checkpoint(args.weights, args.checkpoint)
            backbone = sum(p.stat().st_size for p in checkpoint.glob('*.safetensors'))
            if backbone <= 0:
                raise ValueError('No backbone safetensors found')
            weights = backbone + (checkpoint / 'codec.pth').stat().st_size
            available = memory_available()
            minimum_available = available
            floor = (args.min_available_gib * GIB if args.min_available_gib is not None
                     else max(weights + HEADROOM, ABORT_AVAILABLE))
            if available < floor:
                raise MemoryError(
                    f'Need {floor/GIB:.2f} GiB available, have {available/GIB:.2f}; no model was started')
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_check.bind((args.host, args.port))
            root.mkdir(parents=True, exist_ok=True)
            for name in ('hf-cache', 'runtime-cache'):
                (root / name).mkdir(exist_ok=True)

            serve_args = weight_selection.serve_arguments(args.weights, args.serve_args)
            server_command = [str(executable), 'serve', str(checkpoint), '--host', args.host,
                              '--port', str(args.port), '--name', 'fishaudio/drama-3',
                              '--context', '8192', '--parallel', '1', '--no-drafts',
                              '--prompt-cache-gib', '0', '--keep-warm', '0', *serve_args]
            environment = {
                'HOME': str(Path.home()),
                'PATH': f'{root / "venv/bin"}:/usr/bin:/bin:/usr/sbin:/sbin',
                'LANG': 'C.UTF-8',
                'PYTHONUNBUFFERED': '1',
                'HF_HOME': str(root / 'hf-cache'),
                'HF_HUB_OFFLINE': '1',
                'XDG_CACHE_HOME': str(root / 'runtime-cache'),
                'TENSORFOLD_MEMORY_RESERVE_GIB': f'{args.reserve_gib:g}',
            }
            initial = {**provenance, 'backend': args.backend, 'weights_bytes': weights,
                       'initial_memavailable_bytes': available,
                       'required_memavailable_bytes': int(floor),
                       'headroom_floor_bytes': HEADROOM, 'abort_threshold_bytes': ABORT_AVAILABLE,
                       'memory_source': 'vm_stat free+inactive',
                       'min_available_override_gib': args.min_available_gib,
                       'server_command': server_command, 'environment': environment,
                       'started_at_epoch': started_at_epoch,
                       'admission': 'override' if args.min_available_gib is not None
                                    else 'max(weights + 30 GiB, 34 GiB)',
                       'engine_memory_reserve_gib': args.reserve_gib,
                       'hard_memory_cap': False}
            (evidence / 'session.json').write_text(json.dumps(initial, indent=2) + '\n')
            (root / 'launched-at').write_text(f'{started_at_epoch:.6f}\n')
            print(json.dumps(initial), flush=True)

            # A single loop bounds every observation and avoids a watchdog racing
            # cleanup or signaling a PGID after the leader has been reaped.
            with (evidence / 'server.log').open('w') as server_log, \
                    (evidence / 'memory.jsonl').open('w') as memory_log:
                if stop_signal is None:
                    worker = subprocess.Popen(server_command, stdout=server_log,
                                              stderr=subprocess.STDOUT, cwd=root,
                                              env=environment, start_new_session=True,
                                              pass_fds=(lock.fileno(),))
                    group_settled = False
                    initial['server_pid'] = worker.pid
                    initial['server_pgid'] = worker.pid
                    (evidence / 'session.json').write_text(json.dumps(initial, indent=2) + '\n')
                while stop_signal is None and worker is not None:
                    states = group_states(worker.pid)
                    if states[worker.pid].startswith('Z'):
                        unexpected_exit = True
                        break
                    current = memory_available()
                    minimum_available = min(minimum_available, current)
                    memory_log.write(json.dumps({'elapsed_s': time.monotonic() - started,
                                                 'memavailable_bytes': current}) + '\n')
                    memory_log.flush()
                    if current < ABORT_AVAILABLE:
                        raise MemoryError(
                            f'available memory {current/GIB:.2f} GiB crossed the '
                            f'{ABORT_AVAILABLE/GIB:.0f} GiB abort threshold')
                    if ready_at is None:
                        remaining = args.ready_timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError('Readiness deadline exceeded')
                        health_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
                        try:
                            healthy = health_ready(health_host, args.port, min(1, remaining))
                        except OSError:
                            healthy = False
                        if healthy and stop_signal is None:
                            ready_at = time.monotonic()
                            initial['cold_ready_s'] = ready_at - started
                            initial['ready_at_epoch'] = time.time()
                            ready_file = evidence / 'ready.json.tmp'
                            ready_file.write_text(json.dumps(initial, indent=2) + '\n')
                            ready_file.replace(evidence / 'ready.json')
                            print('DRAMA_PILOT_READY', flush=True)
                    time.sleep(0.2)
        except Exception as exc:
            failure.append(f'{type(exc).__name__}: {exc}')
        finally:
            if worker is not None:
                group_settled = settle_group(worker, cleanup_failure)
                if unexpected_exit:
                    failure.append(f'Server exited unexpectedly with status {worker.returncode}'
                                   + (' after readiness' if ready_at is not None else ' before readiness'))
            try:
                final_available = memory_available()
            except Exception as exc:
                failure.append(f'Final memory observation: {type(exc).__name__}: {exc}')
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            final = {'ready': ready_at is not None, 'elapsed_s': time.monotonic() - started,
                     'minimum_memavailable_bytes': minimum_available,
                     'final_memavailable_bytes': final_available,
                     'server_pid': worker.pid if worker is not None else None,
                     'server_pgid': worker.pid if worker is not None else None,
                     'server_returncode': worker.returncode if worker is not None else None,
                     'stop_signal': stop_signal, 'group_settled': group_settled,
                     'failures': failure, 'cleanup_failures': cleanup_failure}
            (evidence / 'completion.json').write_text(json.dumps(final, indent=2) + '\n')
            print(json.dumps(final), flush=True)
    return 1 if failure or cleanup_failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
