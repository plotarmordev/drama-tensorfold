#!/usr/bin/env python3
"""Run a bounded Linux/systemd pilot without changing another service or GPU policy."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# python3 -I leaves the script's own directory off sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_selection  # noqa: E402

GIB = 1024**3
HEADROOM = 30 * GIB
STOP_MARGIN = 4 * GIB
CGROUP_CAP = 54 * GIB
CUDA_HOME = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))


def memory_available() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable; refusing an unguarded load')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', choices=tuple(weight_selection.WEIGHTS),
                        help='Download (first run), verify and serve these pinned weights with the engine '
                             'scripts/build-engine.sh --weights builds for them; nvfp4 also applies its measured '
                             'guidance, --cfg-gamma 1.5 --cfg-batched, unless a guidance flag follows --')
    parser.add_argument('--backend', choices=('native', 'sglang-omni'), default='native')
    parser.add_argument('--native-bin', type=Path,
                        help='tensorfold-native executable (default with --weights: its build in build/)')
    parser.add_argument('--library-dir', type=Path, action='append', default=[],
                        help='Isolated shared-library directory; may be repeated')
    parser.add_argument('--runtime-dir', type=Path, required=True,
                        help='Isolated runtime and cache directory')
    parser.add_argument('--checkpoint', type=Path,
                        help='Checkpoint directory (with --weights: verified against the selected pins)')
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Existing lock file that every GPU workload on this host takes; never a second lock')
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18793)
    parser.add_argument('--ready-timeout', type=float, default=900)
    parser.add_argument('serve_args', nargs='*', metavar='SERVE_ARG',
                        help='Extra tensorfold-native serve flags, after --, e.g. -- --max-frames 2048')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if args.weights is None and args.checkpoint is None:
        parser.error('Pass --weights nvfp4 or --weights full, or --checkpoint DIR')
    if args.weights == 'nvfp4' and args.backend != 'native':
        parser.error('Only the native backend serves the NVFP4 weights')
    root = args.runtime_dir.resolve()
    checkpoint = weight_selection.checkpoint_path(args.weights, args.checkpoint)
    evidence = args.evidence_dir.resolve()
    if args.backend == 'native':
        if args.native_bin is None and args.weights is None:
            parser.error('The native backend requires --native-bin, or --weights')
        executable = weight_selection.native_path(args.weights, args.native_bin)
        serve_args = weight_selection.serve_arguments(args.weights, args.serve_args)
        working_directory = root
        server_command = [str(executable), 'serve', str(checkpoint), '--host', '127.0.0.1',
                          '--port', str(args.port), '--name', 'fishaudio/drama-3',
                          '--context', '8192', '--parallel', '1', '--no-drafts',
                          '--prompt-cache-gib', '0', '--keep-warm', '0', *serve_args]
        backend_environment = ['TENSORFOLD_MEMORY_RESERVE_GIB=34',
                               'TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54',
                               *weight_selection.native_environment(args.weights)]
        allocator_caps = {'tracked_cuda_and_pinned_host_buffers': 54}
    elif args.serve_args:
        parser.error('Extra serve flags apply to the native backend only')
    else:
        executable = root / 'venv/bin/sgl-omni'
        working_directory = root / 'sglang-omni'
        server_command = [str(executable), 'serve', '--config',
                          str(Path(__file__).parent / 'reference/drama3-gb10.yaml'),
                          '--model-path', str(checkpoint),
                          '--vocoder.factory.codec_path', str(checkpoint),
                          '--host', '127.0.0.1', '--port', str(args.port),
                          '--model-name', 'fishaudio/drama-3']
        backend_environment = []
        allocator_caps = {'preprocessing': 4, 'tts_engine': 44, 'vocoder': 6}
    if not executable.is_file() or not os.access(executable, os.X_OK):
        hint = f'; build it with scripts/build-engine.sh --weights {args.weights}' if args.weights and not args.native_bin else ''
        parser.error(f'Missing executable server: {executable}{hint}')
    if args.library_dir:
        directories = [p.resolve() for p in args.library_dir]
        if any(not p.is_dir() for p in directories):
            parser.error('Every --library-dir must be an existing directory')
        backend_environment.append('LD_LIBRARY_PATH=' + ':'.join(map(str, directories)))
    if not args.gpu_lock.is_file():
        parser.error('The operator-specified shared GPU lock must already exist')
    try:
        provenance = weight_selection.prepare_checkpoint(args.weights, args.checkpoint)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    evidence.mkdir(parents=True, exist_ok=False)
    weights = sum(p.stat().st_size for p in checkpoint.glob('*.safetensors'))
    weights += (checkpoint / 'codec.pth').stat().st_size
    if weights <= (checkpoint / 'codec.pth').stat().st_size:
        parser.error('No backbone safetensors found')

    with args.gpu_lock.open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another workload owns the GPU slot; no model was started')
        available = memory_available()
        required = max(weights + HEADROOM, CGROUP_CAP + HEADROOM + STOP_MARGIN)
        if available < required:
            parser.error(f'Need {required/GIB:.2f} GiB available, have {available/GIB:.2f}; no model was started')
        try:
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_check.bind(('127.0.0.1', args.port))
        except OSError as exc:
            parser.error(f'Pilot port is unavailable: {exc}; no model was started')
        unit = f'drama3-pilot-{os.getpid()}-{time.time_ns()}.service'
        stop = threading.Event()
        failure: list[str] = []
        minimum_available = [available]
        started = time.monotonic()
        started_at_epoch = time.time()
        ready_at = None
        worker = None
        watcher = None
        command = ['systemd-run', '--user', f'--unit={unit}', '--wait', '--pipe', '--collect',
                   f'--property=MemoryMax={CGROUP_CAP}', '--property=MemorySwapMax=0',
                   '--property=KillMode=control-group', '--property=OOMPolicy=kill',
                   '--property=TimeoutStopSec=10', f'--working-directory={working_directory}',
                   '/usr/bin/env', '-i', f'HOME={Path.home()}',
                   f'PATH={root / "venv/bin"}:{CUDA_HOME / "bin"}:/usr/bin:/bin',
                   'LANG=C.UTF-8', 'PYTHONUNBUFFERED=1', 'CUDA_VISIBLE_DEVICES=0',
                   f'HF_HOME={root / "hf-cache"}', 'HF_HUB_OFFLINE=1',
                   f'XDG_CACHE_HOME={root / "runtime-cache"}',
                   f'TRITON_CACHE_DIR={root / "runtime-cache/triton"}',
                   f'TORCHINDUCTOR_CACHE_DIR={root / "runtime-cache/inductor"}',
                   f'CUDA_CACHE_PATH={root / "runtime-cache/cuda"}',
                   f'FLASHINFER_WORKSPACE_BASE={root / "runtime-cache/flashinfer"}',
                   *backend_environment,
                   *server_command]
        initial = {**provenance, 'backend': args.backend, 'weights_bytes': weights, 'initial_memavailable_bytes': available,
                   'required_memavailable_bytes': required, 'cgroup_memory_max_bytes': CGROUP_CAP,
                   'headroom_floor_bytes': HEADROOM, 'abort_threshold_bytes': HEADROOM + STOP_MARGIN,
                   'unit': unit, 'server_command': server_command, 'launch_command': command,
                   'started_at_epoch': started_at_epoch,
                   'admission': 'operator-approved weights + 30 GiB; additional cgroup-cap reserve',
                   'allocator_caps_gib': allocator_caps,
                   'allocation_cap_excludes_driver_and_library_private_memory': True}
        (evidence / 'session.json').write_text(json.dumps(initial, indent=2) + '\n')
        (root / 'launched-at').write_text(f'{started_at_epoch:.6f}\n')
        print(json.dumps(initial), flush=True)

        def request_stop(signum: int, _frame: object) -> None:
            stop.set()

        def watch_memory() -> None:
            try:
                with (evidence / 'memory.jsonl').open('w') as output:
                    while not stop.wait(0.1):
                        current = memory_available()
                        minimum_available[0] = min(minimum_available[0], current)
                        output.write(json.dumps({'elapsed_s': time.monotonic() - started,
                                                 'memavailable_bytes': current}) + '\n')
                        output.flush()
                        if current < HEADROOM + STOP_MARGIN:
                            raise MemoryError(f'MemAvailable {current/GIB:.2f} GiB crossed the 34 GiB abort threshold')
            except Exception as exc:
                failure.append(f'{type(exc).__name__}: {exc}')
                stop.set()
                try:
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all',
                                    '--signal=SIGKILL', unit], check=False, timeout=10)
                except (OSError, subprocess.TimeoutExpired) as kill_error:
                    failure.append(f'Memory-abort unit kill failed: {kill_error}')

        previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with (evidence / 'server.log').open('w') as server_log:
                worker = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
                watcher = threading.Thread(target=watch_memory, daemon=True)
                watcher.start()
                while not stop.is_set() and worker.poll() is None:
                    if ready_at is None:
                        if time.monotonic() - started > args.ready_timeout:
                            failure.append('Readiness deadline exceeded')
                            break
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=1) as response:
                                if response.status == 200:
                                    ready_at = time.monotonic()
                                    initial['cold_ready_s'] = ready_at - started
                                    initial['ready_at_epoch'] = time.time()
                                    (evidence / 'ready.json').write_text(json.dumps(initial, indent=2) + '\n')
                                    print('DRAMA_PILOT_READY', flush=True)
                        except (urllib.error.URLError, TimeoutError):
                            pass
                    stop.wait(0.2)
                if worker.poll() not in (None, 0) and not stop.is_set():
                    failure.append(f'Server exited with status {worker.returncode}')
                elif worker.poll() is not None and ready_at is None and not stop.is_set():
                    failure.append('Server exited before readiness, despite exit status zero')
        except Exception as exc:
            failure.append(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            stop.set()
            if watcher is not None:
                watcher.join(timeout=3)
            # Never release the shared lock on an unknown manager state. Retain it
            # until this exact unit is confirmed inactive, failed, or collected.
            cleanup_failure_recorded = False
            while True:
                try:
                    subprocess.run(['systemctl', '--user', 'stop', unit], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                    state = subprocess.run(
                        ['systemctl', '--user', 'show', unit, '--property=LoadState',
                         '--property=ActiveState'], capture_output=True, text=True,
                        check=False, timeout=10)
                    properties = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
                    stopped = (properties.get('ActiveState') in ('inactive', 'failed')
                               and (state.returncode == 0 or properties.get('LoadState') == 'not-found'))
                    if stopped:
                        break
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if not cleanup_failure_recorded:
                    failure.append('Unit shutdown could not be confirmed; retained GPU lock until confirmed')
                    print(f'Cannot confirm {unit} stopped; retaining GPU lock', flush=True)
                    cleanup_failure_recorded = True
                try:
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all',
                                    '--signal=SIGKILL', unit], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                time.sleep(1)
            if worker is not None:
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            final = {'ready': ready_at is not None, 'elapsed_s': time.monotonic() - started,
                     'minimum_memavailable_bytes': minimum_available[0],
                     'final_memavailable_bytes': memory_available(), 'failures': failure}
            (evidence / 'completion.json').write_text(json.dumps(final, indent=2) + '\n')
            print(json.dumps(final), flush=True)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
