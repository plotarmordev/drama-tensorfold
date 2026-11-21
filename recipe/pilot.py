#!/usr/bin/env python3
"""Run a bounded Linux/systemd pilot without changing another service or GPU policy."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import NamedTuple
import urllib.error
import urllib.request

# python3 -I leaves the script's own directory off sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_selection  # noqa: E402

GIB = 1024**3
MEMINFO = Path('/proc/meminfo')
CUDA_HOME = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))

# A GB10's 128 GiB pool is shared by the host and the device: these are the figures measured for it.
HEADROOM = 30 * GIB
STOP_MARGIN = 4 * GIB
CGROUP_CAP = 54 * GIB
GB10_RESERVE_GIB = 34
# A discrete card keeps the weights in device memory. The driver and its libraries hold 1 GiB of the card, and the
# engine's fit check keeps 0.5 GiB back: issue #68 loaded the 14.5 GB int3 build on a 16 GB card with 0.5 GiB. The
# host pins buffers and caches checkpoint pages, inside a quarter of its RAM (at least 8 GiB), with 4 GiB of
# headroom and a 2 GiB stop margin.
DISCRETE_DRIVER_RESERVE = 1 * GIB
DISCRETE_RESERVE_GIB = 0.5
DISCRETE_CGROUP_MIN = 8 * GIB
DISCRETE_HEADROOM = 4 * GIB
DISCRETE_STOP_MARGIN = 2 * GIB


class Sizing(NamedTuple):
    """Host and device memory figures the launcher admits, caps and reserves against."""
    unified: bool
    cgroup_cap: int
    headroom: int
    admission_floor: int
    abort_below: int
    reserve_gib: float
    cuda_limit_gib: float


GB10_SIZING = Sizing(True, CGROUP_CAP, HEADROOM, CGROUP_CAP + HEADROOM + STOP_MARGIN, HEADROOM + STOP_MARGIN,
                     GB10_RESERVE_GIB, CGROUP_CAP / GIB)


def meminfo(field: str) -> int:
    for line in MEMINFO.read_text().splitlines():
        if line.startswith(f'{field}:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f'{field} is unavailable; refusing an unguarded load')


def memory_available() -> int:
    return meminfo('MemAvailable')


def gpu() -> tuple[str, str]:
    """GPU 0's name and memory.total, as nvidia-smi prints them."""
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits',
                                '-i', '0'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'nvidia-smi is unavailable or did not answer, so the launcher cannot size memory: '
                           f'{exc}') from exc
    name, comma, total = query.stdout.partition('\n')[0].rpartition(',')
    if query.returncode != 0 or not comma:
        raise RuntimeError('nvidia-smi did not report GPU 0\'s name and memory')
    return name.strip(), total.strip()


def sizing() -> Sizing:
    """The machine's memory figures: a GB10's unified pool, or a discrete card's device memory and the host's RAM."""
    name, total = gpu()
    if 'GB10' in name.upper():
        # The GB10's unified pool is sized by the measured GB10_SIZING, not by a device figure: its NVML reports
        # memory.total as [N/A] (Syllo/nvtop#531).
        return GB10_SIZING
    if not total.isdigit():
        raise RuntimeError(f'nvidia-smi did not report {name}\'s device memory')
    device_memory = int(total) * 1024**2
    cuda_limit = device_memory - DISCRETE_DRIVER_RESERVE
    if cuda_limit <= 0:
        raise RuntimeError(f'{name} reports {device_memory / GIB:.2f} GiB of device memory, '
                           'less than the driver reserve')
    cgroup_cap = max(DISCRETE_CGROUP_MIN, meminfo('MemTotal') // 4)
    return Sizing(False, cgroup_cap, DISCRETE_HEADROOM, cgroup_cap + DISCRETE_HEADROOM + DISCRETE_STOP_MARGIN,
                  DISCRETE_HEADROOM + DISCRETE_STOP_MARGIN, DISCRETE_RESERVE_GIB, cuda_limit / GIB)


def guard(weights: int, machine: Sizing, external_admission: bool) -> tuple[int | None, int | None]:
    """MemAvailable required before loading and the abort threshold; neither when the caller admits the load."""
    if external_admission:
        return None, None
    floor = machine.admission_floor
    if machine.unified:
        floor = max(weights + HEADROOM, floor)
    return floor, machine.abort_below


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', choices=tuple(weight_selection.WEIGHTS),
                        help='Download (first run, published weights), verify and serve these pinned weights '
                             f'(default {weight_selection.DEFAULT_WEIGHTS} unless --checkpoint is given) with the '
                             'engine scripts/build-engine.sh --weights builds for them; the NVFP4 quants also '
                             'apply their measured guidance, --cfg-gamma 1.5 --cfg-batched, unless a guidance flag '
                             'follows --')
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
    parser.add_argument('--parent-pid', type=int, default=None,
                        help='Stop the unit, as for SIGTERM, once this process is no longer the parent '
                             '(for a supervisor that cannot set a parent-death signal on the launcher)')
    parser.add_argument('--reserve-gib', type=float, default=None,
                        help='TENSORFOLD_MEMORY_RESERVE_GIB passed to the native server (not a hard cap; default '
                             '34 GiB on a GB10 and 0.5 GiB on a discrete card)')
    parser.add_argument('--external-admission', action='store_true',
                        help='A supervisor that budgets host memory admits this load: skip the MemAvailable '
                             'admission and the abort threshold; the cgroup and allocator caps and the lock stay')
    parser.add_argument('serve_args', nargs='*', metavar='SERVE_ARG',
                        help='Extra tensorfold-native serve flags, after --, e.g. -- --max-frames 2048')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if args.parent_pid is not None and os.getppid() != args.parent_pid:
        parser.error(f'--parent-pid {args.parent_pid} is not this launcher\'s parent')
    if args.reserve_gib is not None and not (math.isfinite(args.reserve_gib) and args.reserve_gib >= 0):
        parser.error('--reserve-gib must be finite and non-negative')
    if args.weights is None and args.checkpoint is None:
        args.weights = weight_selection.DEFAULT_WEIGHTS
    engine = args.weights and weight_selection.WEIGHTS[args.weights]['engine']
    if engine == 'nvfp4' and args.backend != 'native':
        parser.error('Only the native backend serves the NVFP4 weights')
    if args.backend == 'native':
        try:
            machine = sizing()
        except RuntimeError as exc:
            parser.error(str(exc))
        reserve_gib = machine.reserve_gib if args.reserve_gib is None else args.reserve_gib
    else:
        # The reference backend serves on a GB10 only.
        machine = GB10_SIZING
        reserve_gib = args.reserve_gib
    root = args.runtime_dir.resolve()
    checkpoint = weight_selection.checkpoint_path(args.weights, args.checkpoint)
    evidence = args.evidence_dir.resolve()
    if args.backend == 'native':
        if args.native_bin is None and args.weights is None:
            parser.error('The native backend requires --native-bin, or --weights')
        executable = weight_selection.native_path(args.weights, args.native_bin)
        server_command = weight_selection.server_command(
            executable, checkpoint, '127.0.0.1', args.port,
            weight_selection.serve_arguments(args.weights, args.serve_args))
        working_directory = root
        backend_environment = [f'TENSORFOLD_MEMORY_RESERVE_GIB={reserve_gib:g}',
                               f'TENSORFOLD_CUDA_MEMORY_LIMIT_GB={machine.cuda_limit_gib:g}',
                               *weight_selection.native_environment(args.weights)]
        allocator_caps = {'tracked_cuda_and_pinned_host_buffers': machine.cuda_limit_gib}
    elif args.serve_args:
        parser.error('Extra serve flags apply to the native backend only')
    elif args.reserve_gib is not None:
        parser.error('--reserve-gib applies to the native backend only')
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
        hint = f'; build it with scripts/build-engine.sh --weights {engine}' if engine and not args.native_bin else ''
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
        required, abort_below = guard(weights, machine, args.external_admission)
        if required is not None and available < required:
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
        parent_exited = False
        worker = None
        watcher = None
        command = ['systemd-run', '--user', f'--unit={unit}', '--wait', '--pipe', '--collect',
                   f'--property=MemoryMax={machine.cgroup_cap}', '--property=MemorySwapMax=0',
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
        if args.external_admission:
            admission = 'external: the caller admits the load by its own memory budget'
        elif machine.unified:
            admission = 'operator-approved weights + 30 GiB; additional cgroup-cap reserve'
        else:
            admission = 'operator-approved host floor for a discrete card; the weights stay on the device'
        initial = {**provenance, 'backend': args.backend, 'weights_bytes': weights, 'initial_memavailable_bytes': available,
                   'required_memavailable_bytes': required, 'cgroup_memory_max_bytes': machine.cgroup_cap,
                   'headroom_floor_bytes': machine.headroom, 'abort_threshold_bytes': abort_below,
                   'unit': unit, 'server_command': server_command, 'launch_command': command,
                   'started_at_epoch': started_at_epoch, 'parent_pid': args.parent_pid,
                   'admission': admission,
                   'engine_memory_reserve_gib': reserve_gib,
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
                        if abort_below is not None and current < abort_below:
                            raise MemoryError(f'MemAvailable {current/GIB:.2f} GiB crossed the '
                                              f'{abort_below/GIB:.0f} GiB abort threshold')
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
                    if args.parent_pid is not None and os.getppid() != args.parent_pid:
                        parent_exited = True  # the supervisor is gone: a planned stop, not a failure
                        break
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
                     'final_memavailable_bytes': memory_available(), 'parent_exited': parent_exited,
                     'failures': failure}
            (evidence / 'completion.json').write_text(json.dumps(final, indent=2) + '\n')
            print(json.dumps(final), flush=True)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
