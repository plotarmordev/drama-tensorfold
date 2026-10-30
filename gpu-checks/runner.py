#!/usr/bin/env python3
"""Serial device checks; --plan performs no filesystem writes or subprocesses."""
import contextlib
import datetime
import fnmatch
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

W = Path(__file__).resolve().parents[1]
ROOT = W / 'root'
PLAN = '--plan' in sys.argv
DIAG = '--diag' in sys.argv


def _argv_value(flag):
    # Read at import so the output directory and the resume/manifest paths behave like the existing env override.
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


DIAG_RESUME = _argv_value('--diag-resume')
BUILD_MANIFEST = _argv_value('--build-manifest')
OUT = Path(_argv_value('--out-dir') or os.environ.get('GPU_CHECK_OUT', W / 'out-diag' if DIAG else W / 'out'))
HOME = Path.home()


def _setting(name):
    # Host-local configuration comes only from the environment (see README "Device configuration"); an unset
    # value is an explicit <NAME> placeholder, which --plan prints and a device run refuses before any effect.
    return os.environ.get(name) or f'<{name}>'


DEVICE_HOST = _setting('GPU_CHECK_HOST')
LOCK = Path(_setting('GPU_CHECK_LOCK'))
PILOT_ROOT = Path(_setting('GPU_CHECK_PILOT_ROOT'))
CHECKPOINTS = {'full': PILOT_ROOT / 'checkpoint', 'nvfp4': Path(_setting('GPU_CHECK_NVFP4_CHECKPOINT'))}
ZIG = Path(_setting('GPU_CHECK_ZIG'))
# The pilot's Whisper code directory (transcribe.py), read-only; D4 only.
ASR_DIR = Path(_setting('GPU_CHECK_ASR_DIR'))
NVCC = os.environ.get('GPU_CHECK_NVCC') or '/usr/local/cuda/bin/nvcc'
LIB = PILOT_ROOT / 'venv/lib/python3.12/site-packages/nvidia/cu13/lib'
# Units the runner may SIGSTOP for a GPU slot: space-separated exact unit names or fnmatch patterns. Unset means
# none (only the selftest dummy); the runner never pauses anything the owner did not list.
PAUSE_ALLOW = (os.environ.get('GPU_CHECK_PAUSE_ALLOW') or '').split()
# A GPU process listening on this port (an owner-designated service) is tolerated at admission; unset tolerates none.
TOLERATED_PORT = os.environ.get('GPU_CHECK_TOLERATED_PORT')
REQUIRED_SETTINGS = ('GPU_CHECK_HOST', 'GPU_CHECK_LOCK', 'GPU_CHECK_PILOT_ROOT', 'GPU_CHECK_NVFP4_CHECKPOINT', 'GPU_CHECK_ZIG') + (('GPU_CHECK_ASR_DIR',) if DIAG else ())


def device_preflight():
    # Before claim_out or any other effect: every host-local setting present, and this is the configured host.
    missing = [name for name in REQUIRED_SETTINGS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f'device run needs {", ".join(missing)} (README "Device configuration")')
    if os.uname().nodename != DEVICE_HOST:
        raise RuntimeError(f'device runner is restricted to GPU_CHECK_HOST ({DEVICE_HOST})')


SHAS = {'A': 'a7752b4591da9b33bf6e7f8f2b478a638bc64bd1', 'B': '0b2e000b52adb8344f1e922c175680753dc31747'}
# Linux patch series per (tree, profile) build contract; None means unpatched by contract.
PATCH_DIRS = {('A', 'full'): 'engine/patches', ('A', 'nvfp4'): None, ('B', 'full'): 'engine/patches', ('B', 'nvfp4'): 'engine/nvfp4/patches'}
BASE = 'http://127.0.0.1:18795'
TEXT = 'The train leaves at seven fifteen. Please bring the blue notebook.'
REFS = ['The small boat crossed the lake.', 'A bright bird landed near the window.']
MARKUP = ('<whisper>The evening train is nearly here. Please keep this quiet.</whisper> '
          '[laughs] We brought the blue notebook. ¶ 天空很蓝，火车到了。 ' * 6)[:600]
# End the fixture outside a protected span rather than truncating a tag.
MARKUP = MARKUP[:MARKUP.rfind('。') + 1]
MARKUP += ' ' + ('The next train arrives soon. ' * 30)[:600 - len(MARKUP) - 1]
SPEECH = ('We are waiting by the old station. ' * 20)[:400]
DIALOGUE = '<|speaker:1|> Is the train here? <|speaker:15|> It arrives at seven fifteen.'
G4_TEXT = f'<|speaker:1|> {SPEECH} <|speaker:2|> {SPEECH}'
CORPUS = W / 'bench/corpus/drama-bench-en-v1.jsonl'
# The device host receives the corpus only through sync-to-device.sh; the preflight compares this pinned hash.
CORPUS_SHA256 = 'b8995194915f75ffa8860eaca5fc75821d30d209db907d80a07af946ebf59d3b'
# pilot.py's own full-profile admission floor, mirrored instead of discovered by a failed launch; never lowered.
PILOT_ADMISSION_FLOOR = 88 * 2**30
LONG_FORM = ['--long-form', '--long-form-chars', '120']
PILOT = {'log': None}
BUILDS = {}
REUSE = {}
D1REC = {'trees': {}}
D2REC = {}
D3REC = {}
D4REC = {}
DIAG_DEVIATIONS = [
    'The full-profile engine (upstream d1018b3 series) has no long-form mode: those trees hold no speech_long.zig and their CLI refuses --long-form, so D1 runs on the NVFP4 family (both trees); the binaries\' own refusals are recorded in d1.full_profile.',
    'No corpus item is 600-900 characters (the largest is 313), so D1\'s three corpus paragraphs are consecutive items of bench/corpus/drama-bench-en-v1.jsonl joined verbatim with single spaces until the band is reached; file and item ids are recorded.',
    'The server records one whole-reply long_form line (speech_long.zig finish) and no per-segment prompts: every segment before the last ends in stop by the upstream rule `if (r.reason != .stop) break;`, and D2\'s speaker carry is verified by the CPU probe on the candidate\'s actual cuts.',
    'D4 transcribes through the pilot\'s Whisper code (GPU_CHECK_ASR_DIR transcribe.py, read-only, its own venv) and computes WER locally; run-2 wavs under out/ are read, never written.',
    'D3\'s five baseline attempts are unseeded because tree A rejects the seed field.',
    'The runner exits non-zero only on infrastructure errors; every diagnostic observation (early stop, carry failure, WER flag) is recorded, never fatal.',
    'The corpus preflight (pinned sha256) gates the diagnostic run before any build, unit pause, GPU lock or pilot; a missing or mismatched file is an infrastructure failure that writes an INCONCLUSIVE summary and exits non-zero.',
    'Full-profile pilots launch only at or above the 88 GiB MemAvailable floor; below it the check records WAIT/INCONCLUSIVE without launching, without holding the GPU lock and without pausing or killing any scoring process.',
    'A complete verdict requires every D1-D4 check PASS plus actual A and B observations; any failed, WAITed or missing check renders the whole run INCONCLUSIVE, never a comparative phrase or PASS.',
    'Retained builds are reused only when a recorded manifest (--build-manifest, or root/build-manifest.json written by each run) verifies the engine head and both binary sha256s; --diag-resume re-executes only the missing D1-D4 checks into a fresh --out-dir.',
]
COMMANDS = []
RESULTS = []
DEVIATIONS = [
    'ROOT is under the assigned work directory; host-local paths come from the GPU_CHECK_* environment settings.',
    'X-Long-Form-Segments contains audio sample offsets, not text boundaries.',
    'Text cuts outside spans and continuation-speaker carry are checked by the CPU probe only; probe compilation failure is unavailable (not verified), never a pass.',
    'PCM is signed 16-bit integer audio: NaN cannot be represented on the wire. Nonzero RMS is an automatic silence gate, not proof of intelligible speech.',
    'NVIDIA peak memory is unavailable; pilot host-memory samples are reported separately.',
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def shown(text):
    # Printed plans and logs name the work directory as $W, so committed outputs carry no host-local path.
    return text.replace(str(W), '$W')


def log(message):
    print(f'{now()} {shown(str(message))}', flush=True)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def command(argv, cwd=None, capture=False):
    argv = list(map(str, argv))
    text = (f'cd {shlex.quote(str(cwd))} && ' if cwd else '') + shlex.join(argv)
    COMMANDS.append(text)
    if PLAN:
        print(shown(text))
        return ''
    log(text)
    result = subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ''


def memavailable():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024


class MemoryAdmission(RuntimeError):
    """Full-profile pilot floor not met: infrastructure WAIT. No GPU lock is held, nothing launches, and no
    scoring process is paused or killed to make room."""


def allowed(unit):
    return unit == 'gpuchk-selftest-sleep.service' or any(fnmatch.fnmatchcase(unit, pattern) for pattern in PAUSE_ALLOW)


def pause(selftest=False):
    if selftest:
        if os.environ.get('PAUSE_UNITS') != 'gpuchk-selftest-sleep':
            raise RuntimeError('selftest requires the exact dummy unit override')
    else:
        apps = command(['systemd-run', '--user', '--wait', '--pipe', '--quiet', '--', 'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture=True)
        for pid in apps.splitlines():
            argv = Path('/proc') / pid.strip() / 'cmdline'
            args = argv.read_bytes().split(b'\0')
            if TOLERATED_PORT is None or not any(args[i:i+2] == [b'--port', TOLERATED_PORT.encode()] for i in range(len(args))):
                raise RuntimeError(f'foreign GPU process {pid}')
        weights = max(sum(p.stat().st_size for p in root.glob('*.safetensors')) + (root / 'codec.pth').stat().st_size for root in CHECKPOINTS.values())
        if memavailable() < 34 * 2**30 + weights:
            raise RuntimeError('insufficient MemAvailable for largest checkpoint plus reserve')
    override = os.environ.get('PAUSE_UNITS')
    if override:
        units = [x if x.endswith('.service') else x + '.service' for x in override.split()]
    elif PAUSE_ALLOW:
        output = command(['systemctl', '--user', 'list-units', '--state=active', '--no-legend', '--plain', *PAUSE_ALLOW], capture=True)
        units = [line.split()[0] for line in output.splitlines()]
    else:
        units = []
    for unit in units:
        if not allowed(unit):
            raise RuntimeError(f'unit not authorized: {unit}')
        if command(['systemctl', '--user', 'show', '-p', 'ActiveState', '--value', unit], capture=True) != 'active':
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'paused.txt').open('a') as record:
            record.write(f'{unit}\t{now()}\n')
            record.flush()
            os.fsync(record.fileno())
        command(['systemctl', '--user', 'kill', '--kill-whom=all', '-s', 'SIGSTOP', unit])


def resume():
    if not (OUT / 'paused.txt').exists():
        return
    with (OUT / 'resume.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        resumed = OUT / 'resumed.json'
        records = json.loads(resumed.read_text()) if resumed.exists() else {}
        for line in (OUT / 'paused.txt').read_text().splitlines():
            unit, stamp = line.split('\t')
            if line in records:
                continue
            # Only units this runner recorded after its allowlist check reach paused.txt; SIGCONT is harmless, so
            # recovery never depends on GPU_CHECK_PAUSE_ALLOW still being set (a stranded SIGSTOP would be worse).
            if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.service', unit):
                raise RuntimeError(f'malformed paused unit record: {unit}')
            state = command(['systemctl', '--user', 'show', '-p', 'ActiveState', '--value', unit], capture=True)
            if state == 'active':
                command(['systemctl', '--user', 'kill', '--kill-whom=all', '-s', 'SIGCONT', unit])
            records[line] = {'unit': unit, 'paused_at': stamp, 'resumed_at': now(), 'state': state}
            save(resumed, records)
    summary()


def builddir(tree, profile):
    return ROOT / f'build-{tree}-{profile}'


def load_build_manifest():
    # Retained builds are trusted only through recorded hashes: --build-manifest (e.g. a prior summary.json
    # builds block) or the manifest each run writes next to the builds.
    path = Path(BUILD_MANIFEST) if BUILD_MANIFEST else ROOT / 'build-manifest.json'
    if PLAN or not path.exists():
        return {}
    return json.loads(path.read_text()).get('builds') or {}


def build_reusable(build, head, recorded, in_force):
    """head is the build checkout's current HEAD. Reuse requires a manifest in force (--build-manifest or the
    runner-written root/build-manifest.json) whose recorded engine head and both binary sha256s verify against
    the stamped checkout; the stamp alone proves source identity, not binary integrity (run 1's device builds
    predate build-manifest.json), so without one exactly that build is rebuilt."""
    stamp = build / 'gpu-check-head'
    binary = build / 'zig-out/native/bin/tensorfold-native'
    generate = build / 'zig-out/bin/drama3-generate'
    stamp_value = stamp.read_text().strip() if stamp.exists() else None
    if not binary.exists() or stamp_value is None or stamp_value != head:
        return False, 'stamp or native binary missing/stale'
    if not in_force:
        return False, 'no manifest in force; rebuilt rather than trust the stamp alone'
    if recorded is None:
        return False, 'no manifest entry for this build'
    if recorded.get('engine_head') != stamp_value:
        return False, 'engine_head differs from the manifest'
    if recorded.get('tensorfold_native_sha256') != hashlib.sha256(binary.read_bytes()).hexdigest():
        return False, 'tensorfold-native sha256 differs from the manifest'
    wanted = recorded.get('drama3_generate_sha256')
    present = hashlib.sha256(generate.read_bytes()).hexdigest() if generate.exists() else None
    if wanted != present:
        return False, 'drama3-generate sha256 differs from the manifest'
    return True, 'verified against the manifest'


def prep():
    manifest = load_build_manifest()
    if PLAN or not (ROOT / 'B').exists():
        command(['git', 'clone', W / 'pr.bundle', ROOT / 'B'])
        command(['git', 'checkout', '--detach', SHAS['B']], cwd=ROOT / 'B')
    # Run 2's cleanup removed root/A while root/B stayed; restore the worktree without re-cloning.
    if PLAN or not (ROOT / 'A').exists():
        command(['git', 'worktree', 'add', '--detach', ROOT / 'A', SHAS['A']], cwd=ROOT / 'B')
    for tree in SHAS:
        source = ROOT / tree
        head = command(['git', 'rev-parse', 'HEAD'], cwd=source, capture=True)
        if not PLAN and head != SHAS[tree]:
            raise RuntimeError(f'{tree} HEAD mismatch')
        for profile in ('full', 'nvfp4'):
            verify_patches(tree, profile, source)
            build = builddir(tree, profile)
            key = f'{tree}-{profile}'
            stamp = build / 'gpu-check-head'
            binary = build / 'zig-out/native/bin/tensorfold-native'
            if not PLAN and binary.exists() and stamp.exists():
                reusable, reason = build_reusable(build, command(['git', 'rev-parse', 'HEAD'], cwd=build, capture=True),
                                                  manifest.get(key), bool(manifest))
                REUSE[key] = {'decision': 'reused' if reusable else 'rebuilt', 'reason': reason}
                log(f'build {key}: {REUSE[key]["decision"]} ({reason})')
                if reusable:
                    continue
            elif not PLAN:
                REUSE[key] = {'decision': 'rebuilt', 'reason': 'stamp or native binary missing'}
            if not PLAN:
                while memavailable() < 50 * 2**30:
                    log('waiting for build MemAvailable floor')
                    time.sleep(60)
            command(['env', f'ZIG={ZIG}', f'NVCC={NVCC}', 'JOBS=4', 'bash', source / 'scripts/build-engine.sh', '--weights', profile, build])
            if not PLAN:
                stamp.write_text(command(['git', 'rev-parse', 'HEAD'], cwd=build, capture=True) + '\n')
    cpu_probe()


def verify_patches(tree, profile, source):
    patch = PATCH_DIRS[(tree, profile)]
    if patch is None:
        return
    path = source / patch
    if not PLAN and not path.is_dir():
        raise RuntimeError(f'{tree}/{profile} patch series missing: {path}')
    manifest = path / 'SHA256SUMS'
    # Same rule as scripts/build-engine.sh: an empty full-model manifest declares an empty series, which
    # `sha256sum -c` would reject. Any other profile's empty manifest still goes to the check and fails.
    if profile == 'full' and manifest.is_file() and manifest.stat().st_size == 0 and not any(path.glob('*.patch')):
        log(f'{tree}/{profile} patch series is empty by manifest; no checksums to verify')
        return
    command(['sha256sum', '-c', 'SHA256SUMS'], cwd=path)


def patch_selftest():
    base = Path(tempfile.mkdtemp(prefix='gpuchk-patches-'))

    def fixture(tree, patch):
        path = base / tree / patch
        path.mkdir(parents=True, exist_ok=True)
        (path / '0001-fixture.patch').write_text(f'fixture series for {tree} {patch}\n')
        (path / 'SHA256SUMS').write_text(hashlib.sha256((path / '0001-fixture.patch').read_bytes()).hexdigest() + '  0001-fixture.patch\n')

    for tree, patch in (('A', 'engine/patches'), ('B', 'engine/patches'), ('B', 'engine/nvfp4/patches')):
        fixture(tree, patch)
    mark = len(COMMANDS)
    verify_patches('A', 'nvfp4', base / 'A')
    if any('sha256sum' in x for x in COMMANDS[mark:]):
        raise RuntimeError('A/nvfp4 is unpatched by contract but ran a checksum command')
    verify_patches('B', 'nvfp4', base / 'B')
    if not any('sha256sum' in x for x in COMMANDS[mark:]):
        raise RuntimeError('B/nvfp4 patch series was not checksum-verified')
    for tree, profile in (('B', 'nvfp4'), ('A', 'full'), ('B', 'full')):
        patch = PATCH_DIRS[(tree, profile)]
        shutil.rmtree(base / tree / patch)
        try:
            verify_patches(tree, profile, base / tree)
        except RuntimeError:
            fixture(tree, patch)
        else:
            raise RuntimeError(f'{tree}/{profile} accepted a missing patch series')
    (base / 'A/engine/patches/0001-fixture.patch').write_text('corrupted\n')
    try:
        verify_patches('A', 'full', base / 'A')
    except subprocess.CalledProcessError:
        pass
    else:
        raise RuntimeError('A/full accepted a checksum mismatch')
    # An empty full manifest with no patches is an empty series and skips the check; NVFP4 never does.
    (base / 'A/engine/patches/0001-fixture.patch').unlink()
    (base / 'A/engine/patches/SHA256SUMS').write_text('')
    mark = len(COMMANDS)
    verify_patches('A', 'full', base / 'A')
    if any('sha256sum' in x for x in COMMANDS[mark:]):
        raise RuntimeError('A/full empty series ran a checksum command')
    (base / 'B/engine/nvfp4/patches/0001-fixture.patch').unlink()
    (base / 'B/engine/nvfp4/patches/SHA256SUMS').write_text('')
    try:
        verify_patches('B', 'nvfp4', base / 'B')
    except subprocess.CalledProcessError:
        pass
    else:
        raise RuntimeError('B/nvfp4 accepted an empty manifest')
    shutil.rmtree(base)
    log('patch selftest PASS')


def cpu_probe():
    build = builddir('B', 'nvfp4')
    target = build / 'zig/src/server/gpuchk_probe.zig'
    argv = [ZIG, 'test', '-lc', '--test-filter', 'gpuchk',
            '--dep', 'tokenizer', f'-Mroot={target}',
            f'-Mtokenizer={build}/zig/src/core/tokenizer/tokenizer.zig']
    if PLAN:
        command(argv)
        return
    template = (W / 'gpu-checks/probe.zig.in').read_text()
    # Byte escapes keep the Zig literal identical to the HTTP UTF-8 payload.
    literal = lambda text: '"' + ''.join(f'\\x{x:02x}' for x in text.encode()) + '"'
    source = template.replace('@MARKUP@', literal(MARKUP)).replace(
        '@DIALOGUE@', literal(f'<|speaker:1|> {SPEECH} <|speaker:2|> {SPEECH}'))
    report = {'status': 'unavailable (not verified)', 'start': now()}
    try:
        with target.open('x') as output:
            output.write(source)
        try:
            report['stdout'] = command(argv, capture=True)
            report['status'] = 'PASS'
        except subprocess.CalledProcessError as error:
            report['stderr'] = error.stderr
            if 'FAIL' in (error.stderr or '') or 'tests failed' in (error.stderr or ''):
                report['status'] = 'FAIL'
    finally:
        if target.exists() and target.read_text() == source:
            target.unlink()
        report['end'] = now()
        save(OUT / 'cpu-probe.json', report)


def admit_full_profile():
    # Admission precedes every full-profile Popen and any filesystem effect of that attempt: no GPU lock is held
    # here and no scoring process is paused or killed to make room (run 1 died on this pilot's own refusal).
    available = memavailable()
    if available < PILOT_ADMISSION_FLOOR:
        raise MemoryAdmission(f'MemAvailable {available / 2**30:.2f} GiB is below the full-profile pilot '
                              f'floor {PILOT_ADMISSION_FLOOR / 2**30:.2f} GiB; pilot not launched')


@contextlib.contextmanager
def server(tree, profile, run, flags=()):
    pilot = None
    attempt = 0
    try:
        while True:
            attempt += 1
            evidence = run / f'pilot-{attempt}'
            argv = ['python3', ROOT / tree / 'recipe/pilot.py', '--weights', profile, '--checkpoint', CHECKPOINTS[profile], '--native-bin', builddir(tree, profile) / 'zig-out/native/bin/tensorfold-native', '--library-dir', LIB, '--runtime-dir', ROOT / f'runtime-{tree}-{profile}', '--gpu-lock', LOCK, '--evidence-dir', evidence, '--port', '18795']
            if flags:
                argv += ['--', *flags]
            text = shlex.join(list(map(str, argv)))
            if PLAN:
                COMMANDS.append(text)
                print(shown(text))
                break
            if profile == 'full':
                # Rechecked before every attempt, including after a lost lock race: memory may have dropped while waiting.
                admit_full_profile()
            COMMANDS.append(text)
            run.mkdir(parents=True, exist_ok=True)
            (ROOT / f'runtime-{tree}-{profile}').mkdir(parents=True, exist_ok=True)
            logfile = run / f'pilot-{attempt}.out'
            log(text)
            with logfile.open('w') as output:
                pilot = subprocess.Popen(list(map(str, argv)), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic()
            while pilot.poll() is None and not (evidence / 'ready.json').exists() and time.monotonic() - began < 950:
                time.sleep(.5)
            if (evidence / 'ready.json').exists() and pilot.poll() is None:
                break
            stop(pilot)
            pilot = None
            if 'Another workload owns the GPU slot' in logfile.read_text():
                log('pilot lost shared lock race; waiting and retrying same start')
                command(['flock', LOCK, 'true'])
                continue
            raise RuntimeError(f'pilot not ready: {logfile}')
        PILOT['log'] = None if PLAN else logfile
        yield
    finally:
        PILOT['log'] = None
        if pilot is not None:
            stop(pilot)


def stop(pilot):
    if pilot.poll() is None:
        pilot.send_signal(signal.SIGINT)
        # The pilot retains its lock until its exact server unit is confirmed stopped.
        pilot.wait()


def request(run, label, text=TEXT, expected=200, record_only=False, **fields):
    body = {'model': 'fishaudio/drama-3', 'input': text, 'response_format': 'pcm', **fields}
    payload = json.dumps(body, ensure_ascii=False)
    cmd = ['curl', '--silent', '--show-error', '--max-time', '900', '-H', 'Content-Type: application/json', '--data-binary', payload, BASE + '/v1/audio/speech']
    COMMANDS.append(shlex.join(cmd))
    if PLAN:
        print(shown(shlex.join(cmd)))
        return b'', {}
    run.mkdir(parents=True, exist_ok=True)
    save(run / f'{label}-request.json', body)
    started = time.monotonic()
    req = urllib.request.Request(BASE + '/v1/audio/speech', payload.encode(), {'Content-Type': 'application/json'})
    try:
        reply = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as error:
        reply = error
    with reply:
        data = reply.read()
        headers = dict(reply.headers.items())
        status = reply.status
    info = {'status': status, 'headers': headers, 'latency_s': time.monotonic() - started, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if status == 200:
        path = run / f'{label}.pcm'
        path.write_bytes(data)
        info['pcm'] = str(path)
        if data and len(data) % 2 == 0:
            samples = [v[0] for v in struct.iter_unpack('<h', data)]
            info['duration_s'] = len(samples) / 44100
            info['rms'] = (sum(x*x for x in samples) / len(samples)) ** .5
            wav = path.with_suffix('.wav')
            with wave.open(str(wav), 'wb') as audio:
                audio.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                audio.writeframes(data)
            info['wav'] = str(wav)
            info['wav_sha256'] = hashlib.sha256(wav.read_bytes()).hexdigest()
    else:
        info['error'] = data.decode(errors='replace')
    save(run / f'{label}.json', info)
    if record_only:
        # Diagnostic mode: every reply is evidence; only a transport failure raises.
        return data, headers
    if status != expected:
        raise RuntimeError(f'{label}: HTTP {status}, expected {expected}')
    if expected == 200 and (not data or len(data) % 2 or info.get('rms', 0) == 0 or not headers.get('Content-Type', '').startswith('audio/pcm')):
        raise RuntimeError(f'{label}: invalid or silent PCM')
    return data, headers


def segments(data, headers):
    if PLAN:
        return
    header = headers.get('X-Long-Form-Segments', '')
    count, sep, values = header.partition(';')
    try:
        count = int(count)
    except ValueError:
        count = None
    if not sep or count is None or count < 1:
        raise RuntimeError(f'invalid long-form segment count in X-Long-Form-Segments {header!r}')
    if count == 1:
        # The G4 outcome: a one-segment reply carries no cuts, so the check fails with its real reason.
        raise RuntimeError(f'long-form reply spoke 1 segment, so there are no segment cuts (X-Long-Form-Segments {header!r})')
    try:
        offsets = [int(x) for x in values.split(',')]
    except ValueError:
        offsets = None
    if offsets is None or len(offsets) != count - 1 or offsets != sorted(set(offsets)) or not all(0 < x < len(data)//2 for x in offsets):
        raise RuntimeError(f'invalid long-form segment sample offsets in X-Long-Form-Segments {header!r}')


def segments_selftest():
    # segments() no-ops under --plan; this selftest must pass regardless of argv.
    global PLAN
    PLAN = False

    def outcome(data, headers):
        try:
            segments(data, headers)
        except RuntimeError as error:
            return str(error)
        except Exception as error:
            raise RuntimeError(f'segments let {type(error).__name__} escape: {error}') from None
        return ''

    # None header means the header is absent; fragment is the required rejection reason, None to accept.
    cases = [
        (1110016, '1;', 'spoke 1 segment'),
        (3794526, '7;507904,592059,841915,921062,1261187,1346351', None),
        (1000, '3;100,200', None),
        (1000, None, ''),
        (1000, '', ''),
        (1000, '3', ''),
        (1000, 'x;1,2', ''),
        (1000, '3;100,abc', ''),
        (1000, '3;100,', ''),
        (1000, '3;200,100', ''),
        (1000, '3;100,100', ''),
        (1000, '3;100', ''),
        (1000, '3;100,500', ''),
        (1000, '3;0,100', ''),
        (1000, '2;100;200', ''),
    ]
    for size, value, fragment in cases:
        headers = {} if value is None else {'X-Long-Form-Segments': value}
        message = outcome(bytes(size), headers)
        if fragment is None and message:
            raise RuntimeError(f'segments rejected a valid header {value!r}: {message}')
        if fragment is not None and (not message or fragment not in message):
            raise RuntimeError(f'segments did not reject {value!r} as expected: {message or "accepted"}')
    log('segments selftest PASS')


def bench(name, *args):
    command(['python3', ROOT / 'B' / 'bench' / name, *args])


def catalog(profile):
    return OUT / 'synthetic' / profile / 'catalog'


def voices(profile):
    run = OUT / 'synthetic' / profile
    if not PLAN and (run / 'complete.json').exists():
        return
    with server('B', profile, run / 'render'):
        for i, text in enumerate(REFS):
            request(run, f'ref-{i}', text, seed=42 + i)
    for i in range(len(REFS)):
        command(['flock', LOCK, 'env', f'LD_LIBRARY_PATH={LIB}', builddir('B', 'full') / 'zig-out/bin/drama3-generate', '--codec', CHECKPOINTS[profile] / 'codec.pth', '--encode', run / f'ref-{i}.codes', '--ref-audio', run / f'ref-{i}.wav'])
    for name, refs in [('A', [(0, REFS[0])]), ('B', [(1, REFS[1])]), ('dialogue', [(0, '<|speaker:1|> ' + REFS[0]), (1, '<|speaker:15|> ' + REFS[1])])]:
        target = catalog(profile) if profile == 'full' else run / 'http-catalog'
        command(['python3', ROOT / 'B' / 'recipe/build_voice.py', '--source', 'synthetic', '--output-dir', target, name, 'Model-invented clips rendered by candidate built-in voice; request evidence retained.', *[f'{text}:{run}/ref-{i}.codes' for i, text in refs]])
        if not PLAN and profile == 'nvfp4':
            doc = json.loads((target / f'{name}.json').read_text())
            entries = []
            for ref, (i, _) in zip(doc['references'], refs):
                grid = list(map(list, zip(*ref['codes'])))
                raw = b''.join(struct.pack('<i', x) for row in grid for x in row)
                entries.append({'text': ref['text'], 'vq_codes': grid, 'codes_sha256': hashlib.sha256(raw).hexdigest(), 'audio_sha256': hashlib.sha256((run / f'ref-{i}.wav').read_bytes()).hexdigest(), 'frames': len(ref['codes'])})
            source = {'kind': 'model-invented', 'note': doc['source']['note']}
            legacy = {'format': 'drama3-voice/2', 'source': source, 'references': entries} if len(entries) > 1 else {'format': 'drama3-voice/1', 'source': source, **entries[0]}
            save(catalog(profile) / f'{name}.json', legacy)
    if not PLAN:
        save(run / 'complete.json', {'finished_at': now()})


def g1(profile):
    run = OUT / 'G1' / profile
    for tree in SHAS:
        with server(tree, profile, run / tree / 'default'):
            bench('corpus_capture.py', BASE, run / tree / 'capture', '--passes', '2')
            bench('stream_check.py', '--base', BASE, '--model', 'fishaudio/drama-3', '--input', TEXT, '--out', run / tree / 'stream-check.json')
    bench('compare_gate.py', run / 'A/capture', run / 'A/capture', run / 'B/capture', run / 'gate-B.json')
    voices(profile)
    # Baseline rejects the dialogue catalog's new speaker tags; load untagged references only.
    untagged = OUT / 'synthetic' / profile / 'untagged'
    if not PLAN:
        untagged.mkdir(exist_ok=True)
        for name in ('A', 'B'):
            (untagged / f'{name}.json').write_bytes((catalog(profile) / f'{name}.json').read_bytes())
    for tree in SHAS:
        with server(tree, profile, run / tree / 'reference', ['--voices', untagged]):
            bench('reference_cache_test.py', '--base', BASE, '--voice', 'A', '--other-voice', 'B', '--out', run / tree / 'rc.json')


def g2(profile):
    run = OUT / 'G2' / profile
    with server('B', profile, run):
        for seed in (42, 9007199254740993, 18446744073709551615):
            a, _ = request(run, f'seed-{seed}-first', seed=seed)
            b, _ = request(run, f'seed-{seed}-repeat', seed=seed)
            if a != b:
                raise RuntimeError(f'non-repeatable seed {seed}')
        request(run, 'temperature-zero', temperature=0)
        request(run, 'top-k-one', top_k=1)
        request(run, 'sampling', temperature=.7, top_p=.8, top_k=20)
        request(run, 'invalid-top-k', 'hi', expected=400, top_k=65)


def g3(pause_s):
    run = OUT / 'G3' / f'pause-{pause_s}'
    flags = ['--long-form', '--long-form-chars', '120']
    if pause_s:
        flags += ['--shift-pause-s', str(pause_s)]
    with server('B', 'nvfp4', run, flags):
        segments(*request(run, 'markup', MARKUP))
    probe_gate()


def g4():
    run = OUT / 'G4'
    with server('B', 'nvfp4', run, ['--long-form', '--long-form-chars', '120']):
        segments(*request(run, 'speakers', f'<|speaker:1|> {SPEECH} <|speaker:2|> {SPEECH}', seed=7))
    probe_gate()


def probe_gate():
    if not PLAN and json.loads((OUT / 'cpu-probe.json').read_text())['status'] == 'FAIL':
        raise RuntimeError('CPU splitter/continuation probe failed; see cpu-probe.json')


def g5(profile):
    voices(profile)
    run = OUT / 'G5' / profile
    with server('B', profile, run, ['--voices', catalog(profile)]):
        for voice in ('default', 'dialogue'):
            request(run, f'{voice}-tagged', DIALOGUE, voice=voice)
            request(run, f'{voice}-untagged', re.sub(r'<\|speaker:\d+\|>', '', DIALOGUE), voice=voice)


def tag_free(text):
    return re.sub(r'<\|speaker:\d+\|>', '', text)


def plain(text):
    # The engine's controls removed: speaker tags, paired direction tags, bracketed cues and the shift mark.
    return ' '.join(re.sub(r'<\|speaker:\d+\|>|</?whisper>|\[[^\]]*\]|¶', ' ', text).split())


def corpus_paragraphs(count=3):
    # The corpus holds no 600-900 character item (its largest is 313), so a paragraph is consecutive verbatim items
    # joined with single spaces until the band is reached; the pick is file-order deterministic and ids are recorded.
    items = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    groups, current = [], []
    for item in items:
        current.append(item)
        if len(' '.join(x['text'] for x in current)) >= 600:
            groups.append(current)
            current = []
    if len(groups) < count:
        raise RuntimeError(f'corpus yields only {len(groups)} paragraphs of at least 600 characters')
    paragraphs = []
    for group in groups[:count]:
        text = ' '.join(x['text'] for x in group)
        if not 600 <= len(text) <= 900:
            raise RuntimeError(f'corpus paragraph {group[0]["id"]}.. is {len(text)} characters, outside 600-900')
        paragraphs.append({'text': text, 'ids': [x['id'] for x in group], 'item_chars': [len(x['text']) for x in group]})
    return paragraphs


def corpus_preflight(path=CORPUS, pinned=CORPUS_SHA256):
    """First gate of a diagnostic run: the device host receives the corpus only via sync-to-device.sh, so verify the
    pinned sha256 before any build, unit pause, GPU lock or pilot (run 1 lost D1/D2/D4 to its absence)."""
    if not Path(path).exists():
        raise RuntimeError(f'corpus missing: {path} (sync-to-device.sh copies it; pinned sha256 {pinned})')
    if PLAN:
        command(['sha256sum', path])
        return None
    digest = command(['sha256sum', path], capture=True).split()[0]
    if digest != pinned:
        raise RuntimeError(f'corpus sha256 mismatch: {digest} != pinned {pinned}: {path}')
    return digest


def d1_inputs():
    paragraphs = corpus_paragraphs()
    D1REC['inputs'] = {
        'corpus_file': str(CORPUS),
        'fixed': [{'label': 'g4-plain', 'source': 'G4 text with both speaker tags removed', 'chars': len(tag_free(G4_TEXT))},
                  {'label': 'g3-plain', 'source': 'G3 markup text with its markup removed', 'chars': len(plain(MARKUP))}],
        'corpus': [{'label': f'corpus-{i + 1}', 'ids': p['ids'], 'item_chars': p['item_chars'], 'chars': len(p['text'])}
                   for i, p in enumerate(paragraphs)],
    }
    return ([('g4-plain', tag_free(G4_TEXT)), ('g3-plain', plain(MARKUP))]
            + [(f'corpus-{i + 1}', p['text']) for i, p in enumerate(paragraphs)])


def asr_words(text):
    # Lowercase, punctuation and markup stripped; numbers stay exactly as written in the input.
    stripped = re.sub(r'<\|speaker:\d+\|>|</?whisper>|\[[^\]]*\]|¶', ' ', text.lower())
    return [w for w in (''.join(c for c in word if c.isalnum() or c == "'") for word in stripped.split()) if w]


def word_error_rate(reference, hypothesis):
    r, h = asr_words(reference), asr_words(hypothesis)
    if not r:
        return 1.0 if h else 0.0
    row = list(range(len(h) + 1))
    for i, ref in enumerate(r, 1):
        following = [i]
        for j, hyp in enumerate(h, 1):
            following.append(min(row[j] + 1, following[j - 1] + 1, row[j - 1] + (ref != hyp)))
        row = following
    return row[-1] / len(r)


def pilot_new(offset):
    if PILOT['log'] is None or not PILOT['log'].exists():
        return ''
    return PILOT['log'].read_text()[offset:]


def diag_request(run, label, text, **fields):
    # Every reply is a recorded outcome: status, headers, PCM, wav, hashes and the server's own log lines.
    offset = PILOT['log'].stat().st_size if PILOT['log'] is not None and PILOT['log'].exists() else 0
    request(run, label, text, record_only=True, **fields)
    if PLAN:
        return {}
    info = json.loads((run / f'{label}.json').read_text())
    fresh = pilot_new(offset)
    info.update(label=label, input_chars=len(text), server_log=fresh.splitlines()[-3:])
    spoken = re.search(r'long_form segments=(\d+)/(\d+)', fresh)
    info['segments_spoken_log'], info['segments_planned_log'] = (int(spoken.group(1)), int(spoken.group(2))) if spoken else (None, None)
    finish = re.search(r'finish=(\w+)', fresh)
    info['finish_log'] = finish.group(1) if finish else None
    header = info['headers'].get('X-Long-Form-Segments', '')
    info['segments_header'] = header
    count = header.partition(';')[0]
    info['segments_spoken_header'] = int(count) if count.isdigit() else None
    info['finish_header'] = info['headers'].get('X-Finish-Reason')
    info['segments_spoken'] = info['segments_spoken_log'] if info['segments_spoken_log'] is not None else info['segments_spoken_header']
    info['early_stop'] = None if None in (info['segments_spoken'], info['segments_planned_log']) else info['segments_spoken'] < info['segments_planned_log']
    save(run / f'{label}.json', info)
    return info


def d1_full_profile_probes():
    # D1 as specified asks for full-profile long-form servers, but the full-profile engine has no long-form mode;
    # the binaries' own refusals are the evidence, and a surprise acceptance fails the check instead of narrowing it.
    rows = []
    for tree in ('A', 'B'):
        argv = [builddir(tree, 'full') / 'zig-out/native/bin/tensorfold-native', 'serve', *LONG_FORM]
        text = shlex.join(list(map(str, argv)))
        COMMANDS.append(text)
        if PLAN:
            print(shown(text))
            continue
        done = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=300)
        output = (done.stdout + done.stderr).strip()
        rows.append({'tree': tree, 'exit': done.returncode, 'output': output[-400:]})
        refused = ('unrecognized' in output or 'not served by the native engine' in output) and '--long-form' in output
        if not refused:
            raise RuntimeError(f'{tree}/full did not refuse --long-form: exit {done.returncode}: {output[-200:]}')
    D1REC['full_profile'] = {'note': 'the full-profile engine has no long-form mode; both binaries refused --long-form at argument parsing', 'probes': rows}


def d1():
    run = OUT / 'D1'
    D1REC['start'] = now()
    d1_full_profile_probes()
    inputs = d1_inputs()
    with server('A', 'nvfp4', run / 'A' / 'nvfp4', LONG_FORM):
        D1REC['trees']['A-nvfp4'] = [diag_request(run / 'A' / 'nvfp4', label, text) for label, text in inputs]
    D1REC['end'] = now()


def d2():
    run = OUT / 'D2'
    D2REC['start'] = now()
    inputs = d1_inputs()
    with server('B', 'nvfp4', run, LONG_FORM):
        # One server per tree x profile x mode: D1's B column and D2's seeds share the B nvfp4 long-form server.
        D1REC['b_start'] = now()
        D1REC['trees']['B-nvfp4'] = [diag_request(OUT / 'D1' / 'B' / 'nvfp4', label, text) for label, text in inputs]
        D1REC['b_end'] = now()
        D2REC['seeds'] = [diag_request(run, f'seed-{seed}', G4_TEXT, seed=seed) for seed in range(1, 9)]
    D2REC['end'] = now()
    reached = [row for row in D2REC['seeds'] if (row.get('segments_spoken') or 0) >= 2]
    D2REC['reached_segment_2'] = len(reached)
    D2REC['fraction_reaching_segment_2'] = len(reached) / len(D2REC['seeds'])
    carry_probe()


def carry_probe():
    # The server log carries no per-segment prompt or speaker, so carry follows the CPU probe on the candidate's
    # actual cuts for the G4 text; seeds move sampling only, never the text cuts, so one probe covers every reply.
    build = builddir('B', 'nvfp4')
    target = build / 'zig/src/server/gpuchk_carry_probe.zig'
    argv = [ZIG, 'test', '-lc', '--test-filter', 'gpuchk carry',
            '--dep', 'tokenizer', f'-Mroot={target}', f'-Mtokenizer={build}/zig/src/core/tokenizer/tokenizer.zig']
    if PLAN:
        command(argv)
        return
    template = (W / 'gpu-checks/probe-carry.zig.in').read_text()
    # Byte escapes keep the Zig literal identical to the HTTP UTF-8 payload.
    literal = lambda text: '"' + ''.join(f'\\x{x:02x}' for x in text.encode()) + '"'
    source = template.replace('@DIALOGUE@', literal(G4_TEXT))
    report = {'method': 'cpu-probe (drama_shift.split at 120 chars, drama_prompt.continued per segment, build-B-nvfp4 sources)',
              'why_not_server_log': 'speech_long.zig logs one whole-reply long_form line per reply and no continuation prompts',
              'start': now()}
    try:
        with target.open('x') as output:
            output.write(source)
        done = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=900)
        text = done.stdout + done.stderr
        report.update(exit=done.returncode, output=text[-4000:])
        segments = []
        for line in text.splitlines():
            hit = re.search(r'gpuchk-carry segment=(\d+) start=(\d+) len=(\d+) expected=(<\|speaker:\d+\|>) ok=(true|false)', line)
            if hit:
                segments.append({'segment': int(hit.group(1)), 'start': int(hit.group(2)), 'len': int(hit.group(3)),
                                 'expected': hit.group(4), 'ok': hit.group(5) == 'true'})
        report['segments'] = segments
        report['status'] = 'PASS' if segments and done.returncode == 0 and all(s['ok'] for s in segments) else ('FAIL' if segments else 'unavailable (not verified)')
        report['per_seed'] = {}
        for row in D2REC['seeds']:
            n = row.get('segments_spoken') or 0
            checked = [s for s in segments if 2 <= s['segment'] <= n]
            report['per_seed'][row['label']] = {'segments_spoken': n, 'carry_segments': [s['segment'] for s in checked],
                                                'carry_ok': bool(checked) and all(s['ok'] for s in checked)}
    finally:
        if target.exists() and target.read_text() == source:
            target.unlink()
        report['end'] = now()
    D2REC['carry'] = report


def d3():
    run = OUT / 'D3'
    D3REC['start'] = now()
    voices('full')
    D3REC['groups'] = {}
    with server('B', 'full', run / 'B', ['--voices', catalog('full')]):
        for kind, text in (('tagged', DIALOGUE), ('untagged', tag_free(DIALOGUE))):
            rows = [diag_request(run / 'B', f'{kind}-seed-{seed}', text, voice='dialogue', seed=seed) for seed in range(1, 6)]
            D3REC['groups'][f'B-full-dialogue-{kind}-seed-1-5'] = d3_group(rows, True)
    # A rejects the seed field and speaker tags: five unseeded default-field attempts on the untagged line.
    with server('A', 'full', run / 'A'):
        rows = [diag_request(run / 'A', f'untagged-unseeded-{attempt}', tag_free(DIALOGUE)) for attempt in range(1, 6)]
        D3REC['groups']['A-full-default-untagged-unseeded-1-5'] = d3_group(rows, False)
    D3REC['end'] = now()


def d3_group(rows, seeded):
    detail = [{'label': r.get('label'), 'status': r.get('status'), 'finish': r.get('finish_header'), 'bytes': r.get('bytes'),
               'duration_s': r.get('duration_s'), 'rms': r.get('rms'), 'sha256': r.get('sha256'),
               'capped': r.get('finish_header') == 'length' or r.get('bytes') == 4194304} for r in rows]
    return {'seeded': seeded, 'replies': len(detail), 'finish_length': sum(1 for r in detail if r['finish'] == 'length'),
            'bytes_cap': sum(1 for r in detail if r['bytes'] == 4194304), 'capped': sum(1 for r in detail if r['capped']),
            'rows': detail}


def d4_jobs():
    jobs = []
    seen = set()

    def add(identifier, wav, reference):
        # A --diag-resume run reads wavs from both this run's out dir and the prior one; current evidence wins.
        if identifier in seen:
            return
        seen.add(identifier)
        jobs.append({'id': identifier, 'wav': str(wav), 'reference': reference, 'exists': Path(wav).exists() if not PLAN else None})

    # Run 2's evidence under out/ is read-only here.
    run2 = W / 'out'
    add('run2-G3-pause-0', run2 / 'G3/pause-0/markup.wav', plain(MARKUP))
    add('run2-G3-pause-0.5', run2 / 'G3/pause-0.5/markup.wav', plain(MARKUP))
    add('run2-G4-speakers', run2 / 'G4/speakers.wav', tag_free(G4_TEXT))
    for profile in ('full', 'nvfp4'):
        for voice in ('default', 'dialogue'):
            for kind in ('tagged', 'untagged'):
                add(f'run2-G5-{profile}-{voice}-{kind}', run2 / 'G5' / profile / f'{voice}-{kind}.wav', tag_free(DIALOGUE))
    add('run2-G2-full-temperature-zero', run2 / 'G2/full/temperature-zero.wav', TEXT)
    add('run2-G2-full-top-k-one', run2 / 'G2/full/top-k-one.wav', TEXT)
    texts = dict(d1_inputs())
    roots = [OUT] + ([Path(DIAG_RESUME).resolve().parent] if DIAG_RESUME else [])
    for root in roots:
        for wav in sorted(root.glob('D1/*/*/*.wav')):
            add(f'd1-{wav.parent.parent.name}-{wav.parent.name}-{wav.stem}', wav, texts.get(wav.stem))
        for wav in sorted(root.glob('D2/*.wav')):
            add(f'd2-{wav.stem}', wav, tag_free(G4_TEXT))
        for wav in sorted(root.glob('D3/*/*.wav')):
            add(f'd3-{wav.parent.name}-{wav.stem}', wav, tag_free(DIALOGUE))
    return [j for j in jobs if j['reference'] is not None]


def wav_measure(path):
    with wave.open(str(path), 'rb') as audio:
        data = audio.readframes(audio.getnframes())
    samples = [v[0] for v in struct.iter_unpack('<h', data)]
    return {'duration_s': len(samples) / 44100, 'rms': (sum(x * x for x in samples) / len(samples)) ** .5 if samples else 0}


def d4():
    run = OUT / 'D4'
    D4REC['start'] = now()
    jobs = d4_jobs()
    D4REC['inputs'] = jobs
    manifest, result = run / 'input.json', run / 'asr.json'
    if not PLAN:
        save(manifest, {'asr_dir': str(ASR_DIR),
                        'jobs': [{'id': j['id'], 'wav': j['wav']} for j in jobs if j['exists']]})
    # Whisper runs on the GPU, so it takes the shared lock like the pilots do.
    command(['flock', LOCK, PILOT_ROOT / 'venv/bin/python', W / 'gpu-checks/asr.py', manifest, result])
    if not PLAN:
        asr = json.loads(result.read_text())
        D4REC.update(model=asr['model'], device=asr['device'])
        rows = []
        for job in jobs:
            row = {'id': job['id'], 'wav': job['wav'], 'reference': job['reference']}
            if not job['exists']:
                row['note'] = 'wav absent'
                rows.append(row)
                continue
            row.update(wav_measure(job['wav']))
            transcript = asr['transcripts'].get(job['id'])
            row['transcript'] = transcript
            row['wer'] = word_error_rate(job['reference'], transcript) if transcript is not None else None
            row['flags'] = [name for name, on in (('wer>0.3', row['wer'] is not None and row['wer'] > .3),
                                                  ('rms<500', row['rms'] is not None and row['rms'] < 500)) if on]
            rows.append(row)
        D4REC['rows'] = rows
        D4REC['flagged'] = [row['id'] for row in rows if row.get('flags')]
    D4REC['end'] = now()


def record_builds():
    if PLAN:
        return
    for tree in SHAS:
        for profile in ('full', 'nvfp4'):
            build = builddir(tree, profile)
            native = build / 'zig-out/native/bin/tensorfold-native'
            generate = build / 'zig-out/bin/drama3-generate'
            stamp = build / 'gpu-check-head'
            BUILDS[f'{tree}-{profile}'] = {
                'engine_head': stamp.read_text().strip() if stamp.exists() else None,
                'patch_series': PATCH_DIRS[(tree, profile)],
                'tensorfold_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest() if native.exists() else None,
                'drama3_generate_sha256': hashlib.sha256(generate.read_bytes()).hexdigest() if generate.exists() else None,
            }
    # Written next to the builds so the next run verifies retained-build reuse against recorded hashes.
    save(ROOT / 'build-manifest.json', {'builds': BUILDS})


def num(value):
    return f'{value:.2f}' if isinstance(value, (int, float)) else str(value)


DIAG_GROUPS = ('B-full-dialogue-tagged-seed-1-5', 'B-full-dialogue-untagged-seed-1-5', 'A-full-default-untagged-unseeded-1-5')


def _count(value, low=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def _real(value, low=0.0, strict=False):
    # A finite, non-bool number at or above low (strictly above when strict).
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value > low if strict else value >= low


def _dict(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    # Only dict rows survive; anything else in a carried payload is ignored by the renderer.
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _reply_ok(row):
    # A real reply as request() records it: HTTP 200, a nonempty even-length PCM body and its measurements.
    return (isinstance(row, dict) and row.get('status') == 200 and _count(row.get('bytes'), 1) and row['bytes'] % 2 == 0
            and _real(row.get('duration_s'), strict=True) and _real(row.get('rms')))


def d1_observation(row):
    if not _reply_ok(row):
        return False
    spoken, planned = row.get('segments_spoken'), row.get('segments_planned_log')
    return (_count(spoken, 1) and _count(planned, 1) and spoken <= planned
            and isinstance(row.get('early_stop'), bool) and row['early_stop'] == (spoken < planned))


def _carry_segment(s):
    return (isinstance(s, dict) and _count(s.get('segment'), 1) and _count(s.get('start')) and _count(s.get('len'), 1)
            and isinstance(s.get('expected'), str) and isinstance(s.get('ok'), bool))


def d2_problems(d2):
    seeds = d2.get('seeds')
    labels = [f'seed-{n}' for n in range(1, 9)]
    if not isinstance(seeds, list) or len(seeds) != 8:
        return ['D2 lacks its 8 seeded replies']
    if not all(_reply_ok(row) and _count(row.get('segments_spoken'), 1) for row in seeds):
        return ['D2 has invalid seeded replies']
    if sorted(row.get('label') for row in seeds if isinstance(row.get('label'), str)) != sorted(labels):
        return ['D2 seed labels are not exactly seed-1..seed-8']
    problems = []
    reached = sum(1 for row in seeds if row['segments_spoken'] >= 2)
    fraction = d2.get('fraction_reaching_segment_2')
    if d2.get('reached_segment_2') != reached or not _count(d2.get('reached_segment_2')) \
            or not _real(fraction) or abs(fraction - reached / 8) > 1e-9:
        problems.append('D2 reached count/fraction missing or inconsistent with its seeds')
    carry = _dict(d2.get('carry'))
    segments = carry.get('segments')
    if not isinstance(segments, list) or not segments or not all(_carry_segment(s) for s in segments) or not _count(carry.get('exit')):
        problems.append('D2 has no carry evidence')
        return problems
    passed = carry['exit'] == 0 and all(s['ok'] for s in segments)
    if carry.get('status') != ('PASS' if passed else 'FAIL'):
        problems.append('D2 carry status is inconsistent with its exit and segments')
    per = carry.get('per_seed')
    if not isinstance(per, dict) or any(not isinstance(per.get(label), dict) or not isinstance(per[label].get('carry_ok'), bool)
                                        or not isinstance(per[label].get('carry_segments'), list) for label in labels):
        problems.append('D2 carry evidence misses seeds')
    return problems


def d3_problems(d3):
    groups = d3.get('groups')
    if not isinstance(groups, dict) or set(groups) != set(DIAG_GROUPS):
        present = groups if isinstance(groups, dict) else {}
        return [f'D3 group {name} lacks its 5 replies' for name in DIAG_GROUPS if name not in present] or ['D3 has unexpected groups']
    problems = []
    for name in DIAG_GROUPS:
        group = groups[name]
        rows = group.get('rows') if isinstance(group, dict) else None
        if not isinstance(rows, list) or len(rows) != 5 or group.get('replies') != 5:
            problems.append(f'D3 group {name} lacks its 5 replies')
        elif not all(_reply_ok(r) and isinstance(r.get('label'), str) and isinstance(r.get('capped'), bool) for r in rows):
            problems.append(f'D3 group {name} has invalid replies')
        elif (group.get('finish_length'), group.get('bytes_cap'), group.get('capped')) != (
                sum(r.get('finish') == 'length' for r in rows), sum(r['bytes'] == 4194304 for r in rows), sum(r['capped'] for r in rows)):
            problems.append(f'D3 group {name} counts disagree with its replies')
    return problems


def d4_problems(d4):
    inputs, rows = d4.get('inputs'), d4.get('rows')
    if not isinstance(inputs, list) or not inputs or not isinstance(rows, list):
        return ['D4 has no jobs or results']
    ids = [job.get('id') if isinstance(job, dict) else None for job in inputs]
    if not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        return ['D4 job ids are missing or duplicated']
    row_ids = [row.get('id') if isinstance(row, dict) else None for row in rows]
    if not all(isinstance(i, str) for i in row_ids) or sorted(row_ids) != sorted(ids):
        return ['D4 result rows do not match its jobs one to one']
    bad = [row['id'] for row in rows if not (_real(row.get('wer')) and _real(row.get('rms')) and _real(row.get('duration_s'), strict=True)
                                            and isinstance(row.get('flags'), list))]
    return [f'D4 lacks results for {len(bad)} of {len(ids)} jobs'] if bad else []


def payload_problems(report):
    """Each D1-D4 payload must hold the coherent, producer-shaped evidence its check exists to produce; a PASS row
    (run here or carried by --diag-resume) over missing, empty, failed, inconsistent or malformed data is not evidence."""
    problems = []
    trees = _dict(_dict(report.get('d1')).get('trees'))
    for key, tree in (('A-nvfp4', 'A'), ('B-nvfp4', 'B')):
        rows = trees.get(key)
        if not isinstance(rows, list) or not rows:
            problems.append(f'D1 has zero {tree} observations')
        elif not all(d1_observation(row) for row in rows):
            problems.append(f'D1 {tree} has invalid observations')
    problems += d2_problems(_dict(report.get('d2')))
    problems += d3_problems(_dict(report.get('d3')))
    problems += d4_problems(_dict(report.get('d4')))
    return problems


def diag_verdict(report):
    """A complete verdict requires every D1-D4 check PASS plus valid payloads for each (actual A and B
    observations, D2 seeds and carry, D3 groups, a D4 result per job). Anything else (failed, WAIT,
    infra-failed or missing check; missing, empty or malformed evidence) is INCONCLUSIVE, never a comparative
    phrase (run 1 claimed 'as often as' from zero observations) and never PASS."""
    report = _dict(report)
    rows = {row.get('name'): row for row in _rows(report.get('checks'))}
    reasons = []
    for name in ('D1', 'D2', 'D3', 'D4'):
        row = rows.get(name)
        if row is None:
            reasons.append(f'{name} did not run')
        elif row.get('status') != 'PASS':
            detail = f"{row.get('status')} ({row.get('error', '')})" if row.get('error') else str(row.get('status'))
            reasons.append(f'{name} {detail}')
    reasons += payload_problems(report)
    return ('INCONCLUSIVE', reasons) if reasons else ('complete', [])


def _fixed(value, places):
    return f'{value:.{places}f}' if _real(value, low=float('-inf')) else str(value)


def diag_markdown(report):
    """Rendered from the report (summary.json), so prior runs and selftest fixtures render identically. Every
    field goes through a safe accessor: carried, incomplete or malformed data renders (INCONCLUSIVE), never throws."""
    report = _dict(report)
    status, reasons = diag_verdict(report)
    why = (' — ' + '; '.join(reasons)) if reasons else ''
    lines = ['# Diagnostic round D1-D4', '', f'Status: {status}{why}', '']
    lines += [f"- {row.get('name')}: {row.get('status')} ({row.get('error', '')}); {row.get('start')} to {row.get('end')}"
              for row in _rows(report.get('checks'))]
    d1 = _dict(report.get('d1'))
    inputs = _dict(d1.get('inputs'))
    if inputs:
        fixed = _rows(inputs.get('fixed')) + [{}, {}]
        lines += ['', '## D1 long-form early stops', '',
                  f"- Inputs: g4-plain ({fixed[0].get('chars')} chars), g3-plain ({fixed[1].get('chars')} chars), "
                  + ', '.join(f"{p.get('label')} {p.get('chars')} chars = "
                              + '+'.join(map(str, p['ids'] if isinstance(p.get('ids'), list) else []))
                              for p in _rows(inputs.get('corpus')))
                  + f" from {inputs.get('corpus_file')}"]
    for probe in _rows(_dict(d1.get('full_profile')).get('probes')):
        lines.append(f"- {probe.get('tree')}/full refused --long-form: exit {probe.get('exit')}: {str(probe.get('output'))[:160]}")
    lines += ['', '| tree-profile | input | HTTP | N (header) | N/M (log) | finish (header/log) | duration s | RMS | early stop |', '|---|---|---|---|---|---|---|---|---|']
    trees = {key: _rows(rows) for key, rows in _dict(d1.get('trees')).items()}
    for key in ('A-nvfp4', 'B-nvfp4'):
        for r in trees.get(key, []):
            lines.append(f"| {key} | {r.get('label')} | {r.get('status')} | {r.get('segments_spoken_header')} | "
                         f"{r.get('segments_spoken_log')}/{r.get('segments_planned_log')} | {r.get('finish_header')}/{r.get('finish_log')} | "
                         f"{num(r.get('duration_s'))} | {num(r.get('rms'))} | {r.get('early_stop')} |")
    counts = {str(key): {'replies': len(rows), 'early': sum(1 for r in rows if r.get('early_stop') is True),
                         'unknown': sum(1 for r in rows if not isinstance(r.get('early_stop'), bool))} for key, rows in trees.items()}
    lines += ['', '**Early stops (N < M) per tree x profile:** '
              + ', '.join(f"{key} {v['early']}/{v['replies']}" + (f" ({v['unknown']} without N/M evidence)" if v['unknown'] else '')
                         for key, v in sorted(counts.items()))]
    a = sum(v['early'] for k, v in counts.items() if k.startswith('A'))
    b = sum(v['early'] for k, v in counts.items() if k.startswith('B'))
    if status == 'complete':
        verdict = 'more often than' if b > a else ('less often than' if b < a else 'as often as')
        lines.append(f'**Verdict: B stops early {verdict} A** (B {b} early stops, A {a}).')
    else:
        detail = '; '.join(reasons)
        lines.append(f'**Verdict: INCONCLUSIVE** ({detail}).')
    d2 = _dict(report.get('d2'))
    seeds = _rows(d2.get('seeds'))
    lines += ['', '## D2 seeded G4 speaker carry (B nvfp4)', '',
              f"- Fraction reaching segment 2: {d2.get('reached_segment_2')}/{len(seeds)}"
              + (f" = {_fixed(d2.get('fraction_reaching_segment_2'), 3)}" if seeds else ''),
              '', '| seed | N/M | finish | duration s | RMS | carry |', '|---|---|---|---|---|---|']
    carry = _dict(d2.get('carry'))
    per_seed = _dict(carry.get('per_seed'))
    for r in seeds:
        label = r.get('label')
        per = per_seed.get(label) if isinstance(label, str) else None
        lines.append(f"| {label} | {r.get('segments_spoken')}/{r.get('segments_planned_log')} | {r.get('finish_header')} | "
                     f"{num(r.get('duration_s'))} | {num(r.get('rms'))} | {per.get('carry_ok') if isinstance(per, dict) else ''} |")
    if carry:
        lines += ['', f"- Carry method: {carry.get('method')}; {carry.get('why_not_server_log')}. Status: {carry.get('status')}.",
                  '- Planned segments: ' + ', '.join(f"{s.get('segment')}@{s.get('start')}->{s.get('expected')}" + (' ok' if s.get('ok') is True else ' WRONG')
                                                     for s in _rows(carry.get('segments')))]
    groups = {str(name): _dict(group) for name, group in _dict(_dict(report.get('d3')).get('groups')).items()}
    lines += ['', '## D3 full-model frame cap (4194304 bytes / finish=length)', '', '| group | replies | finish=length | bytes=cap | capped either |', '|---|---|---|---|---|']
    for name, group in groups.items():
        lines.append(f"| {name} | {group.get('replies')} | {group.get('finish_length')} | {group.get('bytes_cap')} | {group.get('capped')} |")
    lines += ['', '| group | label | finish | bytes | duration s | RMS | capped |', '|---|---|---|---|---|---|---|']
    for name, group in groups.items():
        for r in _rows(group.get('rows')):
            lines.append(f"| {name} | {r.get('label')} | {r.get('finish')} | {r.get('bytes')} | {num(r.get('duration_s'))} | {num(r.get('rms'))} | {r.get('capped')} |")
    d4 = _dict(report.get('d4'))
    lines += ['', '## D4 ASR (Whisper through the pilot code)', '']
    if d4.get('model'):
        lines += [f"- Model {d4.get('model')} on {d4.get('device')}; WER > 0.3 and RMS < 500 are flags, not failures.", '']
    lines += ['| id | WER | RMS | duration s | flags |', '|---|---|---|---|---|']
    for r in _rows(d4.get('rows')):
        flags = r.get('flags') if isinstance(r.get('flags'), list) else []
        if not _real(r.get('wer')):
            lines.append(f"| {r.get('id')} | - | {num(r.get('rms'))} | {num(r.get('duration_s'))} | {r.get('note', 'no transcript')} |")
        else:
            lines.append(f"| {r.get('id')} | {r['wer']:.3f} | {_fixed(r.get('rms'), 0)} | {num(r.get('duration_s'))} | {', '.join(map(str, flags))} |")
    lines += ['', '## Builds', '']
    for name, b in _dict(report.get('builds')).items():
        b = _dict(b)
        lines.append(f"- {name}: engine {b.get('engine_head')}, patches {b.get('patch_series')}, tensorfold-native {b.get('tensorfold_native_sha256')}, drama3-generate {b.get('drama3_generate_sha256')}")
    for name, r in _dict(report.get('build_reuse')).items():
        r = _dict(r)
        lines.append(f"- {name}: {r.get('decision')} ({r.get('reason')})")
    lines += ['', f"- Min MemAvailable: {report.get('min_memavailable_bytes')} bytes", f"- Paused: {report.get('paused')}",
              f"- Resumed: {list(_dict(report.get('resumed')).values())}"]
    deviations = report.get('deviations') if isinstance(report.get('deviations'), list) else []
    commands = report.get('commands') if isinstance(report.get('commands'), list) else []
    lines += ['', '## Deviations', *['- ' + str(x) for x in deviations]]
    lines += ['', '## Commands', '```sh', *map(str, commands), '```', '',
              'Full per-reply measurements: summary.json and the per-check JSON files.']
    return '\n'.join(lines)


def diag_summary():
    if PLAN or not OUT.exists():
        return
    path = OUT / 'summary.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    if RESULTS:
        report.update({'checks': RESULTS, 'commands': COMMANDS, 'deviations': DEVIATIONS + DIAG_DEVIATIONS,
                       'builds': BUILDS, 'build_reuse': REUSE, 'd1': D1REC, 'd2': D2REC, 'd3': D3REC, 'd4': D4REC})
    report['status'], report['status_reasons'] = diag_verdict(report)
    values = [json.loads(line)['memavailable_bytes'] for p in OUT.glob('**/memory.jsonl')
              for line in p.read_text().splitlines() if line.strip()]
    report['min_memavailable_bytes'] = min(values) if values else None
    report['nvidia_peak_bytes'] = None
    report['audio'] = [str(p) for p in OUT.glob('**/*.wav')]
    report['paused'] = (OUT / 'paused.txt').read_text().splitlines() if (OUT / 'paused.txt').exists() else []
    report['resumed'] = json.loads((OUT / 'resumed.json').read_text()) if (OUT / 'resumed.json').exists() else {}
    probe = OUT / 'cpu-probe.json'
    report['cpu_probe'] = json.loads(probe.read_text()) if probe.exists() else {'status': 'unavailable (not verified)'}
    save(OUT / 'summary.json', report)
    (OUT / 'summary.md').write_text(diag_markdown(report) + '\n')


def acquire_gpu():
    # Block on the shared lock so the first release goes to us; the pause step re-checks the GPU and memory floor.
    argv = (['python3', str(Path(__file__).resolve())] + (['--diag'] if DIAG else [])
            + (['--out-dir', str(OUT)] if _argv_value('--out-dir') else []) + ['--pause-step'])
    acquire = ['flock', LOCK, '-c', shlex.join(argv)]
    if PLAN:
        command(acquire)
        return
    while True:
        try:
            command(acquire)
            return
        except subprocess.CalledProcessError:
            log('GPU acquisition deferred; retrying')
            time.sleep(15)


def run_jobs(jobs):
    """Run checks in order. An expected memory WAIT stops every later job (each would meet the same floor) and
    returns REFUSED, which run-diag.sh maps to exit 0; any genuine failure returns 1 and takes precedence."""
    for name, prs, function, args in jobs:
        row = {'name': name, 'prs': prs, 'start': now(), 'status': 'PASS'}
        try:
            function(*args)
        except MemoryAdmission as error:
            # Infrastructure WAIT, not a diagnostic failure: nothing launched, the run stays INCONCLUSIVE.
            row.update(status='WAIT', error=f'{type(error).__name__}: {error}')
            log(row['error'])
        except Exception as error:
            row.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            log(row['error'])
        finally:
            row['end'] = now()
            RESULTS.append(row)
            summary()
        if row['status'] == 'WAIT':
            log('memory WAIT: no further checks start this run')
            break
    if any(x['status'] not in ('PASS', 'WAIT') for x in RESULTS):
        return 1
    return REFUSED if any(x['status'] == 'WAIT' for x in RESULTS) else 0


def diag_selftest():
    # CPU-only unit test of the WER function and the D1 text preparation, per the brief's test list.
    checks = []
    g4 = tag_free(G4_TEXT)
    checks.append(('g4 tags removed', '<|speaker:' not in g4 and g4 == f' {SPEECH}  {SPEECH}'))
    g3 = plain(MARKUP)
    checks.append(('g3 markup removed', all(token not in g3 for token in ('<', '>', '[', ']', '¶', 'whisper', 'laughs')) and '天空很蓝' in g3))
    paragraphs = corpus_paragraphs()
    items = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    by_id = {x['id']: x['text'] for x in items}
    order = [x['id'] for x in items]
    for index, p in enumerate(paragraphs):
        checks.append((f'paragraph {index + 1} in band', 600 <= len(p['text']) <= 900))
        checks.append((f'paragraph {index + 1} verbatim', p['text'] == ' '.join(by_id[i] for i in p['ids'])))
        checks.append((f'paragraph {index + 1} file order', p['ids'] == sorted(p['ids'], key=order.index)))
    checks.append(('paragraph pick deterministic', [p['ids'] for p in corpus_paragraphs()] == [p['ids'] for p in paragraphs]))
    checks.append(('asr words strip markup', asr_words('<|speaker:1|> Is [laughs] the ¶ train?') == ['is', 'the', 'train']))
    checks.append(('asr words keep cjk', asr_words('天空很蓝，火车到了。') == ['天空很蓝火车到了']))
    cases = [
        (('The train leaves at seven fifteen.', 'the train leaves at seven fifteen'), 0.0),
        (('The train leaves at seven fifteen.', 'The train leaves at eight fifteen.'), 1 / 6),
        (('one two three four five', 'one two three four'), 0.2),
        (('one two three', 'one two three four'), 1 / 3),
        (('<|speaker:1|> Is the train here?', 'is the train here'), 0.0),
        (('platform 9', 'platform nine'), 0.5),
        (('', 'something'), 1.0),
        (('', ''), 0.0),
    ]
    for (reference, hypothesis), expected in cases:
        got = word_error_rate(reference, hypothesis)
        checks.append((f'wer {reference[:24]!r} vs {hypothesis[:24]!r}', abs(got - expected) < 1e-9))
    recovery_selftest(checks)
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise RuntimeError(f'diag selftest failed: {failed}')
    log('diag selftest PASS')


def recovery_selftest(checks):
    """Behavioral regressions for the run-1 failure modes: corpus preflight before builds, memory admission,
    verdict honesty, sealed output directories and manifest-verified build reuse. CPU-only; no GPU, network or units."""
    global PLAN, OUT
    base = Path(tempfile.mkdtemp(prefix='gpuchk-recovery-'))
    saved = (PLAN, OUT, memavailable, RESULTS[:], COMMANDS[:])
    PLAN = False
    try:
        # Corpus preflight: missing or mismatched file is an infrastructure failure before any build command.
        marker = len(COMMANDS)
        try:
            corpus_preflight(path=base / 'absent.jsonl')
            checks.append(('missing corpus preflight fails', False))
        except RuntimeError as error:
            checks.append(('missing corpus preflight fails', 'corpus missing' in str(error)))
        wrong = base / 'wrong.jsonl'
        wrong.write_text('not the corpus\n')
        try:
            corpus_preflight(path=wrong, pinned=CORPUS_SHA256)
            checks.append(('mismatched corpus hash fails', False))
        except RuntimeError as error:
            checks.append(('mismatched corpus hash fails', 'sha256 mismatch' in str(error)))
        checks.append(('corpus preflight ran no build commands', not any('build-engine' in x or x.startswith('git') or 'git ' in x for x in COMMANDS[marker:])))
        right = base / 'right.jsonl'
        right.write_text('{"id": "fixture", "text": "hi"}\n')
        pinned = hashlib.sha256(right.read_bytes()).hexdigest()
        checks.append(('matching corpus preflight passes', corpus_preflight(path=right, pinned=pinned) == pinned))

        # Memory admission: below the floor nothing launches, no lock is touched, the check records WAIT.
        globals()['memavailable'] = lambda: PILOT_ADMISSION_FLOOR - 1
        marker = len(COMMANDS)
        admitted = False
        try:
            with server('B', 'full', base / 'admission'):
                admitted = True
        except MemoryAdmission as error:
            checks.append(('low MemAvailable refuses the full-profile pilot', 'floor' in str(error)))
        checks.append(('low MemAvailable launched no pilot and took no lock',
                       not admitted and not any('pilot.py' in x or 'flock' in x or 'gpu.lock' in x for x in COMMANDS[marker:])))
        checks.append(('low MemAvailable created no run directory', not (base / 'admission').exists()))

        # Lost lock race: admission is rechecked before the next Popen; a drop below the floor while waiting
        # launches nothing and creates no attempt-2 log or evidence (attempt-1 evidence is kept).
        levels = iter([PILOT_ADMISSION_FLOOR, PILOT_ADMISSION_FLOOR - 1])
        globals()['memavailable'] = lambda: next(levels)
        launched = []

        class LostRace:
            def __init__(self, argv, stdout, **_):
                launched.append(argv)
                if len(launched) > 1:
                    raise RuntimeError('relaunched below the floor (fixture)')
                stdout.write('Another workload owns the GPU slot\n')

            def poll(self):
                return 1

        real = (subprocess.Popen, command, ROOT)
        subprocess.Popen, globals()['command'], globals()['ROOT'] = LostRace, (lambda argv, **_: COMMANDS.append(shlex.join(map(str, argv)))), base / 'root'
        race = base / 'race'
        marker = len(COMMANDS)
        try:
            with server('B', 'full', race):
                pass
            checks.append(('lock-race retry rechecks admission', False))
        except MemoryAdmission as error:
            checks.append(('lock-race retry rechecks admission', 'floor' in str(error)))
        finally:
            subprocess.Popen, globals()['command'], globals()['ROOT'] = real
        checks.append(('lock-race retry below the floor launches nothing more',
                       len(launched) == 1 and sum('pilot.py' in x for x in COMMANDS[marker:]) == 1))
        checks.append(('lock-race retry below the floor leaves no attempt-2 path',
                       (race / 'pilot-1.out').exists() and not (race / 'pilot-2.out').exists() and not (race / 'pilot-2').exists()))
        globals()['memavailable'] = saved[2]
        OUT = base / 'wait-out'
        OUT.mkdir()
        RESULTS.clear()

        def refuse():
            raise MemoryAdmission('below floor (fixture)')

        ran = []
        checks.append(('memory WAIT returns REFUSED and starts no later check',
                       run_jobs([('D3', [], refuse, ()), ('D4', [], lambda: ran.append('D4'), ())]) == REFUSED
                       and RESULTS[-1]['status'] == 'WAIT' and [r['name'] for r in RESULTS] == ['D3'] and ran == []))
        RESULTS.clear()

        def broken():
            raise RuntimeError('harness error (fixture)')

        checks.append(('genuine failure stays exit 1 even beside a WAIT',
                       run_jobs([('D1', [], broken, ()), ('D3', [], refuse, ())]) == 1))
        RESULTS.clear()
        checks.append(('all-PASS run exits 0', run_jobs([('D1', [], lambda: None, ())]) == 0))
        RESULTS.clear()
        OUT = base / 'preflight-out'
        OUT.mkdir()
        globals()['memavailable'] = lambda: PILOT_ADMISSION_FLOOR - 1
        marker = len(COMMANDS)
        refused = not memory_preflight()
        checks.append(('launch preflight refuses below the floor with WAIT and a summary, running nothing',
                       refused and RESULTS[-1]['name'] == 'preflight-memory' and RESULTS[-1]['status'] == 'WAIT'
                       and (OUT / 'summary.json').exists() and COMMANDS[marker:] == []))
        RESULTS.clear()
        globals()['memavailable'] = lambda: PILOT_ADMISSION_FLOOR
        checks.append(('launch preflight admits at the floor', memory_preflight() and RESULTS == []))
        globals()['memavailable'] = saved[2]
        OUT = saved[1]

        # Verdict honesty: a complete, comparative verdict needs PASS rows AND real D1-D4 payloads.
        comparative = ('as often as', 'more often than', 'less often than')

        def check_row(name, status):
            return {'name': name, 'prs': [], 'start': now(), 'end': now(), 'status': status,
                    'error': 'fixture failure' if status != 'PASS' else ''}

        def observation(early, label='fixture'):
            return {'label': label, 'status': 200, 'bytes': 88200, 'duration_s': 1.0, 'rms': 900.0,
                    'early_stop': early, 'segments_spoken': 1 if early else 2, 'segments_planned_log': 2}

        def full_report(a, b):
            seeds = [observation(False, f'seed-{s}') for s in range(1, 9)]
            reply = {'label': 'r', 'status': 200, 'finish': 'stop', 'bytes': 88200, 'duration_s': 1.0, 'rms': 900.0, 'capped': False}
            return {'checks': [check_row(n, 'PASS') for n in ('D1', 'D2', 'D3', 'D4')], 'builds': {}, 'commands': [], 'deviations': [],
                    'd1': {'trees': {'A-nvfp4': a, 'B-nvfp4': b}},
                    'd2': {'seeds': seeds, 'reached_segment_2': 8, 'fraction_reaching_segment_2': 1.0,
                           'carry': {'method': 'm', 'why_not_server_log': 'w', 'status': 'PASS', 'exit': 0,
                                     'segments': [{'segment': 2, 'start': 0, 'len': 40, 'expected': '<|speaker:1|>', 'ok': True}],
                                     'per_seed': {s['label']: {'segments_spoken': 2, 'carry_segments': [2], 'carry_ok': True} for s in seeds}}},
                    'd3': {'groups': {name: {'replies': 5, 'finish_length': 0, 'bytes_cap': 0, 'capped': 0, 'rows': [dict(reply) for _ in range(5)]}
                                      for name in DIAG_GROUPS}},
                    'd4': {'inputs': [{'id': 'j1'}, {'id': 'j2'}],
                           'rows': [{'id': j, 'wer': 0.0, 'rms': 900.0, 'duration_s': 1.0, 'flags': []} for j in ('j1', 'j2')]}}

        def inconclusive(name, report, needle=''):
            try:
                status, reasons = diag_verdict(report)
                markdown = diag_markdown(report)
            except Exception as error:
                checks.append((f'{name} (raised {type(error).__name__})', False))
                return
            checks.append((name, status == 'INCONCLUSIVE' and any(needle in r for r in reasons)
                           and 'Verdict: INCONCLUSIVE' in markdown and not any(p in markdown for p in comparative)))

        def mutated(change):
            report = full_report([observation(True), observation(False)], [observation(False)])
            change(report)
            return report

        # Round-2 review observations: each malformed field must be INCONCLUSIVE, never comparative, never a crash.
        d2, d4 = (lambda r: r['d2']), (lambda r: r['d4'])
        round2 = {
            'D2 fraction missing': lambda r: d2(r).pop('fraction_reaching_segment_2'),
            'D2 reached count missing': lambda r: d2(r).pop('reached_segment_2'),
            'D2 reached count inconsistent': lambda r: d2(r).update(reached_segment_2=3),
            'D4 duration missing': lambda r: d4(r)['rows'][0].pop('duration_s'),
            'D1 early stop with spoken > planned': lambda r: r['d1']['trees']['A-nvfp4'][0].update(segments_spoken=2, segments_planned_log=1),
            'D1 early stop disagrees with counts': lambda r: r['d1']['trees']['B-nvfp4'][0].update(early_stop=True),
            'D2 seeds all labelled seed-1': lambda r: [s.update(label='seed-1') for s in d2(r)['seeds']],
            'D2 carry PASS with exit 1': lambda r: d2(r)['carry'].update(exit=1),
            'D2 carry per_seed values null': lambda r: d2(r)['carry'].update(per_seed={k: None for k in d2(r)['carry']['per_seed']}),
            'D2 carry per_seed not a dict': lambda r: d2(r)['carry'].update(per_seed=['seed-1']),
            'D3 extra group': lambda r: r['d3']['groups'].update(extra={'replies': 5, 'rows': []}),
            'D3 reply rms NaN': lambda r: r['d3']['groups'][DIAG_GROUPS[1]]['rows'][0].update(rms=float('nan')),
            'D3 count disagrees with replies': lambda r: r['d3']['groups'][DIAG_GROUPS[0]].update(capped=2),
            'D4 wer bool': lambda r: d4(r)['rows'][0].update(wer=True),
            'D4 wer NaN': lambda r: d4(r)['rows'][0].update(wer=float('nan')),
            'D4 rms Inf': lambda r: d4(r)['rows'][0].update(rms=float('inf')),
            'D4 duration zero': lambda r: d4(r)['rows'][0].update(duration_s=0),
            'D4 duplicate job id': lambda r: d4(r)['inputs'].append({'id': 'j1'}),
            'D4 duplicate row': lambda r: d4(r)['rows'].append(dict(d4(r)['rows'][0])),
            'D4 row id unhashable': lambda r: d4(r)['rows'][1].update(id=['j2']),
            'D1 rows not a list': lambda r: r['d1']['trees'].update({'A-nvfp4': 'x'}),
            'whole payloads of wrong type': lambda r: r.update(d1=[], d2='x', d3=None, d4=7, builds=[], checks={}),
        }
        for label, change in round2.items():
            inconclusive(f'malformed: {label}', mutated(change))
        try:
            garbage = diag_markdown('not a report')
            checks.append(('non-dict report renders INCONCLUSIVE', 'Verdict: INCONCLUSIVE' in garbage))
        except Exception:
            checks.append(('non-dict report renders INCONCLUSIVE', False))
        # Context fields (inputs, builds, reuse) are not verdict evidence; malformed ones render safely.
        context = mutated(lambda r: (r['d1'].update(inputs={'fixed': None, 'corpus': [None, {'ids': 5}]}),
                                     r.update(builds={'X': None}, build_reuse={'Y': 3}, deviations=None, commands=7)))
        try:
            checks.append(('malformed context fields render without throwing', 'Verdict: B stops early' in diag_markdown(context)))
        except Exception:
            checks.append(('malformed context fields render without throwing', False))

        inconclusive('zero-observation failed run is INCONCLUSIVE without comparison',
                     {'checks': [check_row(n, 'FAIL') for n in ('D1', 'D2', 'D3', 'D4')], 'builds': {}, 'd1': {'trees': {}},
                      'commands': [], 'deviations': []}, 'zero A observations')
        inconclusive('zero B observations stays INCONCLUSIVE', full_report([observation(True)], []), 'zero B observations')
        partial = full_report([observation(False)], [observation(False)])
        partial['checks'][2] = check_row('D3', 'FAIL')
        inconclusive('one failed check stays INCONCLUSIVE', partial, 'D3 FAIL')
        waited = full_report([observation(False)], [observation(False)])
        waited['checks'][1] = check_row('D2', 'WAIT')
        inconclusive('a WAIT check stays INCONCLUSIVE', waited, 'D2 WAIT')
        # The blind-review repro: PASS rows (as carried by --diag-resume) over invalid D1 rows and empty D2-D4.
        hollow = {'checks': [check_row(n, 'PASS') for n in ('D1', 'D2', 'D3', 'D4')], 'builds': {}, 'commands': [], 'deviations': [],
                  'd1': {'trees': {'A-nvfp4': [{'label': 'x', 'status': 500, 'early_stop': None}],
                                   'B-nvfp4': [{'label': 'x', 'status': 500, 'early_stop': None}]}},
                  'd2': {}, 'd3': {}, 'd4': {}}
        status, reasons = diag_verdict(hollow)
        checks.append(('carried PASS rows over invalid D1 and empty D2-D4 are INCONCLUSIVE',
                       status == 'INCONCLUSIVE' and all(any(r.startswith(d) for r in reasons) for d in ('D1', 'D2', 'D3', 'D4'))
                       and not any(p in diag_markdown(hollow) for p in comparative)))
        bad_d1 = full_report([observation(True)], [dict(observation(False), status=503)])
        inconclusive('failed D1 reply is not an observation', bad_d1, 'D1 B has invalid')
        bad_d1b = full_report([dict(observation(True), bytes=0)], [observation(False)])
        inconclusive('empty D1 reply is not an observation', bad_d1b, 'D1 A has invalid')
        bad_d2 = full_report([observation(True)], [observation(False)])
        bad_d2['d2']['seeds'] = bad_d2['d2']['seeds'][:7]
        inconclusive('missing D2 seed is INCONCLUSIVE', bad_d2, 'D2 lacks')
        no_carry = full_report([observation(True)], [observation(False)])
        no_carry['d2']['carry'] = {'status': 'unavailable (not verified)', 'segments': []}
        inconclusive('missing D2 carry evidence is INCONCLUSIVE', no_carry, 'D2 has no carry')
        bad_d3 = full_report([observation(True)], [observation(False)])
        del bad_d3['d3']['groups'][DIAG_GROUPS[2]]
        inconclusive('missing D3 group is INCONCLUSIVE', bad_d3, 'D3 group A-full')
        bad_d3b = full_report([observation(True)], [observation(False)])
        bad_d3b['d3']['groups'][DIAG_GROUPS[0]]['rows'] = [dict(r, status=500) for r in bad_d3b['d3']['groups'][DIAG_GROUPS[0]]['rows']]
        inconclusive('failed D3 replies are INCONCLUSIVE', bad_d3b, 'has invalid replies')
        bad_d4 = full_report([observation(True)], [observation(False)])
        bad_d4['d4']['rows'][1] = {'id': 'j2', 'note': 'wav absent'}
        inconclusive('D4 job without a result is INCONCLUSIVE', bad_d4, 'D4 lacks results for 1 of 2')
        empty_d4 = full_report([observation(True)], [observation(False)])
        empty_d4['d4'] = {'inputs': [], 'rows': []}
        inconclusive('D4 with no jobs is INCONCLUSIVE', empty_d4, 'D4 has no jobs')
        complete = full_report([observation(True), observation(False)], [observation(False)])
        status, reasons = diag_verdict(complete)
        checks.append(('completed A/B run renders the comparative verdict',
                       status == 'complete' and '**Verdict: B stops early less often than A**' in diag_markdown(complete)))

        # Every existing output path is refused unchanged; only a fresh path is claimed.
        sealed = base / 'sealed'
        sealed.mkdir()
        (sealed / 'summary.json').write_text('{}\n')
        bare = base / 'bare'
        bare.mkdir()
        (bare / 'pilot-1.out').write_text('evidence\n')
        plainfile = base / 'plainfile'
        plainfile.write_text('bytes\n')
        for name, target in (('sealed', sealed), ('summary-free', bare), ('empty', base / 'admission-empty'), ('file', plainfile)):
            if name == 'empty':
                target.mkdir()
            before = sorted((str(p), p.read_bytes() if p.is_file() else None) for p in [target, *target.rglob('*')])
            OUT = target
            try:
                claim_out()
                refused = False
            except RuntimeError as error:
                refused = 'fresh --out-dir' in str(error)
            after = sorted((str(p), p.read_bytes() if p.is_file() else None) for p in [target, *target.rglob('*')])
            checks.append((f'existing {name} out path is refused unchanged', refused and before == after))
        fresh = base / 'fresh'
        OUT = fresh
        claim_out()
        checks.append(('fresh out dir is claimed', fresh.is_dir()))

        # Device settings: a run refuses without every required setting or on another host; pause allowlist is
        # empty unless configured.
        saved_env = {name: os.environ.get(name) for name in REQUIRED_SETTINGS}
        try:
            for name in REQUIRED_SETTINGS:
                os.environ.pop(name, None)
            try:
                device_preflight()
                checks.append(('device run refuses missing settings', False))
            except RuntimeError as error:
                checks.append(('device run refuses missing settings', all(name in str(error) for name in REQUIRED_SETTINGS)))
            for name in REQUIRED_SETTINGS:
                os.environ[name] = 'fixture'
            globals()['DEVICE_HOST'] = os.uname().nodename + '-not-this-host'
            try:
                device_preflight()
                checks.append(('device run refuses another host', False))
            except RuntimeError as error:
                checks.append(('device run refuses another host', 'restricted to GPU_CHECK_HOST' in str(error)))
            globals()['DEVICE_HOST'] = os.uname().nodename
            device_preflight()
            checks.append(('device run admits the configured host', True))
        finally:
            globals()['DEVICE_HOST'] = _setting('GPU_CHECK_HOST')
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        saved_allow = PAUSE_ALLOW[:]
        PAUSE_ALLOW[:] = []
        checks.append(('empty pause allowlist authorizes only the selftest dummy',
                       allowed('gpuchk-selftest-sleep.service') and not allowed('any-queue.service')))
        PAUSE_ALLOW[:] = ['queue-*.service', 'exact.service']
        checks.append(('pause allowlist matches names and patterns only',
                       allowed('queue-7.service') and allowed('exact.service') and not allowed('exact2.service')))
        PAUSE_ALLOW[:] = saved_allow

        # Build reuse: a manifest in force must verify engine head and both binary hashes.
        build = base / 'build-X-full'
        (build / 'zig-out/native/bin').mkdir(parents=True)
        (build / 'zig-out/bin').mkdir(parents=True)
        native = build / 'zig-out/native/bin/tensorfold-native'
        native.write_bytes(b'native-fixture')
        (build / 'gpu-check-head').write_text('abc123\n')
        manifest = {'engine_head': 'abc123', 'tensorfold_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
                    'drama3_generate_sha256': None}
        checks.append(('build reuse verifies a matching manifest',
                       build_reusable(build, 'abc123', manifest, True) == (True, 'verified against the manifest')))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, tensorfold_native_sha256='0' * 64), True)
        checks.append(('build reuse rejects a hash mismatch', not usable and 'tensorfold-native sha256 differs' in reason))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, engine_head='def456'), True)
        checks.append(('build reuse rejects an engine head mismatch', not usable and 'engine_head differs' in reason))
        usable, reason = build_reusable(build, 'abc123', None, True)
        checks.append(('build reuse rejects a missing manifest entry', not usable and 'no manifest entry' in reason))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, drama3_generate_sha256=hashlib.sha256(b'other').hexdigest()), True)
        checks.append(('build reuse rejects a drama3-generate mismatch', not usable and 'drama3-generate sha256 differs' in reason))
        usable, reason = build_reusable(build, 'other', manifest, False)
        checks.append(('build reuse needs the stamp without a manifest', not usable))
        usable, reason = build_reusable(build, 'abc123', None, False)
        checks.append(('build reuse rebuilds a matching stamp without a manifest',
                       not usable and 'no manifest in force' in reason))
    finally:
        PLAN, OUT = saved[0], saved[1]
        globals()['memavailable'] = saved[2]
        RESULTS[:] = saved[3]
        COMMANDS[:] = saved[4]
        shutil.rmtree(base)


def claim_out():
    # Every existing output path is refused, summary or not (run-1 out-diag, a half-written or foreign directory,
    # a plain file): mkdir without exist_ok is the atomic claim and changes no existing bytes. Each retry or
    # repair round takes a fresh --out-dir.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        OUT.mkdir()
    except FileExistsError:
        raise RuntimeError(f'refusing existing output path {OUT}; pass a fresh --out-dir') from None


def carry_resume(jobs):
    """--diag-resume SUMMARY: prior PASS checks are carried into this run's summary instead of re-running,
    so only the missing D1-D4 checks execute; the prior out dir's wavs stay readable for D4."""
    prior = json.loads(Path(DIAG_RESUME).read_text())
    done = {row['name']: row for row in prior.get('checks', []) if row.get('status') == 'PASS'}
    kept = [job for job in jobs if job[0] not in done]
    RESULTS.extend(done[name] for name, _, _, _ in jobs if name in done)
    if 'D1' in done:
        D1REC.update(prior.get('d1') or {})
    elif 'D2' in done:
        # D2's server also recorded D1's B column; keep that evidence when D1 alone re-runs.
        D1REC['trees']['B-nvfp4'] = (prior.get('d1') or {}).get('trees', {}).get('B-nvfp4', [])
    if 'D2' in done:
        D2REC.update(prior.get('d2') or {})
    if 'D3' in done:
        D3REC.update(prior.get('d3') or {})
    if 'D4' in done:
        D4REC.update(prior.get('d4') or {})
    if not kept:
        summary()
    return kept


def diag_main():
    if not PLAN:
        device_preflight()
        claim_out()
        ROOT.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        # First gate, before any build, unit pause, GPU lock or pilot (run 1 lost D1/D2/D4 here).
        corpus_preflight()
    except Exception as error:
        RESULTS.append({'name': 'preflight-corpus', 'prs': [], 'start': now(), 'end': now(),
                        'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'})
        log(RESULTS[-1]['error'])
        if not PLAN:
            summary()
        return 1
    if not PLAN and not memory_preflight():
        return REFUSED
    prep()
    record_builds()
    acquire_gpu()
    # D1 and D2 first (the deciding checks), then D3, then D4.
    jobs = [('D1', [], d1, ()), ('D2', [], d2, ()), ('D3', [], d3, ()), ('D4', [], d4, ())]
    if DIAG_RESUME and not PLAN:
        jobs = carry_resume(jobs)
    return run_jobs(jobs)


# Clean refusal (memory floor not met, nothing launched): run-diag.sh exits 0 so systemd records no failure.
REFUSED = 75


def memory_preflight():
    """Second gate, before any build, pause, lock or pilot: below the full-profile floor record WAIT, write the
    INCONCLUSIVE summary and return False so diag_main exits REFUSED."""
    available = memavailable()
    if available >= PILOT_ADMISSION_FLOOR:
        return True
    RESULTS.append({'name': 'preflight-memory', 'prs': [], 'start': now(), 'end': now(), 'status': 'WAIT',
                    'error': f'MemAvailable {available / 2**30:.2f} GiB below floor '
                             f'{PILOT_ADMISSION_FLOOR / 2**30:.2f} GiB; refused before launch'})
    log(RESULTS[-1]['error'])
    summary()
    return False


def summary():
    if DIAG:
        diag_summary()
        return
    if PLAN or not OUT.exists():
        return
    path = OUT / 'summary.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    if RESULTS:
        report.update({'checks': RESULTS, 'commands': COMMANDS, 'deviations': DEVIATIONS, 'build_reuse': REUSE})
    memory = []
    for path in OUT.glob('G*/**/memory.jsonl'):
        values = [json.loads(line)['memavailable_bytes'] for line in path.read_text().splitlines() if line.strip()]
        if values:
            memory.append({'path': str(path), 'min_memavailable_bytes': min(values), 'max_available_drop_bytes': max(values) - min(values)})
    report['host_memory'] = memory
    report['nvidia_peak_bytes'] = None
    report['audio'] = [str(p) for p in OUT.glob('G*/**/*.wav')]
    report['paused'] = (OUT / 'paused.txt').read_text().splitlines() if (OUT / 'paused.txt').exists() else []
    report['resumed'] = json.loads((OUT / 'resumed.json').read_text()) if (OUT / 'resumed.json').exists() else {}
    report['listening'] = 'pending owner'
    probe = OUT / 'cpu-probe.json'
    report['cpu_probe'] = json.loads(probe.read_text()) if probe.exists() else {'status': 'unavailable (not verified)'}
    report['per_pr'] = {str(pr): ('FAIL' if any(row['status'] == 'FAIL' for row in report.get('checks', []) if pr in row['prs']) else ('PASS' if len(report.get('checks', [])) == 9 else 'INCOMPLETE')) for pr in (26, 27, 28)}
    save(OUT / 'summary.json', report)
    lines = ['# GPU checks', '', 'Listening: pending owner', '', 'CPU text-boundary and speaker probe: ' + report['cpu_probe']['status'], '', '## Automatic checks']
    for row in report.get('checks', []):
        lines.append(f"- {row['name']}: {row['status']} ({row.get('error', '')}); PRs {row['prs']}; {row['start']} to {row['end']}")
    lines += ['', '## Deviations', *['- ' + x for x in DEVIATIONS], '', '## Listening files', *report['audio'], '', '## Commands', '```sh', *report.get('commands', []), '```', '', 'Full measurements and pause/resume timestamps: summary.json and raw check JSON files.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    if '--resume' in sys.argv:
        resume()
        return 0
    if '--pause-selftest' in sys.argv:
        pause(selftest=True)
        return 0
    if '--patch-selftest' in sys.argv:
        patch_selftest()
        return 0
    if '--segments-selftest' in sys.argv:
        segments_selftest()
        return 0
    if '--diag-selftest' in sys.argv:
        diag_selftest()
        return 0
    if '--pause-step' in sys.argv:
        pause()
        return 0
    if DIAG:
        return diag_main()
    if not PLAN:
        device_preflight()
        claim_out()
        ROOT.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    prep()
    acquire_gpu()
    jobs = [(f'G1-{p}', [26, 27, 28], g1, (p,)) for p in ('full', 'nvfp4')]
    jobs += [(f'G2-{p}', [27], g2, (p,)) for p in ('full', 'nvfp4')]
    jobs += [(f'G3-{p}', [26], g3, (p,)) for p in (0, .5)]
    jobs += [('G4', [26, 27, 28], g4, ())]
    jobs += [(f'G5-{p}', [28], g5, (p,)) for p in ('full', 'nvfp4')]
    return run_jobs(jobs)


if __name__ == '__main__':
    raise SystemExit(main())
