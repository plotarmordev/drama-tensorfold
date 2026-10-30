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
# Only the full model is served; the NVFP4 builds are compared by source tree, never loaded.
CHECKPOINTS = {'full': PILOT_ROOT / 'checkpoint'}
ZIG = Path(_setting('GPU_CHECK_ZIG'))
# The pilot's Whisper code directory (transcribe.py), read-only; diagnostic D2 only.
ASR_DIR = Path(_setting('GPU_CHECK_ASR_DIR'))
NVCC = os.environ.get('GPU_CHECK_NVCC') or '/usr/local/cuda/bin/nvcc'
LIB = PILOT_ROOT / 'venv/lib/python3.12/site-packages/nvidia/cu13/lib'
# Units the runner may SIGSTOP for a GPU slot: space-separated exact unit names or fnmatch patterns. Unset means
# none (only the selftest dummy); the runner never pauses anything the owner did not list.
PAUSE_ALLOW = (os.environ.get('GPU_CHECK_PAUSE_ALLOW') or '').split()
# A GPU process listening on this port (an owner-designated service) is tolerated at admission; unset tolerates none.
TOLERATED_PORT = os.environ.get('GPU_CHECK_TOLERATED_PORT')
REQUIRED_SETTINGS = ('GPU_CHECK_HOST', 'GPU_CHECK_LOCK', 'GPU_CHECK_PILOT_ROOT', 'GPU_CHECK_ZIG') + (('GPU_CHECK_ASR_DIR',) if DIAG else ())


def device_preflight():
    # Before claim_out or any other effect: every host-local setting present, and this is the configured host.
    missing = [name for name in REQUIRED_SETTINGS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f'device run needs {", ".join(missing)} (README "Device configuration")')
    if os.uname().nodename != DEVICE_HOST:
        raise RuntimeError(f'device runner is restricted to GPU_CHECK_HOST ({DEVICE_HOST})')


# PR #33 A/B. A: main f68738ea, the old full-model pin d1018b31 plus patches 0001-0015 and the CUDA NVFP4 series.
# B: 3f44de62, PR #33's TensorFold v1.1.9 tree; its engine/, scripts/, recipe/ and bench/ equal the PR head's, and
# only those come from root/B (this runner and its plan run from $W/gpu-checks). Only the full model is compared.
SHAS = {'A': 'f68738ea91de74dcf8bb698a26df44b5cf078396', 'B': '3f44de621379bbe20f1ae6599a26465edf9dcbce'}
# Full-model patch series per tree; None means unpatched by contract. B's full model is the empty v1.1.9 series;
# scripts/build-engine.sh in that tree still checks its manifest at build time.
PATCH_DIRS = {'A': 'engine/patches', 'B': None}
# build-engine.sh fixes the committer and takes dates from the patches, so each pinned build has one HEAD:
# A = d1018b31 + patches 0001-0015 (reproduced with git am outside the device), B = v1.1.9 465e48b6 + 0 patches.
# A build dir at any other HEAD (e.g. from an earlier pin) is never reused and a fresh build must reach it.
EXPECTED_HEADS = {'A': '3e1721df2b913a6748fc686b99385229fc8adb02', 'B': '465e48b624202d027cb72144749753578846aa7c'}
# NVFP4 identity without building: the same engine/nvfp4 tree (UPSTREAM and patch series) in A and B, and the two
# reviewed build-engine.sh blobs, which differ only in the full-model empty-series exemption.
NVFP4_TREE = '7c9e2d863e76b2495f103f7755cd159b9e09baaf'
BUILD_SCRIPT_BLOBS = {'A': '838d6672ba7d152081969017df6782222bd98d35', 'B': 'd2dc452b6b6f60ff4d9ce374249cc804fd9e5b7d'}
NVFP4_NOTE = ('NVFP4: A and B pin the same engine/nvfp4 tree and their build-engine.sh differs only in the full-model '
              'empty-series exemption, so their NVFP4 builds would be identical; no NVFP4 engine is built and no '
              'NVFP4 A/B check runs.')
BASE = 'http://127.0.0.1:18795'
TEXT = 'The train leaves at seven fifteen. Please bring the blue notebook.'
REFS = ['The small boat crossed the lake.', 'A bright bird landed near the window.']
DIALOGUE = '<|speaker:1|> Is the train here? <|speaker:15|> It arrives at seven fifteen.'
# pilot.py's own full-profile admission floor, mirrored instead of discovered by a failed launch; never lowered.
PILOT_ADMISSION_FLOOR = 88 * 2**30
PILOT = {'log': None}
BUILDS = {}
REUSE = {}
NVFP4_IDENTITY = {}
GATE = {}
D1REC = {}
D2REC = {}
DIAG_DEVIATIONS = [
    'Both trees take the same seeded, catalog-voice dialogue requests (tagged and untagged, seeds 1-5): main already carries request seeds and speaker tags (patches 0013-0015), so D1 compares like for like.',
    'D2 transcribes D1\'s replies through the pilot\'s Whisper code (GPU_CHECK_ASR_DIR transcribe.py, read-only, its own venv) and computes WER locally.',
    'The runner exits non-zero only on infrastructure errors; every diagnostic observation (frame cap, WER flag) is recorded, never fatal.',
    'Full-profile pilots launch only at or above the 88 GiB MemAvailable floor; below it the check records WAIT/INCONCLUSIVE without launching, without holding the GPU lock and without pausing or killing any scoring process.',
    'A complete verdict requires D1 and D2 PASS plus valid A and B evidence; any failed, WAITed or missing check renders the whole run INCONCLUSIVE, never a frame-cap result or PASS.',
    '--diag-resume re-executes only the missing D1-D2 checks into a fresh --out-dir.',
]
COMMANDS = []
RESULTS = []
DEVIATIONS = [
    'ROOT is under the assigned work directory; host-local paths come from the GPU_CHECK_* environment settings.',
    'Retained builds are reused only at their pinned HEAD (EXPECTED_HEADS) and when a recorded manifest (--build-manifest, or root/build-manifest.json written by each run) verifies that head and both binary sha256s; any other build is rebuilt and must reach the pinned HEAD.',
    NVFP4_NOTE,
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


def build_reusable(build, head, recorded, in_force, expected):
    """head is the build checkout's current HEAD and expected the pinned EXPECTED_HEADS value. Reuse requires that
    head to be the pinned one (a build from an earlier pin never passes as A or B) and a manifest in force
    (--build-manifest or the runner-written root/build-manifest.json) whose recorded engine head and both binary
    sha256s verify against the stamped checkout; the stamp alone proves source identity, not binary integrity, so
    without a manifest exactly that build is rebuilt."""
    stamp = build / 'gpu-check-head'
    binary = build / 'zig-out/native/bin/tensorfold-native'
    generate = build / 'zig-out/bin/drama3-generate'
    stamp_value = stamp.read_text().strip() if stamp.exists() else None
    if head != expected:
        return False, f'build HEAD {head} is not the pinned {expected}'
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
    # root/A is a worktree of root/B; restore it without re-cloning when only root/B remains.
    if PLAN or not (ROOT / 'A').exists():
        command(['git', 'worktree', 'add', '--detach', ROOT / 'A', SHAS['A']], cwd=ROOT / 'B')
    for tree in SHAS:
        head = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT / tree, capture=True)
        if not PLAN and head != SHAS[tree]:
            raise RuntimeError(f'root/{tree} HEAD {head} is not the pinned {SHAS[tree]}; remove root/A and root/B '
                               '(README "Existing root/") and run again')
    # Before any build or GPU work: a NVFP4 difference would need its own A/B, which this run does not have.
    nvfp4_identity()
    for tree in SHAS:
        source = ROOT / tree
        verify_patches(tree, source)
        build = builddir(tree, 'full')
        key = f'{tree}-full'
        stamp = build / 'gpu-check-head'
        binary = build / 'zig-out/native/bin/tensorfold-native'
        if not PLAN and binary.exists() and stamp.exists():
            reusable, reason = build_reusable(build, command(['git', 'rev-parse', 'HEAD'], cwd=build, capture=True),
                                              manifest.get(key), bool(manifest), EXPECTED_HEADS[tree])
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
        command(['env', f'ZIG={ZIG}', f'NVCC={NVCC}', 'JOBS=4', 'bash', source / 'scripts/build-engine.sh', '--weights', 'full', build])
        if not PLAN:
            head = command(['git', 'rev-parse', 'HEAD'], cwd=build, capture=True)
            if head != EXPECTED_HEADS[tree]:
                raise RuntimeError(f'build {key} reached HEAD {head}, not the pinned {EXPECTED_HEADS[tree]}')
            stamp.write_text(head + '\n')


def nvfp4_identity(repo=None, shas=SHAS, tree=NVFP4_TREE, blobs=BUILD_SCRIPT_BLOBS):
    # Source-level proof, no build: the same engine/nvfp4 tree in A and B, and the two reviewed build-engine.sh blobs.
    repo = repo or ROOT / 'B'
    trees = command(['git', 'rev-parse', f"{shas['A']}:engine/nvfp4", f"{shas['B']}:engine/nvfp4"], cwd=repo, capture=True).split()
    scripts = command(['git', 'rev-parse', f"{shas['A']}:scripts/build-engine.sh", f"{shas['B']}:scripts/build-engine.sh"],
                      cwd=repo, capture=True).split()
    if PLAN:
        print(f'# {NVFP4_NOTE}')
        return
    NVFP4_IDENTITY.update(nvfp4_trees=dict(zip('AB', trees)), build_scripts=dict(zip('AB', scripts)),
                          identical=trees == [tree, tree] and scripts == [blobs['A'], blobs['B']])
    if not NVFP4_IDENTITY['identical']:
        raise RuntimeError(f'A and B NVFP4 sources differ from the reviewed pins: engine/nvfp4 {trees} (want {tree}), '
                           f'build-engine.sh {scripts} (want {[blobs["A"], blobs["B"]]})')
    log(f'NVFP4 sources identical (engine/nvfp4 {tree}); no NVFP4 engine is built or compared')


def verify_patches(tree, source):
    patch = PATCH_DIRS[tree]
    if patch is None:
        return
    path = source / patch
    if not PLAN and not path.is_dir():
        raise RuntimeError(f'{tree}/full patch series missing: {path}')
    manifest = path / 'SHA256SUMS'
    # Same rule as scripts/build-engine.sh: an empty full-model manifest declares an empty series, which
    # `sha256sum -c` would reject.
    if manifest.is_file() and manifest.stat().st_size == 0 and not any(path.glob('*.patch')):
        log(f'{tree}/full patch series is empty by manifest; no checksums to verify')
        return
    command(['sha256sum', '-c', 'SHA256SUMS'], cwd=path)


def patch_selftest():
    base = Path(tempfile.mkdtemp(prefix='gpuchk-patches-'))
    path = base / 'A' / PATCH_DIRS['A']

    def fixture():
        path.mkdir(parents=True, exist_ok=True)
        (path / '0001-fixture.patch').write_text('fixture series for A\n')
        (path / 'SHA256SUMS').write_text(hashlib.sha256((path / '0001-fixture.patch').read_bytes()).hexdigest() + '  0001-fixture.patch\n')

    fixture()
    mark = len(COMMANDS)
    verify_patches('B', base / 'B')
    if any('sha256sum' in x for x in COMMANDS[mark:]):
        raise RuntimeError('B/full is unpatched by contract but ran a checksum command')
    verify_patches('A', base / 'A')
    if not any('sha256sum' in x and f'/A/{PATCH_DIRS["A"]}' in x for x in COMMANDS[mark:]):
        raise RuntimeError('A/full patch series was not checksum-verified')
    shutil.rmtree(path)
    try:
        verify_patches('A', base / 'A')
    except RuntimeError:
        fixture()
    else:
        raise RuntimeError('A/full accepted a missing patch series')
    (path / '0001-fixture.patch').write_text('corrupted\n')
    try:
        verify_patches('A', base / 'A')
    except subprocess.CalledProcessError:
        pass
    else:
        raise RuntimeError('A/full accepted a checksum mismatch')
    # An empty full manifest with no patches is an empty series and skips the check.
    (path / '0001-fixture.patch').unlink()
    (path / 'SHA256SUMS').write_text('')
    mark = len(COMMANDS)
    verify_patches('A', base / 'A')
    if any('sha256sum' in x for x in COMMANDS[mark:]):
        raise RuntimeError('A/full empty series ran a checksum command')
    nvfp4_selftest(base / 'repo')
    shutil.rmtree(base)
    log('patch selftest PASS')


def nvfp4_selftest(repo):
    # Two fixture commits with the same engine/nvfp4 tree pass; a changed NVFP4 patch or build script aborts.
    repo.mkdir()
    env = dict(os.environ, GIT_AUTHOR_NAME='f', GIT_AUTHOR_EMAIL='f@example.invalid', GIT_COMMITTER_NAME='f',
               GIT_COMMITTER_EMAIL='f@example.invalid')
    git = lambda *args: subprocess.run(['git', '-c', 'commit.gpgsign=false', *args], cwd=repo, env=env, check=True,
                                       capture_output=True, text=True).stdout.strip()
    git('init', '-q')

    def commit(patch, script):
        (repo / 'engine/nvfp4').mkdir(parents=True, exist_ok=True)
        (repo / 'scripts').mkdir(exist_ok=True)
        (repo / 'engine/nvfp4/0001.patch').write_text(patch)
        (repo / 'scripts/build-engine.sh').write_text(script)
        git('add', '-A')
        git('commit', '-qm', 'fixture', '--allow-empty')
        return git('rev-parse', 'HEAD')

    shas = {'A': commit('nvfp4\n', 'a\n'), 'B': commit('nvfp4\n', 'b\n')}
    tree = git('rev-parse', f"{shas['A']}:engine/nvfp4")
    blobs = {t: git('rev-parse', f'{shas[t]}:scripts/build-engine.sh') for t in shas}
    NVFP4_IDENTITY.clear()
    nvfp4_identity(repo, shas, tree, blobs)
    if NVFP4_IDENTITY.get('identical') is not True:
        raise RuntimeError('identical NVFP4 sources were not accepted')
    for label, changed in (('NVFP4 patch', {'B': commit('changed\n', 'b\n')}), ('build script', {'B': commit('nvfp4\n', 'c\n')})):
        try:
            nvfp4_identity(repo, dict(shas, **changed), tree, blobs)
        except RuntimeError:
            pass
        else:
            raise RuntimeError(f'a changed {label} passed the NVFP4 identity check')
    NVFP4_IDENTITY.clear()


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


def bench(name, *args):
    command(['python3', ROOT / 'B' / 'bench' / name, *args])


def catalog():
    return OUT / 'synthetic' / 'full' / 'catalog'


def voices():
    # Full-model catalog voices A, B and the tagged 'dialogue' voice; both trees load the same catalog.
    run = OUT / 'synthetic' / 'full'
    if not PLAN and (run / 'complete.json').exists():
        return
    with server('B', 'full', run / 'render'):
        for i, text in enumerate(REFS):
            request(run, f'ref-{i}', text, seed=42 + i)
    for i in range(len(REFS)):
        command(['flock', LOCK, 'env', f'LD_LIBRARY_PATH={LIB}', builddir('B', 'full') / 'zig-out/bin/drama3-generate', '--codec', CHECKPOINTS['full'] / 'codec.pth', '--encode', run / f'ref-{i}.codes', '--ref-audio', run / f'ref-{i}.wav'])
    for name, refs in [('A', [(0, REFS[0])]), ('B', [(1, REFS[1])]), ('dialogue', [(0, '<|speaker:1|> ' + REFS[0]), (1, '<|speaker:15|> ' + REFS[1])])]:
        command(['python3', ROOT / 'B' / 'recipe/build_voice.py', '--source', 'synthetic', '--output-dir', catalog(), name, 'Model-invented clips rendered by candidate built-in voice; request evidence retained.', *[f'{text}:{run}/ref-{i}.codes' for i, text in refs]])
    if not PLAN:
        save(run / 'complete.json', {'finished_at': now()})


def g1():
    run = OUT / 'G1' / 'full'
    for tree in SHAS:
        with server(tree, 'full', run / tree / 'default'):
            bench('corpus_capture.py', BASE, run / tree / 'capture', '--passes', '2')
            bench('stream_check.py', '--base', BASE, '--model', 'fishaudio/drama-3', '--input', TEXT, '--out', run / tree / 'stream-check.json')
    voices()
    for tree in SHAS:
        with server(tree, 'full', run / tree / 'reference', ['--voices', catalog()]):
            bench('reference_cache_test.py', '--base', BASE, '--voice', 'A', '--other-voice', 'B', '--out', run / tree / 'rc.json')
    compare_gate(run)


def compare_gate(run):
    # Byte identity between main and v1.1.9 is measured, not assumed: compare_gate.py exits 1 for identical=false,
    # which is a result. Only a missing or inconsistent report (or any other exit code) is a harness failure.
    out = run / 'gate-B.json'
    argv = ['python3', ROOT / 'B' / 'bench' / 'compare_gate.py', run / 'A/capture', run / 'A/capture', run / 'B/capture', out]
    text = shlex.join(list(map(str, argv)))
    COMMANDS.append(text)
    if PLAN:
        print(shown(text))
        return
    log(text)
    done = subprocess.run(list(map(str, argv)), capture_output=True, text=True)
    report = json.loads(out.read_text()) if out.exists() else {}
    identical = report.get('identical')
    if done.returncode not in (0, 1) or not isinstance(identical, bool) or (done.returncode == 0) != identical:
        raise RuntimeError(f'compare_gate.py failed: exit {done.returncode}, identical={identical!r}: {done.stderr[-300:]}')
    GATE.update(identical=identical, normal_mismatches=report.get('normal_mismatches'),
                long05_clipped_identical=report.get('long05_clipped_identical'), report=str(out))
    log(f'G1 byte identity A vs B: {identical}')


def g2():
    # Request sampling: patches 0013/0014 on A, upstream #1277 on B; each tree must honour the same fields.
    for tree in SHAS:
        run = OUT / 'G2' / 'full' / tree
        with server(tree, 'full', run):
            for seed in (42, 9007199254740993, 18446744073709551615):
                a, _ = request(run, f'seed-{seed}-first', seed=seed)
                b, _ = request(run, f'seed-{seed}-repeat', seed=seed)
                if not PLAN and a != b:
                    raise RuntimeError(f'{tree}: non-repeatable seed {seed}')
            request(run, 'temperature-zero', temperature=0)
            request(run, 'top-k-one', top_k=1)
            request(run, 'sampling', temperature=.7, top_p=.8, top_k=20)
            request(run, 'invalid-top-k', 'hi', expected=400, top_k=65)


def g3():
    # Speaker tags: patch 0015 on A, upstream #1278 on B; the same tagged and untagged dialogue on each tree.
    voices()
    for tree in SHAS:
        run = OUT / 'G3' / 'full' / tree
        with server(tree, 'full', run, ['--voices', catalog()]):
            for voice in ('default', 'dialogue'):
                request(run, f'{voice}-tagged', DIALOGUE, voice=voice)
                request(run, f'{voice}-untagged', tag_free(DIALOGUE), voice=voice)


def tag_free(text):
    return re.sub(r'<\|speaker:\d+\|>', '', text)


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
    info.update(label=label, input_chars=len(text), server_log=pilot_new(offset).splitlines()[-3:])
    info['finish_header'] = info['headers'].get('X-Finish-Reason')
    save(run / f'{label}.json', info)
    return info


def d1():
    # Issue #32's full-model frame cap, like for like: both trees, the same tagged and untagged dialogue, seeds 1-5.
    run = OUT / 'D1'
    D1REC['start'] = now()
    voices()
    D1REC['groups'] = {}
    for tree in SHAS:
        with server(tree, 'full', run / tree, ['--voices', catalog()]):
            for kind, text in (('tagged', DIALOGUE), ('untagged', tag_free(DIALOGUE))):
                rows = [diag_request(run / tree, f'{kind}-seed-{seed}', text, voice='dialogue', seed=seed) for seed in range(1, 6)]
                D1REC['groups'][f'{tree}-full-dialogue-{kind}-seed-1-5'] = d1_group(rows)
    D1REC['end'] = now()


def d1_group(rows):
    detail = [{'label': r.get('label'), 'status': r.get('status'), 'finish': r.get('finish_header'), 'bytes': r.get('bytes'),
               'duration_s': r.get('duration_s'), 'rms': r.get('rms'), 'sha256': r.get('sha256'),
               'capped': r.get('finish_header') == 'length' or r.get('bytes') == 4194304} for r in rows]
    return {'replies': len(detail), 'finish_length': sum(1 for r in detail if r['finish'] == 'length'),
            'bytes_cap': sum(1 for r in detail if r['bytes'] == 4194304), 'capped': sum(1 for r in detail if r['capped']),
            'rows': detail}


def d2_jobs():
    jobs = []
    seen = set()
    # A --diag-resume run reads wavs from both this run's out dir and the prior one; current evidence wins.
    roots = [OUT] + ([Path(DIAG_RESUME).resolve().parent] if DIAG_RESUME else [])
    for root in roots:
        for wav in sorted(root.glob('D1/*/*.wav')):
            identifier = f'd1-{wav.parent.name}-{wav.stem}'
            if identifier not in seen:
                seen.add(identifier)
                jobs.append({'id': identifier, 'wav': str(wav), 'reference': tag_free(DIALOGUE),
                             'exists': wav.exists() if not PLAN else None})
    return jobs


def wav_measure(path):
    with wave.open(str(path), 'rb') as audio:
        data = audio.readframes(audio.getnframes())
    samples = [v[0] for v in struct.iter_unpack('<h', data)]
    return {'duration_s': len(samples) / 44100, 'rms': (sum(x * x for x in samples) / len(samples)) ** .5 if samples else 0}


def d2():
    run = OUT / 'D2'
    D2REC['start'] = now()
    jobs = d2_jobs()
    D2REC['inputs'] = jobs
    manifest, result = run / 'input.json', run / 'asr.json'
    if not PLAN:
        save(manifest, {'asr_dir': str(ASR_DIR),
                        'jobs': [{'id': j['id'], 'wav': j['wav']} for j in jobs if j['exists']]})
    # Whisper runs on the GPU, so it takes the shared lock like the pilots do.
    command(['flock', LOCK, PILOT_ROOT / 'venv/bin/python', W / 'gpu-checks/asr.py', manifest, result])
    if not PLAN:
        asr = json.loads(result.read_text())
        D2REC.update(model=asr['model'], device=asr['device'])
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
        D2REC['rows'] = rows
        D2REC['flagged'] = [row['id'] for row in rows if row.get('flags')]
    D2REC['end'] = now()


def record_builds():
    if PLAN:
        return
    for tree in SHAS:
        build = builddir(tree, 'full')
        native = build / 'zig-out/native/bin/tensorfold-native'
        generate = build / 'zig-out/bin/drama3-generate'
        stamp = build / 'gpu-check-head'
        BUILDS[f'{tree}-full'] = {
            'engine_head': stamp.read_text().strip() if stamp.exists() else None,
            'expected_head': EXPECTED_HEADS[tree],
            'patch_series': PATCH_DIRS[tree],
            'tensorfold_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest() if native.exists() else None,
            'drama3_generate_sha256': hashlib.sha256(generate.read_bytes()).hexdigest() if generate.exists() else None,
        }
    # Written next to the builds so the next run verifies retained-build reuse against recorded hashes.
    save(ROOT / 'build-manifest.json', {'builds': BUILDS})


def num(value):
    return f'{value:.2f}' if isinstance(value, (int, float)) else str(value)


DIAG_GROUPS = tuple(f'{tree}-full-dialogue-{kind}-seed-1-5' for tree in ('A', 'B') for kind in ('tagged', 'untagged'))


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


def d1_problems(d1):
    groups = d1.get('groups')
    if not isinstance(groups, dict) or set(groups) != set(DIAG_GROUPS):
        present = groups if isinstance(groups, dict) else {}
        return [f'D1 group {name} lacks its 5 replies' for name in DIAG_GROUPS if name not in present] or ['D1 has unexpected groups']
    problems = []
    for name in DIAG_GROUPS:
        group = groups[name]
        rows = group.get('rows') if isinstance(group, dict) else None
        if not isinstance(rows, list) or len(rows) != 5 or group.get('replies') != 5:
            problems.append(f'D1 group {name} lacks its 5 replies')
        elif not all(_reply_ok(r) and isinstance(r.get('label'), str) and isinstance(r.get('capped'), bool) for r in rows):
            problems.append(f'D1 group {name} has invalid replies')
        elif (group.get('finish_length'), group.get('bytes_cap'), group.get('capped')) != (
                sum(r.get('finish') == 'length' for r in rows), sum(r['bytes'] == 4194304 for r in rows), sum(r['capped'] for r in rows)):
            problems.append(f'D1 group {name} counts disagree with its replies')
    return problems


def d2_problems(d2):
    inputs, rows = d2.get('inputs'), d2.get('rows')
    if not isinstance(inputs, list) or not inputs or not isinstance(rows, list):
        return ['D2 has no jobs or results']
    ids = [job.get('id') if isinstance(job, dict) else None for job in inputs]
    if not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        return ['D2 job ids are missing or duplicated']
    row_ids = [row.get('id') if isinstance(row, dict) else None for row in rows]
    if not all(isinstance(i, str) for i in row_ids) or sorted(row_ids) != sorted(ids):
        return ['D2 result rows do not match its jobs one to one']
    bad = [row['id'] for row in rows if not (_real(row.get('wer')) and _real(row.get('rms')) and _real(row.get('duration_s'), strict=True)
                                            and isinstance(row.get('flags'), list))]
    return [f'D2 lacks results for {len(bad)} of {len(ids)} jobs'] if bad else []


def payload_problems(report):
    """Each D1-D2 payload must hold the coherent, producer-shaped evidence its check exists to produce; a PASS row
    (run here or carried by --diag-resume) over missing, empty, failed, inconsistent or malformed data is not evidence."""
    return d1_problems(_dict(report.get('d1'))) + d2_problems(_dict(report.get('d2')))


def diag_verdict(report):
    """A complete verdict requires D1 and D2 PASS plus valid payloads for each (all four 5-reply A and B groups, a
    D2 result per job). Anything else (failed, WAIT, infra-failed or missing check; missing, empty or malformed
    evidence) is INCONCLUSIVE, never a frame-cap result and never PASS."""
    report = _dict(report)
    rows = {row.get('name'): row for row in _rows(report.get('checks'))}
    reasons = []
    for name in ('D1', 'D2'):
        row = rows.get(name)
        if row is None:
            reasons.append(f'{name} did not run')
        elif row.get('status') != 'PASS':
            detail = f"{row.get('status')} ({row.get('error', '')})" if row.get('error') else str(row.get('status'))
            reasons.append(f'{name} {detail}')
    reasons += [f"{row.get('name')} {row.get('status')} ({row.get('error', '')})" for name, row in rows.items()
                if name not in ('D1', 'D2') and row.get('status') != 'PASS']
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
    lines = ['# Diagnostic round D1-D2 (PR #33 full model)', '', f'Status: {status}{why}', '']
    lines += [f"- {row.get('name')}: {row.get('status')} ({row.get('error', '')}); {row.get('start')} to {row.get('end')}"
              for row in _rows(report.get('checks'))]
    groups = {str(name): _dict(group) for name, group in _dict(_dict(report.get('d1')).get('groups')).items()}
    lines += ['', '## D1 full-model frame cap (4194304 bytes / finish=length)', '', '| group | replies | finish=length | bytes=cap | capped either |', '|---|---|---|---|---|']
    for name, group in groups.items():
        lines.append(f"| {name} | {group.get('replies')} | {group.get('finish_length')} | {group.get('bytes_cap')} | {group.get('capped')} |")
    lines += ['', '| group | label | finish | bytes | duration s | RMS | capped |', '|---|---|---|---|---|---|---|']
    for name, group in groups.items():
        for r in _rows(group.get('rows')):
            lines.append(f"| {name} | {r.get('label')} | {r.get('finish')} | {r.get('bytes')} | {num(r.get('duration_s'))} | {num(r.get('rms'))} | {r.get('capped')} |")
    if status == 'complete':
        capped = {tree: sum(groups[name]['capped'] for name in DIAG_GROUPS if name.startswith(tree + '-')) for tree in ('A', 'B')}
        lines.append(f"**Verdict: frame cap reached by A {capped['A']}/10 and B {capped['B']}/10 replies.**")
    else:
        lines.append(f"**Verdict: INCONCLUSIVE** ({'; '.join(reasons)}).")
    d2 = _dict(report.get('d2'))
    lines += ['', '## D2 ASR (Whisper through the pilot code)', '']
    if d2.get('model'):
        lines += [f"- Model {d2.get('model')} on {d2.get('device')}; WER > 0.3 and RMS < 500 are flags, not failures.", '']
    lines += ['| id | WER | RMS | duration s | flags |', '|---|---|---|---|---|']
    for r in _rows(d2.get('rows')):
        flags = r.get('flags') if isinstance(r.get('flags'), list) else []
        if not _real(r.get('wer')):
            lines.append(f"| {r.get('id')} | - | {num(r.get('rms'))} | {num(r.get('duration_s'))} | {r.get('note', 'no transcript')} |")
        else:
            lines.append(f"| {r.get('id')} | {r['wer']:.3f} | {_fixed(r.get('rms'), 0)} | {num(r.get('duration_s'))} | {', '.join(map(str, flags))} |")
    lines += ['', '## Builds', '']
    for name, b in _dict(report.get('builds')).items():
        b = _dict(b)
        lines.append(f"- {name}: engine {b.get('engine_head')}, patches {b.get('patch_series')}, tensorfold-native {b.get('tensorfold_native_sha256')}, drama3-generate {b.get('drama3_generate_sha256')}")
    identity = _dict(report.get('nvfp4_identity'))
    if identity:
        lines.append(f"- NVFP4 source trees identical: {identity.get('identical')} ({identity.get('trees')})")
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
                       'builds': BUILDS, 'build_reuse': REUSE, 'nvfp4_identity': NVFP4_IDENTITY, 'd1': D1REC, 'd2': D2REC})
    report['status'], report['status_reasons'] = diag_verdict(report)
    values = [json.loads(line)['memavailable_bytes'] for p in OUT.glob('**/memory.jsonl')
              for line in p.read_text().splitlines() if line.strip()]
    report['min_memavailable_bytes'] = min(values) if values else None
    report['nvidia_peak_bytes'] = None
    report['audio'] = [str(p) for p in OUT.glob('**/*.wav')]
    report['paused'] = (OUT / 'paused.txt').read_text().splitlines() if (OUT / 'paused.txt').exists() else []
    report['resumed'] = json.loads((OUT / 'resumed.json').read_text()) if (OUT / 'resumed.json').exists() else {}
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
    # CPU-only unit test of the WER function and the dialogue reference text.
    checks = []
    checks.append(('dialogue tags removed', tag_free(DIALOGUE) == ' Is the train here?  It arrives at seven fifteen.'))
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
    """Behavioral regressions for the run-1 failure modes: memory admission, verdict honesty, sealed output
    directories and manifest-verified build reuse. CPU-only; no GPU, network or units."""
    global PLAN, OUT
    base = Path(tempfile.mkdtemp(prefix='gpuchk-recovery-'))
    saved = (PLAN, OUT, memavailable, RESULTS[:], COMMANDS[:])
    PLAN = False
    try:
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
                       run_jobs([('D1', [], refuse, ()), ('D2', [], lambda: ran.append('D2'), ())]) == REFUSED
                       and RESULTS[-1]['status'] == 'WAIT' and [r['name'] for r in RESULTS] == ['D1'] and ran == []))
        RESULTS.clear()

        def broken():
            raise RuntimeError('harness error (fixture)')

        checks.append(('genuine failure stays exit 1 even beside a WAIT',
                       run_jobs([('D1', [], broken, ()), ('D2', [], refuse, ())]) == 1))
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

        # Verdict honesty: a frame-cap result needs PASS rows AND real D1-D2 payloads for both trees.
        comparative = ('frame cap reached by',)

        def check_row(name, status):
            return {'name': name, 'prs': [], 'start': now(), 'end': now(), 'status': status,
                    'error': 'fixture failure' if status != 'PASS' else ''}

        def full_report(capped=()):
            # capped names the groups whose first two replies hit the frame cap.
            def group(name):
                rows = [{'label': f'r{i}', 'status': 200, 'finish': 'length' if name in capped and i < 2 else 'stop',
                         'bytes': 88200, 'duration_s': 1.0, 'rms': 900.0, 'capped': name in capped and i < 2} for i in range(5)]
                hits = sum(r['capped'] for r in rows)
                return {'replies': 5, 'finish_length': hits, 'bytes_cap': 0, 'capped': hits, 'rows': rows}
            return {'checks': [check_row(n, 'PASS') for n in ('D1', 'D2')], 'builds': {}, 'commands': [], 'deviations': [],
                    'd1': {'groups': {name: group(name) for name in DIAG_GROUPS}},
                    'd2': {'inputs': [{'id': 'j1'}, {'id': 'j2'}],
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
            report = full_report()
            change(report)
            return report

        # Each malformed field must be INCONCLUSIVE, never a frame-cap result, never a crash.
        d1, d2 = (lambda r: r['d1']['groups']), (lambda r: r['d2'])
        malformed = {
            'D1 extra group': lambda r: d1(r).update(extra={'replies': 5, 'rows': []}),
            'D1 groups not a dict': lambda r: r['d1'].update(groups=['x']),
            'D1 reply rms NaN': lambda r: d1(r)[DIAG_GROUPS[1]]['rows'][0].update(rms=float('nan')),
            'D1 count disagrees with replies': lambda r: d1(r)[DIAG_GROUPS[0]].update(capped=2),
            'D2 duration missing': lambda r: d2(r)['rows'][0].pop('duration_s'),
            'D2 wer bool': lambda r: d2(r)['rows'][0].update(wer=True),
            'D2 wer NaN': lambda r: d2(r)['rows'][0].update(wer=float('nan')),
            'D2 rms Inf': lambda r: d2(r)['rows'][0].update(rms=float('inf')),
            'D2 duration zero': lambda r: d2(r)['rows'][0].update(duration_s=0),
            'D2 duplicate job id': lambda r: d2(r)['inputs'].append({'id': 'j1'}),
            'D2 duplicate row': lambda r: d2(r)['rows'].append(dict(d2(r)['rows'][0])),
            'D2 row id unhashable': lambda r: d2(r)['rows'][1].update(id=['j2']),
            'whole payloads of wrong type': lambda r: r.update(d1=[], d2='x', builds=[], checks={}),
        }
        for label, change in malformed.items():
            inconclusive(f'malformed: {label}', mutated(change))
        try:
            garbage = diag_markdown('not a report')
            checks.append(('non-dict report renders INCONCLUSIVE', 'Verdict: INCONCLUSIVE' in garbage))
        except Exception:
            checks.append(('non-dict report renders INCONCLUSIVE', False))
        # Context fields (builds, reuse, identity) are not verdict evidence; malformed ones render safely.
        context = mutated(lambda r: r.update(builds={'X': None}, build_reuse={'Y': 3}, nvfp4_identity={'trees': 5},
                                             deviations=None, commands=7))
        try:
            checks.append(('malformed context fields render without throwing', 'Verdict: frame cap reached by' in diag_markdown(context)))
        except Exception:
            checks.append(('malformed context fields render without throwing', False))

        inconclusive('failed run without evidence is INCONCLUSIVE',
                     {'checks': [check_row(n, 'FAIL') for n in ('D1', 'D2')], 'builds': {}, 'd1': {},
                      'commands': [], 'deviations': []}, 'D1 group A-full')
        no_b = full_report()
        for name in DIAG_GROUPS[2:]:
            del no_b['d1']['groups'][name]
        inconclusive('missing B groups stay INCONCLUSIVE', no_b, 'D1 group B-full')
        no_a = full_report()
        del no_a['d1']['groups'][DIAG_GROUPS[0]]
        inconclusive('a missing A group stays INCONCLUSIVE', no_a, 'D1 group A-full')
        partial = full_report()
        partial['checks'][0] = check_row('D1', 'FAIL')
        inconclusive('one failed check stays INCONCLUSIVE', partial, 'D1 FAIL')
        waited = full_report()
        waited['checks'][1] = check_row('D2', 'WAIT')
        inconclusive('a WAIT check stays INCONCLUSIVE', waited, 'D2 WAIT')
        # PASS rows (as carried by --diag-resume) over empty payloads are not evidence.
        hollow = {'checks': [check_row(n, 'PASS') for n in ('D1', 'D2')], 'builds': {}, 'commands': [], 'deviations': [],
                  'd1': {}, 'd2': {}}
        status, reasons = diag_verdict(hollow)
        checks.append(('carried PASS rows over empty D1-D2 are INCONCLUSIVE',
                       status == 'INCONCLUSIVE' and all(any(r.startswith(d) for r in reasons) for d in ('D1', 'D2'))
                       and not any(p in diag_markdown(hollow) for p in comparative)))
        failed_replies = full_report()
        failed_replies['d1']['groups'][DIAG_GROUPS[0]]['rows'] = [dict(r, status=500) for r in failed_replies['d1']['groups'][DIAG_GROUPS[0]]['rows']]
        inconclusive('failed D1 replies are INCONCLUSIVE', failed_replies, 'has invalid replies')
        empty_reply = full_report()
        empty_reply['d1']['groups'][DIAG_GROUPS[3]]['rows'][0]['bytes'] = 0
        inconclusive('an empty D1 reply is INCONCLUSIVE', empty_reply, 'has invalid replies')
        bad_d2 = full_report()
        bad_d2['d2']['rows'][1] = {'id': 'j2', 'note': 'wav absent'}
        inconclusive('D2 job without a result is INCONCLUSIVE', bad_d2, 'D2 lacks results for 1 of 2')
        empty_d2 = full_report()
        empty_d2['d2'] = {'inputs': [], 'rows': []}
        inconclusive('D2 with no jobs is INCONCLUSIVE', empty_d2, 'D2 has no jobs')
        complete = full_report(capped=(DIAG_GROUPS[0],))
        status, reasons = diag_verdict(complete)
        checks.append(('completed A/B run renders the frame-cap result per tree',
                       status == 'complete' and '**Verdict: frame cap reached by A 2/10 and B 0/10 replies.**' in diag_markdown(complete)))

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
                       build_reusable(build, 'abc123', manifest, True, 'abc123') == (True, 'verified against the manifest')))
        # A retained build from an earlier pin: stamp, manifest and hashes all agree, but it is not the pinned HEAD.
        usable, reason = build_reusable(build, 'abc123', manifest, True, 'pinned999')
        checks.append(('build reuse refuses a stale build from another pin', not usable and 'is not the pinned' in reason))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, tensorfold_native_sha256='0' * 64), True, 'abc123')
        checks.append(('build reuse rejects a hash mismatch', not usable and 'tensorfold-native sha256 differs' in reason))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, engine_head='def456'), True, 'abc123')
        checks.append(('build reuse rejects an engine head mismatch', not usable and 'engine_head differs' in reason))
        usable, reason = build_reusable(build, 'abc123', None, True, 'abc123')
        checks.append(('build reuse rejects a missing manifest entry', not usable and 'no manifest entry' in reason))
        usable, reason = build_reusable(build, 'abc123', dict(manifest, drama3_generate_sha256=hashlib.sha256(b'other').hexdigest()), True, 'abc123')
        checks.append(('build reuse rejects a drama3-generate mismatch', not usable and 'drama3-generate sha256 differs' in reason))
        usable, reason = build_reusable(build, 'other', manifest, False, 'other')
        checks.append(('build reuse needs the stamp without a manifest', not usable))
        usable, reason = build_reusable(build, 'abc123', None, False, 'abc123')
        checks.append(('build reuse rebuilds a matching stamp without a manifest',
                       not usable and 'no manifest in force' in reason))
        # A prep-time abort (e.g. a NVFP4 source mismatch) still writes an INCONCLUSIVE summary with its reason.
        OUT = base / 'prep-abort'
        OUT.mkdir()
        RESULTS.clear()
        real_prep = globals()['prep']
        globals()['prep'] = lambda: (_ for _ in ()).throw(RuntimeError('NVFP4 sources differ (fixture)'))
        try:
            prepared = prepare()
        finally:
            globals()['prep'] = real_prep
        report = json.loads((OUT / 'summary.json').read_text()) if (OUT / 'summary.json').exists() else {}
        checks.append(('prep abort writes an INCONCLUSIVE summary with its reason',
                       not prepared and report.get('status') == 'INCONCLUSIVE'
                       and any('NVFP4 sources differ' in r for r in report.get('status_reasons', []))))
        RESULTS.clear()
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
    so only the missing D1-D2 checks execute; the prior out dir's wavs stay readable for D2."""
    prior = json.loads(Path(DIAG_RESUME).read_text())
    done = {row['name']: row for row in prior.get('checks', []) if row.get('status') == 'PASS'}
    kept = [job for job in jobs if job[0] not in done]
    RESULTS.extend(done[name] for name, _, _, _ in jobs if name in done)
    if 'D1' in done:
        D1REC.update(prior.get('d1') or {})
    if 'D2' in done:
        D2REC.update(prior.get('d2') or {})
    if not kept:
        summary()
    return kept


def diag_main():
    if not PLAN:
        device_preflight()
        claim_out()
        ROOT.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if not PLAN and not memory_preflight():
        return REFUSED
    if not prepare():
        return 1
    acquire_gpu()
    jobs = [('D1', [33], d1, ()), ('D2', [33], d2, ())]
    if DIAG_RESUME and not PLAN:
        jobs = carry_resume(jobs)
    return run_jobs(jobs)


def prepare():
    """Sources, NVFP4 identity and builds. Any abort here (before GPU work) is recorded as a failed prep row and
    written to an INCONCLUSIVE summary with its reason, instead of leaving an empty output directory."""
    try:
        prep()
        record_builds()
        return True
    except Exception as error:
        RESULTS.append({'name': 'prep', 'prs': [33], 'start': now(), 'end': now(), 'status': 'FAIL',
                        'error': f'{type(error).__name__}: {error}'})
        log(RESULTS[-1]['error'])
        summary()
        return False


# Clean refusal (memory floor not met, nothing launched): run-diag.sh exits 0 so systemd records no failure.
REFUSED = 75


def memory_preflight():
    """First gate, before any build, pause, lock or pilot: below the full-profile floor record WAIT, write the
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
        report.update({'checks': RESULTS, 'commands': COMMANDS, 'deviations': DEVIATIONS, 'builds': BUILDS,
                       'build_reuse': REUSE, 'nvfp4_identity': NVFP4_IDENTITY, 'g1_gate': GATE})
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
    checks = report.get('checks', [])
    names = [row.get('name') for row in checks]
    reasons = [f"{row.get('name')} {row.get('status')} ({row.get('error', '')})" for row in checks if row.get('status') != 'PASS']
    reasons += [f'{name} did not run' for name, *_ in MAIN_JOBS if name not in names]
    report['status'], report['status_reasons'] = ('INCONCLUSIVE', reasons) if reasons else ('complete', [])
    report['per_pr'] = {'33': 'FAIL' if any(row['status'] == 'FAIL' and row['name'] != 'prep' for row in checks) else
                        ('PASS' if sorted(names) == sorted(name for name, *_ in MAIN_JOBS) else 'INCOMPLETE')}
    save(OUT / 'summary.json', report)
    lines = ['# GPU checks: PR #33 full model, A = main, B = TensorFold v1.1.9', '',
             f"Status: {report['status']}" + (' — ' + '; '.join(reasons) if reasons else ''), '',
             f"G1 byte identity A vs B: {GATE.get('identical', 'not measured')}", '', 'Listening: pending owner', '', '## Automatic checks']
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
    if not prepare():
        return 1
    acquire_gpu()
    return run_jobs(MAIN_JOBS)


# PR #33 full-model A/B; NVFP4 is identical in both trees (see NVFP4_NOTE) and is not compared.
MAIN_JOBS = [('G1-full', [33], g1, ()), ('G2-full', [33], g2, ()), ('G3-full', [33], g3, ())]


if __name__ == '__main__':
    raise SystemExit(main())
