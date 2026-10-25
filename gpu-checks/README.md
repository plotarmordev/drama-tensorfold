# PR GPU checks

`run-all.sh --plan` prints build, pilot, benchmark, encoder, catalog and HTTP request commands without running subprocesses or writing files. HTTP command lines are equivalent curl representations of the stdlib client's JSON requests; the runner uses urllib to retain status and headers.

Run `GPU_CHECK_SSH_HOST=<device> gpu-checks/sync-to-device.sh` on the workstation after committing. It sends a
git bundle, these scripts and `bench/corpus/drama-bench-en-v1.jsonl` (the device host has no git checkout, so the
corpus travels with the scripts) to the same work-directory path on the device host, without pushing or starting
anything. On the device host, launch from that work directory with the settings below:

```sh
W=/path/to/work-directory
systemd-run --user --unit=gpu-checks \
  -p WorkingDirectory="$W" \
  -p EnvironmentFile="$W/gpu-checks.env" \
  -p StandardOutput="append:$W/run.log" \
  -p StandardError="append:$W/run.log" \
  -p "ExecStopPost=$W/gpu-checks/resume-paused.sh" \
  "$W/gpu-checks/run-all.sh"
```

### Device configuration

Host-local values come only from the environment (e.g. an untracked `gpu-checks.env` passed as `EnvironmentFile`);
nothing host-specific is committed. A device run refuses before claiming its output directory unless every required
setting is present and the host's `uname -n` equals `GPU_CHECK_HOST`. `--plan` prints unset values as `<NAME>`
placeholders and the work directory as `$W`.

| Setting | Required | Meaning |
|---|---|---|
| `GPU_CHECK_HOST` | yes | `uname -n` of the only host allowed to run devices |
| `GPU_CHECK_LOCK` | yes | the GPU lock file every GPU workload on the host shares (`flock`) |
| `GPU_CHECK_PILOT_ROOT` | yes | pilot checkout with `checkpoint/` (full weights) and `venv/` |
| `GPU_CHECK_NVFP4_CHECKPOINT` | yes | NVFP4 checkpoint directory |
| `GPU_CHECK_ZIG` | yes | Zig 0.17 compiler binary |
| `GPU_CHECK_ASR_DIR` | diagnostic runs | the pilot's Whisper code directory (`transcribe.py`), read-only |
| `GPU_CHECK_NVCC` | no | defaults to `/usr/local/cuda/bin/nvcc` |
| `GPU_CHECK_PAUSE_ALLOW` | no | space-separated unit names or fnmatch patterns that admission may SIGSTOP; unset pauses nothing |
| `GPU_CHECK_TOLERATED_PORT` | no | a GPU process whose argv has `--port` this value is not foreign; unset tolerates none |

The runner refuses every existing output path — with or without a summary, directory or file — without changing
its bytes (run-1 `out-diag/` can never be overwritten); every run or retry takes a fresh `--out-dir`. Builds are reused only when a build manifest is
in force (`--build-manifest PATH`, e.g. a prior summary.json, or the runner-written `root/build-manifest.json`)
and the recorded engine head and both binary sha256s verify against the stamped checkout — a stamp alone proves
source identity, not binary integrity, so without a manifest exactly that build is rebuilt. Patch manifests and
source HEADs are checked before reuse.

Admission briefly takes the shared GPU lock, rejects foreign compute processes, checks checkpoint memory plus reserve, and records active units matching `GPU_CHECK_PAUSE_ALLOW` before stopping them. The lock is released before pilots start. A pilot that loses the lock race is retried after blocking on that same lock. Pilots own server shutdown and retain their lock until shutdown is confirmed. The EXIT trap and ExecStopPost both invoke idempotent recovery from `out/paused.txt`; recovery resumes every recorded unit even if the allowlist has since changed. Never remove this file before recovery. SIGKILL cannot run shell traps; the systemd ExecStopPost is required.

`selftest.sh` creates only `gpuchk-selftest-sleep`, uses an isolated temporary evidence directory, checks stopped and sleeping process states, tests repeat recovery, and stops its own dummy. It refuses an already loaded unit with that name. No GPU query or GPU lock is used in this test.

Synthetic reference clips originate from the candidate's built-in voice. The full engine's codec encoder runs under the shared lock after the rendering pilot exits. `build_voice.py` validates attested frame-major reference catalogs; CUDA legacy catalogs are translated to their documented codebook-major format with fresh code hashes. Untagged catalogs alone are loaded by the baseline. No existing personal catalog is read.

## Diagnostic round D1-D4

`run-diag.sh` (or `runner.py --diag`; `--diag --plan` prints every command and nothing else, saved as
`diag-plan-output.txt`) reruns the deciding long-form diagnostics through the same prep, build reuse, admission,
pause/resume and pilot machinery. Evidence lands in `out-diag/` by default (`--out-dir`/`GPU_CHECK_OUT` to
override); any existing output path is refused unchanged, so run-1 `out-diag/` is never rewritten;
run 2's `out/` is only ever read. A corpus preflight runs FIRST, before any build, unit pause, GPU lock or pilot:
`bench/corpus/drama-bench-en-v1.jsonl` must exist and match the sha256 pinned in `runner.py`
(`CORPUS_SHA256`), else the run records an infrastructure failure, writes an INCONCLUSIVE summary and exits
non-zero without starting anything. Right after the corpus gate, a launch preflight checks MemAvailable against the
same 88 GiB floor before any build, pause, GPU lock or pilot; below it the runner writes an INCONCLUSIVE summary
(`preflight-memory: WAIT`) and exits 75, which `run-diag.sh` maps to exit 0 so the unit ends inactive instead of
failed. MemAvailable is rechecked against the pilot's own 88 GiB floor (never lowered) immediately before every
full-profile pilot launch, including the retry after a lost lock race; below it no attempt log or evidence path
is created, nothing launches, no GPU lock is held while checking, and no scoring process is paused or killed to
make room. A late WAIT stops every later check and the run exits 75 (mapped to 0); a genuine corpus, hash,
harness or code failure exits 1. A complete verdict requires every D1-D4 check PASS plus valid payloads: HTTP-200,
nonempty A and B D1 observations with N/M evidence, all 8 D2 seeds with carry evidence, all three 5-reply D3
groups, and a WER/RMS result for every D4 job. Any failed, WAITed or missing check, or missing, empty or malformed
evidence (including PASS rows carried by `--diag-resume`), renders the whole run INCONCLUSIVE (`status` in
`summary.json`), never "as/more/less often" and never PASS.
D1 compares long-form early stops A vs B with default request fields (the full-profile engine has no
long-form mode, so the NVFP4 family carries both trees and the full binaries' own `--long-form` refusals are
recorded); D2 replays the exact G4 request with seeds 1-8 and verifies speaker carry by the CPU probe on the
candidate's cuts (the server logs no per-segment prompts); D3 counts `finish=length` / 4194304-byte replies on
B full (tagged and untagged, seeded) and A full (untagged, unseeded: A rejects the seed field); D4 scores Whisper
word error rates for every run-2 and diagnostic wav through the pilot's `transcribe.py` (`GPU_CHECK_ASR_DIR`) from its venv.
Diagnostic observations never fail the run; only infrastructure errors exit non-zero. `runner.py --diag-selftest`
unit-tests the WER function and the D1 text preparation, plus behavioral regressions for the run-1 failure modes
(corpus preflight, memory admission, verdict honesty, sealed output dirs, manifest-verified build reuse).
Launch on the device host:

```sh
W=/path/to/work-directory
systemd-run --user --unit=gpu-checks-diag \
  -p WorkingDirectory="$W" \
  -p EnvironmentFile="$W/gpu-checks.env" \
  -p StandardOutput="append:$W/run-diag.log" \
  -p StandardError="append:$W/run-diag.log" \
  -p "ExecStopPost=$W/gpu-checks/resume-paused.sh" \
  "$W/gpu-checks/run-diag.sh"
```

`resume-paused.sh` recovers `out-diag/paused.txt` when it exists, then `out/paused.txt`.

### Repair round r2 (run-1 recovery)

Run 1 (`out-diag/`) failed on infrastructure: no corpus on the device host (D1/D2/D4) and
82.82 GiB MemAvailable against the pilot's 88.00 GiB floor (D3). Its evidence is preserved read-only. The repair
runs as its own unit and output directory, reusing run 1's retained builds only after the recorded hashes verify
(`--build-manifest` accepts run 1's `summary.json` directly) and executing only the missing D1-D4 checks
(`--diag-resume` carries prior PASS checks into the new summary and re-runs the rest):

```sh
W=/path/to/work-directory
systemd-run --user --unit=gpu-checks-diag-r2 \
  -p WorkingDirectory="$W" \
  -p EnvironmentFile="$W/gpu-checks.env" \
  -p StandardOutput="append:$W/run-diag-r2.log" \
  -p StandardError="append:$W/run-diag-r2.log" \
  -p Environment=GPU_CHECK_OUT="$W/out-diag-r2" \
  -p "ExecStopPost=$W/gpu-checks/resume-paused.sh" \
  "$W/gpu-checks/run-diag.sh" --out-dir "$W/out-diag-r2" \
    --build-manifest "$W/out-diag/summary.json" \
    --diag-resume "$W/out-diag/summary.json"
```

`run-diag.sh` exports `GPU_CHECK_OUT` from `--out-dir` so the EXIT trap and ExecStopPost recover the same
`out-diag-r2/paused.txt`. A failed corpus preflight seals its own output directory with the INCONCLUSIVE summary
it writes; retry with a fresh `--out-dir`.

### Finisher contract

The paused finisher automation targets the repair round only: it reads `out-diag-r2/summary.json` from unit
`gpu-checks-diag-r2` and appends `## Diagnostic round D1-D4` to the PR thread solely when that summary's
top-level `status` is `complete`. An `INCONCLUSIVE` status (missing/mismatched corpus, memory WAIT, failed or
missing check, zero A or B observations) is infrastructure, never a per-PR verdict for #26-#28, and is never
rewritten into one. Run-1 `out-diag/` is never appended to or overwritten.

## Evidence and limits

Automatic results and PR attribution are in `out/summary.json` and `out/summary.md`; raw HTTP requests, headers, PCM, WAV, hashes, timings, benchmark reports and pilot memory logs remain alongside each check. Listening remains **pending owner**, including intelligibility, joins and stable distinct speakers. Integer PCM cannot encode NaN; the silence check only establishes nonzero RMS, not audible speech quality.

The long-form header exposes audio sample offsets, not text boundaries. Automatic gates check successful PCM and increasing in-range segment offsets. During prep, a CPU Zig probe imports the candidate's actual shift splitter, span collector and speaker continuation function, using the exact HTTP request texts. It checks protected-span boundaries with shift pauses both disabled and enabled, and speaker prefixes across continuation segments. Probe output is retained in `out/cpu-probe.json`; assertion failures fail the associated automatic gates. If the standalone probe cannot compile against the engine's module graph, text-cut placement and continuation-speaker carry are explicitly **unavailable (not verified)**, as authorized by the coordinator. Neither assertion is inferred from audio sample offsets.

ROOT remains under the assigned work directory because the neutral system root requires privileges. Host-local paths come from the device settings and are not anonymized in device-side evidence. Host memory samples report minimum MemAvailable and the largest observed available-memory drop, not process peak RSS. NVIDIA peak allocation is unavailable. These deviations are repeated in generated summaries.

No GPU behavior is claimed by CPU syntax checks, the plan, or the dummy-unit selftest.
