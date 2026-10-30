# PR GPU checks

These checks compare the full model on two trees: A is `main` at `f68738ea` (TensorFold `d1018b31` plus
full-model patches 0001-0015 and the CUDA NVFP4 series) and B is PR #33's TensorFold v1.1.9 tree at `3f44de62`
(no full-model patches). Both trees pin the same NVFP4 engine and patch series, so the runner builds both NVFP4
engines only to prove their patched source trees identical (`git rev-parse HEAD^{tree}`), says so in its plan and
summary, and runs no NVFP4 A/B check.

`run-all.sh --plan` prints build, pilot, benchmark, encoder, catalog and HTTP request commands without running subprocesses or writing files. HTTP command lines are equivalent curl representations of the stdlib client's JSON requests; the runner uses urllib to retain status and headers.

Run `GPU_CHECK_SSH_HOST=<device> gpu-checks/sync-to-device.sh` on the workstation after committing. It sends a
git bundle holding both pinned trees and these scripts to the same work-directory path on the device host,
without pushing or starting anything. Each tree's checkout carries its own benchmark corpus. On the device host,
launch from that work directory with the settings below:

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

The run has three checks, each on both trees with the same requests:

- **G1-full:** two corpus capture passes per tree compared by `compare_gate.py`, the streaming check, and the
  reference-cache transitions with the shared synthetic catalog.
- **G2-full:** request sampling (patches 0013/0014 on A, upstream #1277 on B): seed repeatability for 42,
  2^53+1 and 2^64-1, `temperature`, `top_k`, `top_p`, and a 400 for `top_k` 65.
- **G3-full:** speaker tags (patch 0015 on A, upstream #1278 on B): the tagged and untagged dialogue line with
  the builtin voice and the tagged `dialogue` catalog voice.

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
| `GPU_CHECK_ZIG` | yes | Zig 0.17 compiler binary |
| `GPU_CHECK_ASR_DIR` | diagnostic runs | the pilot's Whisper code directory (`transcribe.py`), read-only |
| `GPU_CHECK_NVCC` | no | defaults to `/usr/local/cuda/bin/nvcc` |
| `GPU_CHECK_PAUSE_ALLOW` | no | space-separated unit names or fnmatch patterns that admission may SIGSTOP; unset pauses nothing |
| `GPU_CHECK_TOLERATED_PORT` | no | a GPU process whose argv has `--port` this value is not foreign; unset tolerates none |

The runner refuses every existing output path — with or without a summary, directory or file — without changing
its bytes; every run or retry takes a fresh `--out-dir`. Builds are reused only when a build manifest is
in force (`--build-manifest PATH`, e.g. a prior summary.json, or the runner-written `root/build-manifest.json`)
and the recorded engine head and both binary sha256s verify against the stamped checkout — a stamp alone proves
source identity, not binary integrity, so without a manifest exactly that build is rebuilt. Patch manifests and
source HEADs are checked before reuse: A's full and NVFP4 series and B's NVFP4 series by `sha256sum -c`; B's
full model is unpatched by contract (its empty manifest is still checked by `scripts/build-engine.sh`).

Admission briefly takes the shared GPU lock, rejects foreign compute processes, checks checkpoint memory plus reserve, and records active units matching `GPU_CHECK_PAUSE_ALLOW` before stopping them. The lock is released before pilots start. A pilot that loses the lock race is retried after blocking on that same lock. Pilots own server shutdown and retain their lock until shutdown is confirmed. The EXIT trap and ExecStopPost both invoke idempotent recovery from `out/paused.txt`; recovery resumes every recorded unit even if the allowlist has since changed. Never remove this file before recovery. SIGKILL cannot run shell traps; the systemd ExecStopPost is required.

`selftest.sh` creates only `gpuchk-selftest-sleep`, uses an isolated temporary evidence directory, checks stopped and sleeping process states, tests repeat recovery, and stops its own dummy. It refuses an already loaded unit with that name. No GPU query or GPU lock is used in this test.

Synthetic reference clips originate from B's built-in voice. The full engine's codec encoder runs under the shared lock after the rendering pilot exits. `build_voice.py` validates attested frame-major reference catalogs; both trees load the same catalog, including the tagged `dialogue` voice. No existing personal catalog is read.

## Diagnostic round D1-D2

`run-diag.sh` (or `runner.py --diag`; `--diag --plan` prints every command and nothing else, saved as
`diag-plan-output.txt`) runs the full-model stopping diagnostics for issue #32 through the same prep, build reuse,
admission, pause/resume and pilot machinery. Evidence lands in `out-diag/` by default (`--out-dir`/`GPU_CHECK_OUT`
to override); any existing output path is refused unchanged. A launch preflight checks MemAvailable against the
88 GiB floor before any build, pause, GPU lock or pilot; below it the runner writes an INCONCLUSIVE summary
(`preflight-memory: WAIT`) and exits 75, which `run-diag.sh` maps to exit 0 so the unit ends inactive instead of
failed. MemAvailable is rechecked against the pilot's own 88 GiB floor (never lowered) immediately before every
full-profile pilot launch, including the retry after a lost lock race; below it no attempt log or evidence path
is created, nothing launches, no GPU lock is held while checking, and no scoring process is paused or killed to
make room. A late WAIT stops every later check and the run exits 75 (mapped to 0); a genuine hash, harness or
code failure exits 1.

D1 sends the same dialogue line, tagged and untagged, with the `dialogue` catalog voice and seeds 1-5 to each
tree, and counts `finish=length` / 4194304-byte replies per group. D2 scores Whisper word error rates for every D1
reply through the pilot's `transcribe.py` (`GPU_CHECK_ASR_DIR`) from its venv. A complete verdict requires D1 and
D2 PASS plus valid payloads: all four 5-reply A and B groups with HTTP-200 replies and consistent counts, and a
WER/RMS result for every D2 job. Any failed, WAITed or missing check, or missing, empty or malformed evidence
(including PASS rows carried by `--diag-resume`), renders the whole run INCONCLUSIVE (`status` in `summary.json`)
and never a frame-cap result. Diagnostic observations never fail the run; only infrastructure errors exit
non-zero. `runner.py --diag-selftest` unit-tests the WER function and the dialogue reference text, plus
behavioral regressions for memory admission, verdict honesty, sealed output dirs and manifest-verified build reuse.
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

`resume-paused.sh` recovers `out-diag/paused.txt` when it exists, then `out/paused.txt`. A retry after an
interrupted diagnostic run takes a fresh `--out-dir`, may reuse verified builds with `--build-manifest
PRIOR/summary.json`, and re-executes only the checks that did not pass with `--diag-resume PRIOR/summary.json`;
`run-diag.sh` exports `GPU_CHECK_OUT` from `--out-dir` so the EXIT trap and ExecStopPost recover the same
`paused.txt`.

## Evidence and limits

Automatic results are in `out/summary.json` and `out/summary.md`; raw HTTP requests, headers, PCM, WAV, hashes, timings, benchmark reports and pilot memory logs remain alongside each check. Listening remains **pending owner**, including intelligibility and stable distinct speakers. Integer PCM cannot encode NaN; the silence check only establishes nonzero RMS, not audible speech quality.

ROOT remains under the assigned work directory because the neutral system root requires privileges. Host-local paths come from the device settings and are not anonymized in device-side evidence. Host memory samples report minimum MemAvailable and the largest observed available-memory drop, not process peak RSS. NVIDIA peak allocation is unavailable. These deviations are repeated in generated summaries.

No GPU behavior is claimed by CPU syntax checks, the plan, or the dummy-unit selftest.
