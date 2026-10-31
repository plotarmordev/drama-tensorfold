# macOS: running the pinned recipe on Apple silicon

Both `--weights full` and `--weights nvfp4` use the native Zig/Metal engine, not Python/MLX.
The full model uses the Linux recipe's pin, TensorFold v1.1.9, with no patches. NVFP4 uses the separate
[Mac engine pin](../engine/nvfp4/macos/UPSTREAM) and patch series; the CUDA NVFP4 pin is unchanged.
Neither path requires CUDA, `nvcc`, systemd, or `/proc`. Use `recipe/pilot_macos.py`.

This is a local, cooperative pilot—not a production service installer, GPU reservation
system, memory sandbox, or proof that the whole host is idle. Do not interrupt another
application to make an admission check pass.

## Prerequisites and build

- Apple silicon with Xcode command-line tools, including `xcrun metal`.
- Zig **0.17.0**. The official `zig-aarch64-macos-0.17.0.tar.xz` archive has SHA-256
  `b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a`.
- Python 3.9 or newer for the stdlib-only launcher and validation tools.
- The Hugging Face CLI (`hf`) on `PATH`. Install it in an isolated Python environment before downloading:

  ```sh
  python3 -m venv .venv-hf
  . .venv-hf/bin/activate
  python -m pip install huggingface_hub
  ```

- Approximately 21 GiB for the NVFP4 checkpoint, or 39 GiB for the full checkpoint, plus
  build/cache space and sufficient free + inactive memory. Disk size is not a resident-memory prediction.
- Permission to use the checkpoint under the [pinned model license](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE).

From the repository root, for the NVFP4 checkpoint:

```sh
hf download plotarmordev/drama-3-NVFP4-Pruned \
  --revision 494db08481db952f834b2241bce63fb3dfa72554 --local-dir drama-3-NVFP4-Pruned
(cd drama-3-NVFP4-Pruned && shasum -a 256 -c ../recipe/checksums/nvfp4.sha256)
ZIG=/absolute/path/to/zig-0.17.0/zig JOBS=2 scripts/build-engine.sh --weights nvfp4
```

All fifteen checkpoint files must report `OK`. The launcher also verifies the selected manifest
on every start, including an explicit `--checkpoint` override; an incomplete or changed directory
is refused rather than repaired. An absent default directory can be downloaded automatically.
The build verifies its ordered patch checksums and does not require a CUDA compiler on Darwin.
Outputs are `build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native` and
`build/tensorfold-nvfp4/zig-out/bin/{drama3-generate,drama3-voice}`.

For the full-model recipe (the Mac build at TensorFold v1.1.9 has not been run yet):

```sh
hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir drama-3
(cd drama-3 && shasum -a 256 -c ../recipe/checksums/checkpoint.sha256)
ZIG=/absolute/path/to/zig-0.17.0/zig JOBS=2 scripts/build-engine.sh --weights full
```

All ten full-model files must report `OK`. Its server and `drama3-generate` remain under
`build/tensorfold/zig-out/`. A build directory belonging to another engine repository is refused:
pass a fresh positional `BUILD_DIR`, then select its server explicitly with `--native-bin`.

Use the installed build's `serve --help` for its supported options; do not assume features
from newer upstream releases exist in this pinned tree.

## Launch and stop

First coordinate with other GPU users. Use the **same agreed lock file** as other cooperating
launchers; a newly invented private lock does not exclude unrelated workloads. For an otherwise
unshared local pilot, the following creates its local runtime directory and lock:

```sh
mkdir -p runtime
touch runtime/gpu.lock
python3 recipe/pilot_macos.py --weights nvfp4 \
  --runtime-dir runtime --gpu-lock runtime/gpu.lock --evidence-dir runs/macos-pilot-01
```

The default bind address is **127.0.0.1**, port **18793**; `--host` accepts numeric IPv4 addresses. An occupied port or lock is a refusal,
not permission to displace its owner. This unauthenticated API should not be exposed casually;
LAN placement does not exempt a hosted service from the model license.

Use `--weights full` for the full model. Explicit `--native-bin` and `--checkpoint` remain available;
with `--weights` the selected checkpoint manifest is still mandatory. Without `--weights`, the legacy
explicit-binary/checkpoint mode makes no pinned-manifest verification claim. `session.json` records
the selection, checkpoint revision, manifest, actual binary and command.

NVFP4 defaults to `--cfg-gamma 1.5 --cfg-batched`. On Metal, paired guidance means two independent
backbone states submitted in one command buffer, sharing weights; it is not a two-row/vectorized GEMM.
Only semantic logits are guided; residual codec heads stay conditional. Builtin voices remain unguided.
Admission conservatively budgets the additional guidance state. To replace these defaults, pass
`-- --cfg-gamma 1.5` for separate submissions or `-- --cfg-gamma 0` to disable guidance.
An explicit `--cfg-gamma`, `--cfg-batched`, or `--cfg-allow-request-gamma` replaces the default group.
Finite gamma values 1–5 are supported, with 1 preserving conditional logits; 0 disables the policy.
Request `X-Cfg-Gamma` overrides require `--cfg-allow-request-gamma`. `--cfg-every` is unsupported.

The launcher records `session.json`, `server.log`, `memory.jsonl`, `ready.json` and
`completion.json`. `DRAMA_PILOT_READY` means the launched server passed its health check, not
that audio correctness or performance has been verified. Explicit `--evidence-reuse` clears stale
readiness after acquiring the lock, even when admission then fails. Without it an existing evidence
directory is refused. Prefer a new evidence directory for each attempt.

| Guard | macOS behavior |
|---|---|
| Admission | max(checkpoint bytes + 30 GiB, 34 GiB) of free + inactive memory |
| Memory observation | One `vm_stat` snapshot per sample; free + inactive only |
| Abort floor | 34 GiB; admission overrides cannot be lower |
| Isolation | Owned child process group and inherited cooperative lock |
| Memory cap | None: Metal has no equivalent of the Linux cgroup/CUDA cap |

`vm_stat` speculative pages are already included in free pages. Purgeable pages are not added;
these categories must not be counted twice. `--min-available-gib` is an explicit admission
override, not a memory cap; it must be finite and at least the abort floor. Other numeric
controls must also be finite and valid.

Stop the launcher with Ctrl-C or SIGTERM. A supervisor that starts the launcher can pass
`--parent-pid` with its own PID: macOS has no parent-death signal, so the launcher checks its
parent on each sample and stops the server, as for SIGTERM, once that process is gone. A PID that
is not the launcher's parent is refused before anything starts. It settles only its own process group, with bounded
TERM/KILL cleanup, and retains the unreaped leader identity while checking descendants.
An unexpected server exit is a failure even when the server's exit status is zero. Observation
errors fail closed. The server inherits the lock descriptor, so killing only the launcher
cannot release the lock while that server is still running. This is not an OS sandbox:
escaped descendants or unkillable processes require operator attention and must not be reported
as settled. Read `completion.json` before starting another attempt.

No LaunchAgent is installed by this repository. Do not copy unrelated host-specific service
names or restart commands into an automated setup.

## Speech requests

```sh
curl --fail-with-body -sS -D headers.txt http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"The train leaves at seven fifteen."}' \
  -o hello.wav
afplay hello.wav
```

For streaming, add `"response_format":"pcm","stream":true` and use `curl -N`. The body is
mono signed 16-bit little-endian PCM at 44.1 kHz; `afplay` needs a suitable container such as WAV.
Check the HTTP status and `X-Finish-Reason` rather than treating any saved file as successful audio.

## Attested reference voices

Use a synthetic voice, your own voice, or a voice for which you have the required permission.
The caller must attest the source explicitly; the catalog writer does not invent consent.
Reference audio, codec rows, and generated voice catalogs stay out of Git.

For the NVFP4 Mac engine, use its native attested catalog writer:

```sh
mkdir -p runtime/voices
build/tensorfold-nvfp4/zig-out/bin/drama3-voice \
  --codec drama-3-NVFP4-Pruned/codec.pth --ref-audio clip.wav --ref-text "$(cat clip.txt)" \
  --source-kind self --source-note "My own recording; permission retained locally." \
  --out runtime/voices/my-voice.json \
  --generate "$PWD/build/tensorfold-nvfp4/zig-out/bin/drama3-generate"
```

`--ref-text` takes literal text, so the command reads the exact transcript from `clip.txt`. Encoding
is GPU work and needs a coordinated window with the speech server stopped. The writer emits
`drama3-http-voice/1`; the older CUDA quant engine's `tools/zig/drama_voice.py` is not a Mac encoder.
The Mac speech API accepts attested catalog voices, not request-supplied reference audio or codes.

For the full-model engine, encode the reference and write the same attested catalog format:

```sh
build/tensorfold/zig-out/bin/drama3-generate --codec drama-3/codec.pth \
  --encode clip.codes --ref-audio clip.wav
python3 recipe/build_voice.py --source self my-voice \
  "My own recording; permission retained locally." "Exact transcript of clip.wav:clip.codes"
```

`--source` is required and accepts `synthetic`, `self`, or `consented`. `--output-dir` defaults
to `runtime/voices`. The tool refuses to overwrite an existing catalog and validates the server's
contract before publishing a file:

- ASCII name of 1–64 bytes, beginning with a letter/digit, then letters/digits/`-`/`_`;
  `default` is reserved, case-insensitively.
- Nonblank UTF-8 source note and transcripts, each at most 4096 bytes. Transcripts cannot
  contain structural prompt tokens or unsupported/malformed speaker tags.
- One to four references; each has 1–646 complete rows of ten little-endian int32 codes.
- Codebook zero: 0–4095; codebooks one through nine: 0–1023; catalog at most 256 KiB.

Start a new coordinated pilot with `-- --voices /absolute/path/to/runtime/voices`, then request
`"voice":"my-voice"`. Paths after `--` are server arguments and are resolved from its runtime
directory. `GET /v1/voices` lists loaded voices. Without `--default-voice`, omitted/default
requests use the builtin voice. Setting `--default-voice NAME` changes both cases.

The full-model engine (upstream since TensorFold v1.1.5) and the NVFP4 Mac base reuse one conditional reference-prefix snapshot per
Metal lane. A different or absent prefix invalidates it before prefill; switching voices reuses the
allocated snapshot buffers. Guided requests reset their independent unconditional backbone state;
cancellation invalidates conditional retention before another request. Neither is a multi-voice cache.
The NVFP4 checkpoint stays packed: E2M1 expert codes, block-16 E4M3 scales and F32 global scales,
with BF16 dense/tower weights and FP32 accumulation. No checkpoint conversion or BF16 expert expansion
is performed. Pruned routing masks dropped original IDs before softmax and retains original-ID tie order.

## Correctness gates

CPU tests and bounded subprocess smoke are separate from model work:

```sh
python3 -m unittest discover -s recipe -p 'test_*.py'
PILOT_MACOS_INTEGRATION=1 python3 -m unittest discover -s recipe -p test_pilot_macos.py
```

The opt-in lifecycle cases use harmless short-lived subprocesses and dummy checkpoint files,
not model weights. They exercise real process groups, locks, loopback HTTP and cancellation.
Never run the following model checks without a coordinated compute window.

For a freshly started single-lane server with builtin default and two attested catalog voices:

```sh
python3 bench/reference_cache_test.py --base http://127.0.0.1:18793 \
  --voice synthetic-a --other-voice synthetic-b --out cache-transitions.json
python3 bench/stream_check.py --base http://127.0.0.1:18793 --model fishaudio/drama-3 \
  --voice synthetic-a --input "The train leaves at seven fifteen." --out cached-stream.json
```

The transition check compares actual PCM across same-voice, builtin-default, other-voice and
changed-text requests; it does not accept a log message as correctness proof. The stream check
also abandons a cached-reference stream and verifies that the next reply is unchanged.

Capture the full corpus twice against the uncached baseline, then twice against the candidate,
using the same reference voice and unchanged server settings:

```sh
python3 bench/corpus_capture.py http://127.0.0.1:18793 base-capture --passes 2 --voice synthetic-a
# Stop and settle the baseline; start the candidate in a fresh coordinated window.
python3 bench/corpus_capture.py http://127.0.0.1:18793 new-capture --passes 2 --voice synthetic-a
python3 bench/compare_gate.py base-capture base-capture new-capture gate.json
```

All requests must succeed and the complete corpus must be present before interpreting the gate.
Do not turn a correctness run, component timing, or cross-host comparison into a speed claim.

### Recorded repair verification

[`results/macos-pr17-repair.json`](../results/macos-pr17-repair.json) records the corrected
patch-0012 engine `c265ca2fc4b9b54ba75b6780e110c566483c93d2`, its binary hashes, CPU checks and
the full-model live comparison against the 11-patch uncached baseline. All 48 paired corpus
captures matched; voice/default/text transitions, builtin and cached streaming abandonment,
reference-boundary variants and two-lane parity passed. Eight warmed voice switches showed
0 KiB process-RSS spread on each build. RSS is not direct Metal allocation telemetry.
All three servers and their owned helpers settled.
This correctness evidence makes no speed, CUDA-runtime or NVFP4-Mac claim.

### Recorded NVFP4 qualification

[`results/macos-nvfp4.json`](../results/macos-nvfp4.json) records the exact NVFP4 revision above on
an Apple M5 Ultra with native TensorFold 1.1.4. The public build entry point produced the server,
codec generator and voice writer from the pinned base and verified patch. A second, test-only
patch corrects an accumulation stress fixture; production kernels and tolerances are unchanged.
A final CLI-only patch groups CFG flags beside voice selection to avoid an upstream integration
collision. Its native build, version and help were exercised separately; the completed GPU checks
were not replayed. The receipt distinguishes both binary hashes and their verification scopes.

The initial Metal contract run passed 13 of 14 Drama variants and all runtime resource limits.
Its shared-gate fixture failed to distinguish FP16 accumulation after sigmoid saturation.
An independent CPU regression now proves that distinction; only the corrected failed variant
was rerun and passed. The receipt retains both results, not a fabricated full-suite pass.

Three sequential single-stream servers completed 28 HTTP checks: 15 complete PCM responses,
one early stream cancellation and 12 expected validation refusals. Paired and separate gamma-1.5
submissions produced identical PCM for builtin and two synthetic reference voices. Repetition,
reference transitions, streamed/whole identity, the post-cancellation response, and opt-in gamma-1/2
repeatability passed. All original servers exited zero after their planned stop, and the supervising
process groups settled. No installed runtime or production service was changed.

These are correctness checks, not a speed or voice-quality comparison. Live qualification used
the native engine directly under the existing cooperative guard; the launcher's selected-weight
path has CPU coverage. M1–M4 execution, smaller-memory Macs, CUDA parity, the full multi-family
Metal suite and offline M1/M2 compiler receipts remain unqualified.
