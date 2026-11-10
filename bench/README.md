# Benchmarks

Speed and audio-identity tools for any server that implements the OpenAI-style `POST /v1/audio/speech` route
and returns mono PCM16. All of them use only the Python 3 standard library (tested with 3.12) plus `git`;
`admit.py` also needs the CUDA driver (`libcuda.so.1`). None of them starts a model: point them at a server
started by `recipe/pilot.py` on Linux or `recipe/pilot_macos.py` on macOS.

| Tool | What it does |
|---|---|
| `drama_bench.py` | The bounded benchmark client: cold, warmup and warm phases, all-attempt percentiles, strict WAV/PCM and chunked-stream validation, process, source and checkpoint identity |
| `tables.py` | Before/after tables from four `drama_bench.py` runs (reference and native, whole and streamed) |
| `admit.py` | Pre-load admission: drops the run's own files from the page cache, then requires CUDA free memory and `MemFree` to cover the expected peak |
| `before-after.sh` | The whole before/after benchmark: reference, then native, then the tables |
| `corpus_capture.py` | Whole-PCM replies for every corpus item, for byte comparison between builds |
| `compare_gate.py` | The audio-identity gate for a speed change: every normal reply byte-identical to the baseline capture |
| `head-trim-identity.sh` | Serves a checkpoint and its `recipe/trim_head.py` copy in turn: every corpus take must be byte-identical |
| `stream_check.py` | Streamed PCM against the whole reply of the same server: byte identity, framing, abandonment, timing |
| `reference_cache_test.py` | PCM identity across reference/default/other-voice/text transitions on one lane |
| `corpus/drama-bench-en-v1.jsonl` | The 24-item corpus |

## Corpus

`corpus/drama-bench-en-v1.jsonl` contains 24 English items written for this benchmark: 8 short
(19–47 characters), 8 medium (114–144) and 8 long (283–313). They contain no private content. Its SHA-256 is
`b8995194915f75ffa8860eaca5fc75821d30d209db907d80a07af946ebf59d3b`.

## Running the before/after benchmark

Build the engine (`scripts/build-engine.sh`), set up the reference runtime directory (`recipe/README.md`), then:

```sh
CHECKPOINT=/path/to/drama-3 RUNTIME_DIR=/path/to/runtime ENGINE_DIR=build/tensorfold \
GPU_LOCK=/path/to/gpu.lock OUT=/path/to/new-run bench/before-after.sh
```

The script stops if either admission check refuses. Results land in `OUT/tables.json`; every request's raw
sample is in `OUT/runs/*/samples.jsonl` and every 200 body is in `OUT/runs/*/bodies/`. This runs the same
comparison as `results/before-after-02/`; it does not recreate those exact conditions, and your numbers will
differ with the host, cache state and admission. The recorded reference run went past a refused admission,
which this script now stops on.

## Checking a speed change for identical audio

```sh
# against the build you started from, then against the changed build
python3 bench/corpus_capture.py http://127.0.0.1:18793 base-capture --passes 2
python3 bench/corpus_capture.py http://127.0.0.1:18793 new-capture --passes 2
python3 bench/compare_gate.py base-capture base-capture new-capture gate.json
python3 bench/stream_check.py --base http://127.0.0.1:18793 --model fishaudio/drama-3 \
  --input "$(head -1 bench/corpus/drama-bench-en-v1.jsonl | python3 -c 'import json,sys;print(json.load(sys.stdin)["text"])')" \
  --out stream.json
```

`compare_gate.py BASE CLIPPED NEW OUT` takes the clipped long-05 reply from `CLIPPED`; it is the same capture as
`BASE` when the base build already returns clipped replies. Native sampling is seeded from the prompt, so the same
build gives the same bytes for the same text.

For reference-prefill changes, pass the same `--voice NAME` to both `corpus_capture.py` calls and
to `stream_check.py`; this includes cached-reference stream abandonment. Also run
`reference_cache_test.py --base URL --voice A --other-voice B --out transitions.json` against
a single-lane server whose default voice remains builtin. Use attested references only.
This check compares actual audio, not cache-log wording. It is a correctness gate, not a speed
measurement; a separate cross-build corpus gate remains necessary.


## drama_bench.py

### Request contract

Each request is `POST {endpoint}/v1/audio/speech` over plain HTTP/1.1, on a new connection with
`Connection: close`. The endpoint host must be an IP literal, so no name resolution happens outside a deadline.
The body is exactly:

```json
{"model": "<--model>", "input": "<corpus text>", "voice": "default", "response_format": "wav|pcm"}
```

`--stream` (with `--response-format pcm`) adds `"stream": true`.

### What a 200 must contain

A 200 counts as a success only if all of these hold:

- the media type is exactly what the native route sends for the requested format: `audio/wav` for WAV or
  `audio/pcm` for PCM. Raw PCM has no header, so its media type is the only evidence of its encoding;
- the body is complete. A whole reply (the default) must use `Content-Length` framing: a missing, malformed or
  repeated `Content-Length`, or any `Transfer-Encoding`, is a `protocol` failure, and bytes past the declared
  length are not body. With `--stream` a 200 must instead be `Transfer-Encoding: chunked` without
  `Content-Length`, decoded strictly (below); a stream that ends without its terminating chunk is
  `incomplete_body`;
- for WAV, the RIFF size matches the body, the `fmt` chunk is PCM mono 16-bit at `--sample-rate` with a
  consistent byte rate and block align, and no chunk overruns the body;
- the PCM is a non-zero whole number of `--frame-samples` frames, and there are at most `--max-frames` frames.

The defaults are Drama 3's contract: 44100 Hz, 2048 samples per frame and 1024 frames. Every complete 200 body
is saved, including invalid ones. Truncated or oversized bodies are not saved; their samples record the byte
counts.

### Phases and statistics

- **cold**: the first request to a process that has served no speech request. Pass `--server-start fresh` to
  send it. Startup time (process launch to `/health` 200) comes from the pilot's `ready.json` and is recorded
  separately. It is not a request.
- **warmup**: `--warmup N` requests that are recorded but not scored.
- **warm**: `--repeats R` full passes over the corpus.

The phases are never pooled. For each phase, `summary.json` reports:

- **attempted, succeeded and failures by class**: `timeout`, `connection`, `incomplete_body`, `oversized_body`,
  `protocol`, `http_<status>` and `invalid_audio`.
- **`end_to_end_s` and `realtime_factor`**, each in two forms:
  - `all_attempts`: nearest-rank p50/p95/p99 with every failure and timeout in the denominator. Failures rank
    slower than any success, so a percentile that lands on one reads `"failure"`.
  - `ok_only`: n, mean, stdev, CV, min, p50/p95/p99 and max over successes.
- **audio**: total and per-request audio seconds, the clipped-sample count, and which items produced differing
  audio across repeats.

`realtime_factor` is `end_to_end_s / audio_s`. Timings are client-side end-to-end: connect, upload, server work
and download up to the last body byte.

Snapshots of `/health`, `/v1/models` and `/metrics` taken before and after the run are stored raw as
service-side views. They are never merged with client timings.

First-playable-audio time is reported only with `--stream`. It is the arrival of the first whole codec frame
(`--frame-samples` samples) of decoded audio bytes, counted from the request start and independent of chunk
boundaries (`first_frame_s`; `first_audio_byte_s` is kept too).

The chunked decoder follows the RFC 9112 section 7.1 grammar, with bounds:

- **chunk-size line:** 1–8 hex digits, then `*(BWS ";" BWS token [BWS "=" BWS (token / quoted-string)])`, then
  CRLF. It holds at most 1 KiB before its CRLF.
- **chunk data:** must be followed by exactly CRLF.
- **trailer section:** after the zero chunk, field lines `token ":" field-value` (no control bytes except HTAB),
  each ending in CRLF, then the empty line. The whole section is at most 64 KiB.

A CR or LF outside a CRLF is a `protocol` failure as soon as it is visible, and so is any byte after the final
empty line. Each bound is judged the same however the input is split across reads.

### Identity (captured before any speech request)

| Identity | What is recorded |
|---|---|
| Process `--server-pid` | `/proc` executable sha256, argv, start time and mapped files (path, size, mtime). The start time is re-read after the run, and a restart marks the run unclean. |
| Source `--server-source` | git HEAD, branch (or detached), sha256 of `git diff --binary HEAD`, and the sha256 of each untracked file, read with `--no-optional-locks`. |
| Checkpoint `--checkpoint` / `--checkpoint-pins` | Every non-hidden file's size and mtime. Files ≤ 64 MiB are rehashed and checked against the pins: a mismatch or a missing pinned file refuses the run. Larger files are listed in `not_rehashed`. |
| Pilot `--pilot-evidence` (optional) | The pilot's `session.json`/`ready.json`, whether its `server_command` equals the process argv, and process-start-to-ready seconds. |
| Configs `--config FILE` (repeatable) | A copy of each file and its sha256. |
| Statements | `--precision` and `--label`. The client concurrency of 1 is recorded as well. |

### Outputs (`--evidence-dir`, which must not exist)

- `manifest.json`: identities, contract, schedule, corpus sha256, harness sha256, argv and bounds.
- `samples.jsonl`: one line per request, flushed as each request completes.
- `bodies/`: every 200 body.
- `server-before.json` and `server-after.json`.
- `summary.json`.
- `configs/` and `pilot/`: copied records.

Exit codes: `0` every scheduled request returned valid audio and the server process was unchanged; `1` the run
completed, but some requests failed or the process changed; `2` refused or unreadable inputs, nothing measured.

### Bounds

Every network exchange runs under one absolute deadline covering connect, upload, the response head and every
body read.

| What | Bound |
|---|---|
| A speech request | `--timeout` (finite, at most 3600 s) |
| Each `/health`, `/v1/models` or `/metrics` snapshot | 10 s and 1 MiB |
| All identity collection | 300 s; git is killed at the deadline |
| Corpus | 1 MiB, 1000 items, 4096 characters per text |
| Schedule | 2000 requests |
| Response head | 64 KiB (a chunked trailer section too; each chunk-size line 1 KiB) |
| Response body | `4096 + 2 × max_frames × frame_samples` bytes, at most 64 MiB |
| Metadata reads, `--config` files, pilot records | 16 MiB each group |
| Checkpoint listing, untracked source files hashed | 10000 each |

Inputs past these bounds are refused with exit 2 before any request.

### Comparing runs

```sh
python3 bench/drama_bench.py compare RUN_A RUN_B --out compare.json
```

Both runs must be complete and have the same corpus sha256, schedule, audio contract and timeout; otherwise the
comparison is refused. Samples are paired by (phase, round, item), and each phase reports the `both_ok`,
`a_only_ok`, `b_only_ok` and `both_failed` counts, each side's all-attempts percentiles, matched-success latency,
RTF and B/A ratio statistics, and how many matched pairs had identical audio bytes.
