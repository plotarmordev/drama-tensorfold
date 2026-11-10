# drama-tensorfold

[![code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-0969da?style=flat-square)](LICENSE)
[![model: FACL v1.0](https://img.shields.io/badge/model-FACL_v1.0-8250df?style=flat-square)](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE)
[![GPU: GB10 sm_121](https://img.shields.io/badge/GPU-GB10_sm__121-76b900?style=flat-square)](#quickstart)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76b900?style=flat-square)](#quickstart)
[![status: not yet compared](https://img.shields.io/badge/status-not_yet_compared-a16207?style=flat-square)](#the-patches)
[![weights: NVFP4 recommended](https://img.shields.io/badge/weights-NVFP4_recommended-76b900?style=flat-square)](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned)

> [!TIP]
> **We recommend the [20.8 GB NVFP4 pruned weights](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned).** About half
> the memory of the full model, with the same measured quality: 19.35 GiB peak GPU memory against 36.95 GiB, 20.76 GiB
> on disk against 38.63 GiB, and a word error rate of 1.31 % against 1.42 % (paired difference -0.11 pp
> [-0.35, +0.08], 0 failed takes of 96 on either). First audio arrives sooner, 1.955 s against 2.428 s at p50. The
> trade-offs: a cold start takes 26.3 s against 5.9 s, and only 7,369 of the 10,240 routed experts are kept. Measured
> on one GB10 over 24 English texts with a reference voice: [full model against quant](results/full-vs-quant-01/README.md).
>
> The [Quickstart](#quickstart) serves them with `--weights nvfp4`: it builds their engine, downloads the weights
> on first run and applies the guidance they were measured with.
>
> **Smaller: the [5,900-expert quant](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900)**, `--weights
> nvfp4-5900`, downloaded and checked the same way. 18.34 GiB on disk against 20.76 GiB, 16.93 GiB peak GPU memory
> against 19.35 GiB, and a cold start of 20.1 s against 26.1 s. With a reference voice it matches the 20.8 GB quant:
> word error rate 1.50 % against 1.31 % (paired +0.19 pp [-0.02, +0.49]), 0 failed takes of 96 on either. With
> the built-in voice it does worse: 5.85 % against 2.92 % (paired +2.93 pp [+0.43, +5.81]), 8 failed takes of 96
> against 5. The 20.8 GB quant stays the GB10 default; on a Mac short of memory, serve the 5,900 quant
> ([macOS](recipe/README-macos.md)). Measured on one GB10: [results/quant-15gb](results/quant-15gb/README.md).
>
> **Built with Fish Audio.** The quant is a modified version of
> [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) under the Fish Audio Community License v1.0
> ([License](#license)). `--weights full` serves the full model, the alternative.

**Fish Audio's Drama 3 text-to-speech on NVIDIA GB10 (native Zig/CUDA) or Apple silicon (native Zig/Metal).**

[TensorFold](https://github.com/ashhart/TensorFold) runs Drama 3 natively since v1.1.0. This repository pins
TensorFold v1.1.9 for the full model, which now carries every feature the old patch series added. NVFP4 retains its
measured CUDA engine pin with a checksummed patch series and uses a separate upstream pin plus Metal patches on Mac. It builds the engine
and starts a local speech server with an OpenAI-style `/v1/audio/speech` route, whole or streamed.

**Built with Fish Audio.** The model weights are not in this repository; the launcher downloads them from
[Hugging Face](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) under Fish Audio's own
[license](#license).

> [!NOTE]
> The full-model engine this recipe builds, upstream TensorFold plus the opt-in patches below, has no published
> end-to-end comparison against the previous patched engine yet, and the
> [qualification gates](qualification/README.md) have not been rerun on it: they measure the previous engine.

## Quickstart

Apple silicon: use the [macOS/Metal instructions](recipe/README-macos.md) for `--weights nvfp4`
or `--weights full`. The GB10 measurements above do not qualify Mac speed or quality.
The Linux requirements and cgroup cap below do not apply to the Mac launcher.

You need Linux with user systemd, CUDA 13, [Zig 0.17.0](https://ziglang.org/download/), the `hf` command
(`python3 -m pip install -U huggingface_hub`) and 88 GiB of free memory at load (the server is capped at 54 GiB).
The NVFP4 weights take about 21 GiB of disk, the full model about 39 GiB; the engine checkout, build and runtime
caches need more.

The launcher takes your host's shared GPU lock, the one every GPU job on the machine already uses, so two models
never load at once. Set `GPU_LOCK` to that file. Only if the host has none, create one with `touch` and make
your other GPU jobs use it too.

```sh
ZIG=/path/to/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh --weights nvfp4
GPU_LOCK=/path/to/your/shared/gpu.lock && mkdir -p runtime
python3 recipe/pilot.py --weights nvfp4 --runtime-dir runtime --gpu-lock "$GPU_LOCK" --evidence-dir runs/pilot-01
```

For the full model, write `--weights full` in both commands. For the 5,900-expert quant, keep the engine build and
start with `--weights nvfp4-5900`: it shares the NVFP4 engine and guidance.

The first start downloads the weights at the revision pinned in [`recipe/sources.json`](recipe/sources.json),
into `drama-3-NVFP4-Pruned/`, `drama-3-NVFP4-Pruned-5900/` or `drama-3/`. Every start checks them against [`recipe/checksums`](recipe/checksums)
and refuses missing or changed files; for the 21 GiB NVFP4 weights that takes about 11 seconds on a GB10.
On Linux, `--weights nvfp4` builds [`engine/nvfp4/UPSTREAM`](engine/nvfp4/UPSTREAM) plus
[`engine/nvfp4/patches`](engine/nvfp4/patches) and serves with
guidance γ 1.5 and batched guidance, as measured; a `--cfg-gamma` of your own after `--` replaces both.
On Mac it builds [`engine/nvfp4/macos/UPSTREAM`](engine/nvfp4/macos/UPSTREAM) plus its patch series.
Metal's `--cfg-batched` pairs independent states in one command buffer, not a vectorized GEMM.
`--weights full` builds [`engine/UPSTREAM`](engine/UPSTREAM) with the one-patch [`engine/patches`](engine/patches) series.

When it prints `DRAMA_PILOT_READY`, synthesize from another terminal. The second command streams and plays as
it arrives:

```sh
curl -sS http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The train leaves at seven fifteen."}' -o hello.wav
curl -sSN http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The train leaves at seven fifteen.", "response_format": "pcm", "stream": true}' \
  | aplay -q -f S16_LE -r 44100 -c 1
```

### A reference voice

The NVFP4 results above were measured with a reference voice; requests in the built-in voice, like the two above,
are drawn without guidance, though their replies still say the server's γ in `X-Cfg-Gamma`. To use a voice, encode
a clip of one you may use (your own, one recorded with consent, or one the model made) and serve it as the default,
so a request that names no voice gets it:

```sh
mkdir -p voices
python3 build/tensorfold-nvfp4/tools/zig/drama_voice.py --native build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native \
  --checkpoint drama-3-NVFP4-Pruned --audio clip.wav --text clip.txt --out voices/NAME.json \
  --source-kind owner --source-note "my own voice, recorded 2026-10-23"
python3 recipe/pilot.py --weights nvfp4 --runtime-dir runtime --gpu-lock "$GPU_LOCK" --evidence-dir runs/pilot-02 \
  -- --voices "$PWD/voices" --default-voice NAME
```

`clip.txt` holds the clip's exact words. Encode while no model holds the GPU. The voice file format and the
encoder's options are in `build/tensorfold-nvfp4/docs/drama-voices.md`.

### Try it with your coding agent

```text
Help me run Fish Audio Drama 3 locally with https://github.com/plotarmordev/drama-tensorfold on this GB10.
Read its README, recipe/README.md and the Fish Audio Community License first, and follow them.

1. Check the requirements: CUDA 13 with nvcc, Zig 0.17.0, the hf command, user systemd, disk space and free
   memory.
2. Build the engine with scripts/build-engine.sh --weights nvfp4.
3. Start it with recipe/pilot.py --weights nvfp4 on loopback; the first start downloads and verifies the weights.
4. Synthesize one sentence as WAV, then stream it, and tell me the X-Finish-Reason header.

Ask me which GPU lock file this host already uses before starting, and before stopping anything already
running. Keep the server on 127.0.0.1. Letting other people send text to it can be a Hosted Service under
the license's Section III(c), which needs a separate license from Fish Audio.
```

<details>
<summary>HTTP contract</summary>

`POST /v1/audio/speech` takes `model` (`fishaudio/drama-3`), `input` (at most 4096 characters), `voice`
(`default`, or a name loaded with `--voices`), `response_format` (`wav` or `pcm`) and `stream` (`true` needs
`pcm`). Optional `temperature`, `top_p`, `top_k` and exact integer `seed` select
request-local sampling; see [ranges, defaults and redraw behavior](recipe/README-sampling.md).
Other synthesis modifiers are refused. `GET /v1/voices` lists the voices.
The Linux CUDA NVFP4 engine accepts its encoder's `drama3-voice` catalogs and request `references`
as codec codes (not audio). The Mac NVFP4 engine loads the same `drama3-voice/1` and `/2` files, and its
own attested `drama3-http-voice/1` catalogs, from `--voices`, and refuses request-supplied references; see the
[Mac speech server](recipe/README-macos.md#speech-server) and [voice commands](recipe/README-macos.md#attested-reference-voices).
Guidance-enabled replies carry the configured γ in `X-Cfg-Gamma`; builtin voices remain unguided.

Request text and reference transcripts accept canonical speaker tags 0 through 15.
See [speaker-tagged dialogue](recipe/README-dialogue.md) for catalog examples,
profile-specific reference limits and legacy long-form speaker continuation, and its
[known full-model stopping limitation](recipe/README-dialogue.md#known-full-model-stopping-limitation),
unchanged on v1.1.9 ([issue #32](https://github.com/plotarmordev/drama-tensorfold/issues/32)).

- **Whole reply:** `Content-Length` framing and an `X-Finish-Reason` header: `stop`, `length` or, with the silence
  guard on, `silence`.
- **Streamed reply:** chunked mono 16-bit 44.1 kHz PCM. A reply that ends in `length` or `silence` ends with an
  `X-Finish-Reason` trailer.
- **`length`:** the reply hit the frame cap (1024 frames, 47.55 s, unless `--max-frames` sets another) without an
  end of speech; the audio is clipped.
- **Errors:** `400` for a refused request, `503` while the engine is busy, `500` for a server fault.
</details>

## The patches

### CUDA NVFP4 long speech

The CUDA NVFP4 patch series preserves delivery markup when `--long-form` splits a request.
Paired tags such as `<whisper>…</whisper>`, bracketed cues and special tokens stay whole;
Chinese/Japanese sentence and clause punctuation are recognized. A topic-shift mark inside a
protected span does not create a cut. Plain English without these spans retains its existing cuts.

`--long-form-chars` remains a code-point target, with one exception: an indivisible protected span
may exceed it. The engine's prompt, segment-count and generated-frame limits still apply.
The full-model and Metal NVFP4 engines do not expose this legacy long-form route.

The behavioral reference is [Sogni-AI/drama3-mlx's chunking](https://github.com/Sogni-AI/drama3-mlx/blob/ecc235039ccd5f982f66a6982184f57432354ab4/drama3_mlx/chunking.py).
The native patch extends TensorFold's own implementation; it does not copy that Fish-licensed source.
See the [CUDA patch index](engine/nvfp4/patches/README.md) for scope and tests.

### Full-model engine

The full-model engine (`--weights full`) is TensorFold v1.1.9 plus one Metal-only patch, `drama3-generate
--calibration-rows` ([below](#calibration-rows-for-the-exl3-quant)): every feature the old patch series
added is upstream. Each flagged feature stays off until you pass its flag, so a server started without them answers as plain
TensorFold does. The Linux recipe sets `TF_DRAMA_PREFILL_ROWS=128` for native full-model and custom-checkpoint
launches to keep batched prefill; TensorFold otherwise defaults to single-row prefill. NVFP4 and the sglang
reference backend do not receive this setting. Per-request `temperature`, `top_p`, `top_k` and `seed`, and speaker
tags 0 to 15 in request text, need no launch flag; the CUDA NVFP4 series (0002, 0003) and the Metal NVFP4 series
(0004-0006) carry the same two request-level features. Metal patch 0007 loads the CUDA engine's `drama3-voice`
files in `--voices`; 0008 answers a take that ends speech before its first audio frame with a 500, not empty audio.
CUDA patch 0005 and Metal patch 0009 load a checkpoint whose lm_head keeps only the 4,097 rows the engine scores
(EOS and the semantic band), as [`recipe/trim_head.py`](recipe/trim_head.py) writes it: 0.95 GiB less on disk.
Each feature's notes are in the engine tree's `CHANGELOG.md` or `docs/`.

| In TensorFold 1.1.9 | Opt in with | Upstream |
|---|---|---|
| CUDA prompt prefill in bounded row chunks | `TF_DRAMA_PREFILL_ROWS` (the recipe sets `128`) | [#1120](https://github.com/ashhart/TensorFold/pull/1120) |
| Replay waits for poison fills before buffers are reused | always on | [#1119](https://github.com/ashhart/TensorFold/pull/1119) |
| A configurable frame budget | `--max-frames N` | [#1134](https://github.com/ashhart/TensorFold/pull/1134) |
| An allowlist for bracketed speech controls | `--allowed-controls LIST` | [#1152](https://github.com/ashhart/TensorFold/pull/1152) |
| Attested voice catalogs and `GET /v1/voices` | `--voices DIR`, `--default-voice NAME` | [#1135](https://github.com/ashhart/TensorFold/pull/1135) |
| A sustained-silence guard that ends a reply as `silence` | `--silence-floor-dbfs`, `--silence-seconds` | [#1157](https://github.com/ashhart/TensorFold/pull/1157) |
| The guard's onset rule: silence before enough voiced audio | `--silence-onset-s`, `--silence-onset-voiced-s` | [#1158](https://github.com/ashhart/TensorFold/pull/1158) |
| The guard's mid rule: an ending lost in a long silence | `--silence-mid-s`, `--silence-mid-min-voiced-s` | [#1159](https://github.com/ashhart/TensorFold/pull/1159) |
| Whole replies the guard ended are drawn again | `--silence-redraws N`, `--silence-redraw-budget-s`, `--silence-redraw-controls` | [#1160](https://github.com/ashhart/TensorFold/pull/1160) |
| Whole replies that stopped short of their text are drawn again | `--short-reply-ratio`, `--short-reply-chars-per-s` | [#1161](https://github.com/ashhart/TensorFold/pull/1161) |
| A whole reply is drawn once more when a phrase opens far above the others | `--register-redraw-st`, `--register-redraw-high`, `--register-high-hz` | [#1162](https://github.com/ashhart/TensorFold/pull/1162) |
| One bounded reference-prefix snapshot per Metal lane, with exact identity and safe invalidation | none (automatic) | [#1237](https://github.com/ashhart/TensorFold/pull/1237) |
| Request-local speech sampling and exact integer seeds, with parsed-JSON boundary docs | HTTP `temperature`, `top_p`, `top_k`, `seed` | [#1277](https://github.com/ashhart/TensorFold/pull/1277) |
| Canonical speaker tags 0..15 in speech and catalog transcripts | speaker tags in text | [#1278](https://github.com/ashhart/TensorFold/pull/1278) |

Every old patch's behaviour is in v1.1.9, and a CUDA GPU A/B against the previous pin passed (corpus captures byte-identical; request sampling and speaker tags pass), with two differences. With the mid-silence rule on, `X-Silence-Mid-Trips`
reports only whether the returned take tripped (0 or 1); takes discarded by silence/short/register redraws are no
longer counted. A temperature of `-0` or `-1e-400` reaches the sampler as `+0` rather than `-0`; both were already
greedy. `drama3-voice`, the upstream tool that writes attested voice files for `--voices`, is in TensorFold; the
full-model build script still builds only the native server and `drama3-generate`.

Pass serve flags to the launcher after `--`. An illustrative launch with the frame budget, the silence guard and
one redraw; the numbers are examples, not tuned values, so calibrate thresholds on your own voice:

```sh
python3 recipe/pilot.py … -- --max-frames 2048 --silence-floor-dbfs -50 --silence-seconds 6 --silence-redraws 1
```

The speech-control allowlist holds only words from Fish Audio's public tags; deployments may extend it locally.

A voice catalog is a directory of `drama3-http-voice/1` files, each with a `source` attestation (`synthetic`,
`self` or `consented`); the format is in `docs/drama-http-voices.md` in the engine tree. Clone only a voice the
model invented, your own, or one whose owner has consented on record, as the model license requires
([License](#license), IV(b)). `clips/` and `voices/` are git-ignored here. `recipe/pilot.py` runs the server in
`--runtime-dir`, so pass `--voices` an absolute path.

### Calibration rows for the EXL3 quant

```sh
scripts/build-engine.sh --weights full   # once
python3 recipe/calibrate_experts.py      # about 30 min
```

captures, on a Mac and from the FP8 originals in `/Users/Shared/agent-work/drama-assets/checkpoint-bf16`, the
input every routed expert sees, in the layout of the GPTQ `--rows` dumps: `layer-NN.x.bf16` (the MoE input),
`layer-NN.picks.i32` (the eight original expert IDs), `layer-NN.w.f32` (their routing weights) and `class.u8`
(prompt text, prompt audio or generated) for all 40 layers, in
`/Users/Shared/agent-work/drama-assets/exl3-calibration-rows-v2`. Every bench corpus line is generated alone and
after each of three 29 s clips of the Capri reference, at two seeds, then prefilled again with its generated frames.
No CUDA, torch or Spark is used; the validation step needs numpy. `report.json` gives the class mix, the experts per
layer without rows (all 10,240 and the 5,900 the pruned quant keeps), the size on disk, the check against the NVFP4
build's statistics and how to build each expert's Hessian: H_e = X_eᵀX_e / n_e over the rows whose picks hold e,
and for `down_proj` the same over SwiGLU(gate, up) of those rows. The check recomputes the NVFP4 build's published
`err_h` for every kept gate and up expert of layers 0, 3, 20 and 39 with these rows' per-column sums of squares:
identical statistics give a ratio of exactly 1.

The capture on 11 Nov: 175,396 rows (34,207 prompt text, 90,000 prompt audio, 51,189 generated; the NVFP4 build
had 15 / 43 / 42 %), 29.2 GB on disk, 28 minutes. 9,918 of the 10,240 experts have rows, 5,888 of the 5,900 kept
ones (at least 14 rows each, 1 % under 1,005). The 12 without rows are in layer 36, which the quant keeps whole;
the NVFP4 build's own calibration also gave them none, so the router does not pick them. The `err_h` ratio over
1,208 tensors: median 1.006, 5th–95th percentile 0.991–1.027.

### EXL3 routed experts

```sh
python3 recipe/exl3_experts.py --encoder TENSORFOLD/zig-out/bin/tf-exl3-encode   # about 85 h, resumable
```

quantizes the 5,900 experts the pruned quant keeps to EXL3 at 3.0 and 2.5 bits per weight, from the FP8 originals
(not from NVFP4), with TensorFold's CPU encoder `tf-exl3-encode` (TensorFold card 23; no CUDA or torch). The
encoder checks that its written trellis reloads and decodes bit for bit. Each expert's Hessians come from the
calibration rows above: H = XᵀX / n over the rows routed to it, and for `down_proj` the same over SwiGLU(gate, up)
of the original weights. The 12 layer-36 experts without rows use all of their layer's rows. Finished experts stay in
`drama3-exl3-experts/work`, so a stopped run resumes where it left off. Each finished layer becomes one shard in
`drama3-exl3-experts/exl3-3.0/` and `exl3-2.5/`, holding only the expert groups (`trellis`, `suh`, `svh`, `mul1`).
Each folder also gets an index and `exl3-provenance.json`: input paths and hashes, encoder commit and hash, settings,
bytes against the NVFP4 experts, and per layer and projection `trace((W − Q) H (W − Q)ᵀ)` for EXL3 and for the
published NVFP4 expert on the same H.

Eight experts on 11 Nov (the median-row and most-routed kept expert of layers 0, 13, 26 and 39), EXL3 error over the
published NVFP4's:

| bits | gate | up | down | per expert on disk | seconds per expert |
|---|---|---|---|---|---|
| 3.0 | 7–133 | 6–112 | 3.3–40 | 1.197 MB | 28–30 |
| 2.5 | 12–223 | 11–218 | 5.8–69 | 1.000 MB | 24 |

NVFP4's experts take 1.769 MB each. The encoder matches its own round-to-nearest NVFP4 at 3.0 bits (1.04× on layer 0
expert 117 `gate_proj`). The published NVFP4 is GPTQ'd, though, and its error lies along directions the calibration
H barely weights. `tf-exl3-encode` weights only H's diagonal, with no LDLQ feedback, so it cannot do the same; that
gap, not the bitrate, sets these ratios.

## Legacy: the patched engine

Before upstream shipped Drama 3, this recipe built its own port: 59 patches on TensorFold `c7137bb3`, last at
commit [`0f54000`](https://github.com/plotarmordev/drama-tensorfold/tree/0f54000374831db1023791a5c4b9f0b98e9cff18).
That engine's speed comparison against the sglang-omni reference, its accuracy gates, `--long-form` and its voice
cloning tools (`drama_voice_clip.py`, `drama_voice.py`, `tensorfold-native voice encode`) are not in the new base.
[`qualification/`](qualification/README.md) and [`results/`](results/README.md) measure that engine; check out
`0f54000` to rebuild it.

## Tests

```sh
scripts/test.sh
```

Runs every test that needs no model, GPU or network, on Linux or macOS, and exits non-zero on any failure. On
macOS `gpu-checks/selftest.sh` skips only its pause/resume part, which needs Linux user systemd, with one line;
the Linux launcher tests use mocks and run on both.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Every speed or quality claim carries
[before/after evidence measured against the recipe baseline](CONTRIBUTING.md#performance-and-quality-claims).

## License

**Built with Fish Audio.** All code here is [Apache-2.0](LICENSE), including the engine patches.
The Drama 3 model weights and codec checkpoint are under the
[Fish Audio Community License v1.0](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE)
and are downloaded separately; [NOTICE](NOTICE) has the attributions.

<details>
<summary>What FACL v1.0 allows for the weights (a summary, not legal advice: read the license)</summary>

- **Research and non-commercial use is free** (II), including personal use and evaluation.
- **Commercial use only while your Annual Revenue is under USD 500,000** (III(a)). Annual Revenue is the gross
  revenue of you and your affiliates from all sources over the trailing twelve months. If you deploy, install
  or operate the model for a particular client, or deliver a copy to them, that client's Annual Revenue must
  also be under the threshold. That client condition does not apply to delivering only output you generated
  yourself, or to providing your own generally available product to users who don't receive a copy.
- **Reaching the threshold** (III(b)): notify business@fish.audio within 30 days. The commercial uses you were
  already making on that date may continue for 90 days. After that, commercial use needs a separate license,
  and III(a) applies again if your revenue later falls back below the threshold.
- **No Hosted Service without a separate license**, whatever your purpose or revenue (III(c)). A Hosted Service
  lets third parties generate speech from text or audio they supply, for example through an API. It
  excludes:
  - speech for content you create, select or control, or that your own product generates, delivered in or with
    your products or services;
  - production where you operate the model and direct each generation, delivering only finished output without
    client self-serve or automated access;
  - a free, non-commercial research or demonstration service that is not monetized and is not offered or
    promoted as a production service or API.
- **Show "Built with Fish Audio"** where you distribute or offer anything that uses the materials. Ship the
  license and its notice with any copy (IV(a)).
- **Use restrictions** (IV(b)), which also cover outputs. Among them:
  - no speech imitating an identifiable person's voice without their prior consent (a parent's or guardian's for
    a minor, the rights holder's for a deceased person) or another legal right to do so;
  - no impersonation to deceive, defraud or harass, and no calls that break telemarketing or robocall laws;
  - no generating sexual content involving a minor or a voice presented as a minor's, or sexual content using a
    real person's voice without that person's consent;
  - no intentionally removing, disabling, circumventing or obscuring watermark or provenance signals applied by
    the materials, or distributing tools, derivative works or services designed to do so; ordinary editing,
    mixing, compression or format conversion without that intent is not a breach;
  - if you offer speech generation to third parties, bind them to enforceable terms at least as protective;
  - no using the materials or outputs to create, train, distill or improve another foundational generative AI
    model or speech, audio or voice generation model. Fish Audio's own Models (Drama 3 and its codec) and their
    Derivative Works are excluded, and training solely to detect synthetic audio, verify watermarks or prevent
    spoofing is allowed.

  A breach of IV(b) ends the license automatically (IV(f)).
</details>

[Recipe](recipe/README.md) · [Benchmarks](bench/README.md) · [Qualification](qualification/README.md) · [Results](results/README.md) · [Contributing](CONTRIBUTING.md) · [NOTICE](NOTICE) · [Apache 2.0 license](LICENSE) · [Model license (FACL v1.0)](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE)
