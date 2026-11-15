---
license: other
license_name: fish-audio-community-license
license_link: LICENSE
base_model: fishaudio/drama-3
base_model_relation: quantized
pipeline_tag: text-to-speech
language:
  - en
tags:
  - int3
  - gptq
  - quantized
  - pruned
  - moe
  - text-to-speech
  - fish-audio
  - drama-3
---

# Drama 3, 5,900 experts, 3-bit routed experts (14.50 GiB)

> **Built with Fish Audio.** This is a modified version of
> [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3).

An unofficial community quant of Fish Audio's Drama 3 text-to-speech model. Fish Audio did not make, review or
endorse it. The routed experts of the mixture-of-experts backbone are pruned from 10,240 to 5,900 and stored as
3-bit int3 codes; the output head is trimmed to the rows the engine samples and the codec is stored in BF16.
Weights and codec take **14.50 GiB** on disk, against 18.34 GiB for the NVFP4 build this repository held before
and 38.63 GiB for the original. On 24 English texts it kept the reference-voice word error rate within 0.3 points
of the NVFP4 build and failed fewer built-in-voice takes. The [results](#results) give the numbers and their limits.

## The previous build stays in the history

Until 15 November 2026 this repository held the 4-bit NVFP4 build of the same 5,900 experts (18.34 GiB). It was
replaced in place, not deleted: pin revision
[`4bfe2c32daa771566970ad385292fab43d060bb8`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900/tree/4bfe2c32daa771566970ad385292fab43d060bb8)
to get it.

```sh
hf download plotarmordev/drama-3-NVFP4-Pruned-5900 --revision 4bfe2c32daa771566970ad385292fab43d060bb8 --local-dir drama-3-NVFP4-5900
```

drama-tensorfold's `--weights nvfp4-5900` stays pinned to that revision.

## What changed from the NVFP4 build

| Step | Change | Weights + codec |
|---|---|---:|
| NVFP4 build (previous) | 5,900 routed experts in 4-bit NVFP4 | 18.34 GiB |
| Head trimmed | `lm_head` keeps the 4,097 rows the engine samples (end of speech and the semantic audio tokens), listed under `metadata.drama_head_rows` in `model.safetensors.index.json` | 17.39 GiB |
| Codec in BF16 | `codec.pth`'s FP32 tensors rounded to BF16 | 16.66 GiB |
| **3-bit routed experts (this build)** | the same 5,900 routed experts in int3, GPTQ from the FP8 originals | **14.50 GiB** |

- **Experts pruned.** Drama 3's backbone has 40 layers of 256 routed experts. 5,900 of the 10,240 are kept, the
  same set as the NVFP4 build, and only those are stored. The kept set was chosen from router statistics recorded
  while the model generated speech, calibrated partly on private voice-cloning requests. The kept list is in
  `model.safetensors.index.json` under `metadata.drama_experts`.
- **Experts in int3.** Each 16-weight block is three 16-bit planes (sign, high and low magnitude bit) coding the
  E2M1 values 0, 2, 4 and 6, under NVFP4's E4M3 block scales and FP32 global scale: 3.5 bits per weight. They were
  quantized by GPTQ from the FP8 originals with the NVFP4 build's own calibration rows, kappa and damping.
- **Unchanged.** `config.json`, the tokenizer files and `chat_template.jinja` are byte-identical to the original.
  [`NOTICE`](NOTICE) states every change.

The measurements, the tools and the engine patches are in
[plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold); this build's comparison is
[`results/smaller-builds`](https://github.com/plotarmordev/drama-tensorfold/tree/main/results/smaller-builds).

## Running it

This checkpoint does not load in Transformers or SGLang-Omni, and the plain TensorFold engines do not read int3
experts. It needs TensorFold with drama-tensorfold's patch series, which adds int3 routed experts, the trimmed head
and the BF16 codec on Metal and on CUDA. It is drama-tensorfold's default: `scripts/build-engine.sh` with no
`--weights` builds its engine on either platform, and the launcher with no `--weights` downloads this revision, checks
it against `recipe/checksums/trim-bf16-int3.sha256` and serves it with guidance γ 1.5 batched on port 18793. You need
Zig 0.17.0 exactly and the `hf` command from `huggingface_hub`.

```sh
git clone https://github.com/plotarmordev/drama-tensorfold.git
cd drama-tensorfold
mkdir -p runtime && touch runtime/gpu.lock
```

### Apple silicon (Metal)

The engine is ashhart/TensorFold `fb8fdf3d` plus
[`engine/nvfp4/macos/patches`](https://github.com/plotarmordev/drama-tensorfold/tree/main/engine/nvfp4/macos/patches);
patch `0005-int3-routed-experts.patch` unpacks int3 experts in the Metal expert kernels, `0003` loads the trimmed
head and `0004` the BF16 codec.

```sh
ZIG=/path/to/zig-0.17.0/zig scripts/build-engine.sh
python3 recipe/pilot_macos.py --runtime-dir runtime --gpu-lock runtime/gpu.lock --evidence-dir runs/int3
```

The [macOS recipe](https://github.com/plotarmordev/drama-tensorfold/blob/main/recipe/README-macos.md) covers the
speech server and voices.

### NVIDIA (CUDA)

The engine is branch `drama-nvfp4` of [plotarmordev/TensorFold](https://github.com/plotarmordev/TensorFold) at
[`b1f4b9b`](https://github.com/plotarmordev/TensorFold/commit/b1f4b9b4830c9bd43bf7c432cffe7dea21313537) plus
[`engine/nvfp4/patches`](https://github.com/plotarmordev/drama-tensorfold/tree/main/engine/nvfp4/patches);
patch `0006-int3-routed-experts.patch` reads int3 experts in the CUDA NVFP4 expert kernel, `0005` the trimmed head
and `0007` the BF16 codec, widening each value exactly to FP32 on load. On Linux with CUDA 13's `nvcc` and user
systemd:

```sh
NVCC=/usr/local/cuda/bin/nvcc ZIG=/path/to/zig-0.17.0/zig scripts/build-engine.sh
python3 recipe/pilot.py --runtime-dir runtime --gpu-lock runtime/gpu.lock --evidence-dir runs/int3
```

On a host that already has a GPU lock, pass that one instead. I timed this build on one GB10 with the original FP32
codec in place of its BF16 one; read through patch 0007, the BF16 codec equals that codec rounded to BF16, value for
value. See [Hardware](#hardware).

### A request

```sh
curl -sS http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The lighthouse keeper climbed the stairs as the storm rolled in.", "response_format": "wav"}' \
  -o speech.wav
```

Guidance γ 1.5 acts only on requests with a reference voice; a built-in-voice request like this one runs
unguided.

## Hardware

| | This build | NVFP4 build (previous) |
|---|---:|---:|
| Weights and codec on disk | 14.50 GiB | 18.34 GiB |
| Apple M5 Ultra (Metal): real-time factor per take, warm, median (min to max) | 0.511 (0.475 to 0.579) | 0.520 (0.480 to 0.603) |
| Apple M5 Ultra: whole reply, warm, median | 4.91 s | 4.90 s |
| NVIDIA GB10 / DGX Spark (CUDA): real-time factor, warm p50, built-in / reference voice | 0.614 / 0.809 | 0.619 / 0.831 |
| NVIDIA GB10: whole reply, warm p50, built-in / reference voice | 5.16 s / 8.49 s | 5.05 s / 8.52 s |
| NVIDIA GB10: peak device memory, engine load | 14.77 GiB, 7.5 s | 16.93 GiB, 8.0 s |

The Mac rows are one machine and one engine build (`fb8fdf3d` + 7 patches), both builds served by
`recipe/pilot_macos.py`, the 24 corpus texts in the built-in voice, one request at a time, the second (warm) pass;
real-time factor is the client's time over the audio's length (lower is faster). Each build repeated all 24 takes
byte for byte on the second pass. The machine is shared, so a run counts only when every warm take stays under 0.65;
three other runs on 15 Nov were slowed by other work and are recorded but not shown. The two builds are equally
fast on the Mac: the gap is within the spread between runs. The GB10 rows are one GB10 and one CUDA engine binary
(`b1f4b9b` + `engine/nvfp4/patches`), this build with the original FP32 codec, the same 24 texts one
request at a time, the warm pass, p50 nearest rank, peak device memory as the engine's `/health` reports it
([`spark-speed.json`](https://github.com/plotarmordev/drama-tensorfold/blob/main/results/smaller-builds/spark-speed.json)).
On the Spark the two builds are equally fast too; this one needs 2.16 GiB less device memory.

![Disk size at each step](charts/footprint.png)

![Speed on the M5 Ultra](charts/speed.png)

## Results

Measured on the M5 Ultra with the Metal engine
([`results/smaller-builds`](https://github.com/plotarmordev/drama-tensorfold/tree/main/results/smaller-builds)): the
24 English texts of the benchmark corpus, salts 0 and 1, 48 takes per build and voice, up to 1,024 audio frames per
reply. "Reference voice" is a model-invented voice given as a reference clip with guidance γ 1.5 batched;
"built-in voice" is the engine's default voice with no reference.

| Voice | Measure | This build | NVFP4 build (previous) |
|---|---|---:|---:|
| Reference | Word error rate, mean / pooled | 1.78 % / 0.97 % | 1.50 % / 0.75 % |
| Reference | Failed takes | 0/48 | 0/48 |
| Built-in | Word error rate, mean / pooled | 5.78 % / 3.73 % | 6.91 % / 6.27 % |
| Built-in | Failed takes | 4/48 | 6/48 |

A take fails if it reaches the 1,024-frame cap, its word error rate is above 0.2, or it has a quiet run of 3 s or
more inside the speech. Word error rate: Whisper large-v3 with Whisper's English normalizer.

![Word error rate and failed takes](charts/quality.png)

### What these results do and do not show

- **3.84 GiB smaller, the same quality within this test's reach.** With a reference voice the word error rate rose
  from 1.50 % to 1.78 %, with no failed take in either build. In the built-in voice this build failed 4 takes
  against 6 and scored a lower word error rate.
- **One new failure.** Two of the four built-in failures are the NVFP4 build's own (`short-01` salt 0, `long-01`
  salt 1); the other two are one text, `medium-05`, which on both salts runs to the frame cap after its first
  sentence, where the NVFP4 build says it all in about 8 s. Use a reference voice.
- **Narrow test.** English only, 24 texts, 2 salts, one request at a time, greedy sampling. No speaker-similarity
  or naturalness score was taken for this build, and nothing here detects small differences.

## Limitations

- Measured only in English. Drama 3 lists many more languages; this build inherits them untested, and pruning
  experts by English routing statistics may hurt other languages more than English.
- Multi-speaker scenes, long texts beyond 1,024 audio frames per reply, Drama 3's natural-language and tag controls
  and concurrent requests were not measured.
- 4,340 of the 10,240 routed experts are gone, and the rest are at 3.5 bits. The effect on unusual voices, styles or
  languages is unknown.
- Only the patched TensorFold engines above load it. The GB10 rows were timed with the FP32 codec, not the BF16 one.

## License and use restrictions

This model is a Derivative Work of Fish Audio Drama 3 and is licensed under the **Fish Audio Community License,
Version 1.0**: see [`LICENSE`](LICENSE) (byte-identical to Drama 3's) and [`NOTICE`](NOTICE). In short, and the
license text governs:

- **Research and non-commercial use** is free.
- **Commercial use** is allowed only while you and your affiliates have less than USD 500,000 in annual revenue,
  under the license's terms; above that, or for any other commercial use, you need a separate license from Fish
  Audio.
- **Hosted services** that let third parties generate speech or clone voices (an API, a hosted text-to-speech or
  voice-cloning product) need a separate license from Fish Audio, whatever your revenue.
- **Redistribution** must include `LICENSE`, the `NOTICE` file with Fish Audio's attribution, and "Built with Fish
  Audio".

You may not use this model or its output to:

- imitate the voice of an identifiable person without their consent (or a parent's or guardian's for a minor, or
  the rights holder's for a deceased person) or another legal right to do so;
- impersonate a person or organization to deceive, defraud or harass, or place calls that break telemarketing or
  robocall laws;
- generate sexual content involving a minor or a voice presented as a minor's, or using a real person's voice
  without their consent;
- remove, disable, circumvent or obscure a watermark or provenance signal applied by the Fish Audio Materials;
- create, train, distill or improve another foundation model or speech, audio or voice generation model (training
  solely to detect synthetic audio, verify watermarks or prevent spoofing is allowed).

If you offer voice cloning or speech generation with it to others, the license requires you to bind them to
restrictions at least as protective. The model is provided as is, without warranty of any kind; see the license's
disclaimer and limitation of liability.

## Watermark and provenance

I found no audio watermark or provenance signal in Drama 3's open-weights inference
([SGLang-Omni](https://github.com/sgl-project/sglang-omni) at `ab095ef9`, the runtime Drama 3's card points to), and
the engine used here adds none and removes none: it decodes the codes and writes plain 16-bit PCM or a minimal WAV.
Fish Audio's hosted service may differ, and a signal learned into the codec's weights would not show up in code; I
ran no watermark detectors. This build's codec is the original rounded to BF16, so it carries whatever such a
signal survives that rounding.

The license forbids removing, disabling or obscuring any watermark or provenance signal the Fish Audio Materials
apply; nothing here was changed to do so.
