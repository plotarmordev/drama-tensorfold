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
  - nvfp4
  - quantized
  - pruned
  - moe
  - gptq
  - text-to-speech
  - fish-audio
  - drama-3
---

# Drama 3, NVFP4, pruned to 7,369 experts

> **Built with Fish Audio.** This is a modified version of
> [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3).

An unofficial community quant of Fish Audio's Drama 3 text-to-speech model. Fish Audio did not make, review or
endorse it. The routed experts of the mixture-of-experts backbone are pruned from 10,240 to 7,369 and stored as
4-bit NVFP4; everything else, the audio codec included, is byte-identical to Drama 3 at revision
[`e0a9218`](https://huggingface.co/fishaudio/drama-3/tree/e0a9218aa48888270f06ef1ed9ea67622c0e17d2). Weights and
codec take 20.76 GiB on disk against 38.63 GiB for the original. On one NVIDIA GB10, over 24 English texts, its word
error rate, failed takes and speaker similarity were within noise of the original; it reached first audio sooner,
and it took more than four times as long to start from cold. The [results](#full-model-against-this-quant) give the numbers and
their limits.

## How it was made

- **Experts pruned.** Drama 3's backbone has 40 layers of 256 routed experts. 7,369 of the 10,240 are kept (145 to
  256 per layer), and only those are stored. Each layer keeps the experts that cover 90 % of the routing weight its
  router gave while the model generated speech for English calibration requests and private voice-cloning
  requests. The kept list is in `model.safetensors.index.json` under `metadata.drama_experts`.
- **Experts quantized.** The kept routed experts, FP8 in the original, are 4-bit NVFP4, quantized with GPTQ
  calibrated on 120 English requests that use model-invented voices.
- **Unchanged.** The other weights, `config.json`, the tokenizer files, `chat_template.jinja` and the codec
  `codec.pth` are byte-identical to the original. [`NOTICE`](NOTICE) states every change.

The measurements, the benchmark tools and the serving recipe are in
[plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold); this comparison's tables, raw
records and the exact commands are in
[`results/full-vs-quant-01`](https://github.com/plotarmordev/drama-tensorfold/tree/main/results/full-vs-quant-01).

## Running it

This checkpoint does not load in Transformers or SGLang-Omni. It needs an engine that reads NVFP4 routed experts,
a compact (pruned) expert set and Drama 3's codec. It was measured on [TensorFold](https://github.com/ashhart/TensorFold)'s
native engine, build `v1.0.11-193-g70dec469`. The public engine is branch `drama-nvfp4` of
[plotarmordev/TensorFold](https://github.com/plotarmordev/TensorFold), a fork of ashhart/TensorFold, at commit
[`b1f4b9b`](https://github.com/plotarmordev/TensorFold/commit/b1f4b9b4830c9bd43bf7c432cffe7dea21313537); its run
guide is
[`docs/drama-3-nvfp4.md`](https://github.com/plotarmordev/TensorFold/blob/b1f4b9b4830c9bd43bf7c432cffe7dea21313537/docs/drama-3-nvfp4.md).
That commit is the measured build (`v1.0.11-193-g70dec469`) squashed into one commit, with neutral wording in 11
lines of comments, docs and test messages, plus the run guide; no code that runs changed. The measured build's
own history is not published.

You need Linux with an NVIDIA GPU (the model has been built and run only on an NVIDIA GB10), Zig 0.17.0 exactly,
the CUDA toolkit's `nvcc`, and the `hf` command from `huggingface_hub`. The run guide gives the details.

Build:

```sh
git clone --branch drama-nvfp4 https://github.com/plotarmordev/TensorFold.git tensorfold
cd tensorfold
git checkout --detach b1f4b9b4830c9bd43bf7c432cffe7dea21313537
zig build native fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dstrip=true -j4
```

Serve:

```sh
hf download plotarmordev/drama-3-NVFP4-Pruned --local-dir /path/to/drama-3-NVFP4-Pruned
zig-out/native/bin/tensorfold-native serve /path/to/drama-3-NVFP4-Pruned --name drama-3 --port 8080 --cfg-gamma 1.5 --cfg-batched
```

Request:

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "drama-3", "input": "The lighthouse keeper climbed the stairs as the storm rolled in.", "response_format": "wav"}' \
  -o speech.wav
```

The quant servers were launched with `--cfg-gamma 1.5 --cfg-batched`. Guidance was active for reference-voice
measurements and inactive for built-in-voice measurements, where the engine logged γ 1.00. The quant was not
measured with a reference voice and guidance disabled.

Guidance γ 1.5 acts only on requests with a reference voice. A request in the built-in voice, like the one above,
runs unguided: the engine logs `cfg_gamma=1.00` although the reply's `X-Cfg-Gamma` header says 1.5. The run guide's
[Guidance needs a reference voice](https://github.com/plotarmordev/TensorFold/blob/b1f4b9b4830c9bd43bf7c432cffe7dea21313537/docs/drama-3-nvfp4.md#guidance-needs-a-reference-voice)
gives the two ways to send one: a voice file served with `--voices`, or codes inline in the request.

## Hardware

Measured on one NVIDIA GB10 (CUDA 13.0, driver 580.173.02), with up to 1,024 audio frames per reply:

| | This quant | Original Drama 3 |
|---|---:|---:|
| Weights and codec on disk | 20.76 GiB | 38.63 GiB |
| Peak device memory, engine-reported | 19.35 GiB | 36.95 GiB |
| Host memory used, largest drop in available memory | about 21.3 GiB | about 39.3 GiB |
| Process start to ready, median of 4 cold starts | 26.3 s | 5.9 s |

NVFP4 needs a GPU and an engine that support it. It has been run only on GB10.

## Full model against this quant

One NVIDIA GB10, one engine binary for every arm, the 24 English texts of the benchmark corpus one request at a
time, and 4 seed salts (0 to 3) per arm. "Reference voice" is a model-invented voice given as a reference clip;
"built-in voice" is the engine's default voice with no reference. Arms:

| Arm | Checkpoint | Voice | Guidance |
|---|---|---|---|
| Full FP8 | original | reference voice | none |
| Quant NVFP4, γ 1.5 | this quant | reference voice | γ 1.5 |
| Full FP8, γ 1.5 | original | reference voice | γ 1.5 |
| Full FP8, built-in voice | original | built-in voice | none |
| Quant NVFP4, built-in voice | this quant | built-in voice | γ 1.5 requested; inactive without a reference (the engine logged γ 1.00) |

### Results

| Measure | Full FP8 | Quant NVFP4, γ 1.5 | Full FP8, γ 1.5 | Full FP8, built-in voice | Quant NVFP4, built-in voice |
|---|---:|---:|---:|---:|---:|
| Weights + codec on disk (GiB) | 38.63 | 20.76 | 38.63 | 38.63 | 20.76 |
| Peak device memory (GiB) | 36.95 | 19.35 | 37.20 | 36.95 | 19.35 |
| Host available-memory drop, max (GiB) | 39.34 | 21.27 | 39.37 | 38.67 | 21.23 |
| Process start to ready, median of 4 cold starts (s) | 5.9 | 26.3 | 5.9 | 5.7 | 26.7 |
| Real-time factor, warm p50, salt 0 | 0.846 | 0.825 | 0.940 | 0.629 | 0.618 |
| Real-time factor, warm p95, salt 0 | 1.539 | 1.462 | 1.832 | 0.670 | 0.656 |
| Whole reply, warm p50, salt 0 (s) | 9.53 | 8.38 | 9.30 | 5.33 | 5.37 |
| First audio when streaming, p50, salt 0 (s) | 2.428 | 1.955 | 2.515 | 0.141 | 0.113 |
| Warm requests succeeded | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed |
| Word error rate, mean [95 % CI] | 1.42 % [0.00, 3.48] | 1.31 % [0.04, 3.32] | 1.47 % [0.08, 3.46] | 6.23 % [2.45, 10.74] | 2.92 % [0.24, 6.57] |
| Failed takes (collapsed) | 0/96 | 0/96 | 0/96 | 9/96 | 5/96 |
| Speaker cosine to the full model's take | 0.901 (itself, other salt) | 0.900 | 0.899 | 0.309 (itself, other salt) | 0.308 |

| Paired comparison | Pairs | Failed only in first | Failed only in second | Exact McNemar p | WER difference, mean [95 % CI over texts] |
|---|---:|---:|---:|---:|---|
| Quant NVFP4, γ 1.5 vs Full FP8 | 96 | 0 | 0 | 1.000 | -0.11 pp [-0.35, +0.08] |
| Full FP8, γ 1.5 vs Full FP8 | 96 | 0 | 0 | 1.000 | +0.05 pp [-0.24, +0.34] |
| Quant NVFP4, γ 1.5 vs Full FP8, γ 1.5 | 96 | 0 | 0 | 1.000 | -0.16 pp [-0.47, +0.00] |
| Quant NVFP4, built-in voice vs Full FP8, built-in voice | 96 | 3 | 7 | 0.344 | -3.31 pp [-7.88, +0.54] |

Real-time factor is the client's end-to-end time over the audio's length (lower is faster). p50 and p95 are
nearest-rank over salt 0's warm passes; start-to-ready is an arithmetic median. Quality scores the first warm pass
of each salt: 24 takes per salt, 96 per arm. A take fails if it reaches the 1,024-frame cap or collapses (word
error rate above 0.2, or 3 s or more of silence inside the speech); every failed take here collapsed. Word error
rate: Whisper large-v3 with Whisper's English normalizer; the 95 % intervals resample the 24 texts 10,000 times.
Speaker cosine: SpeechBrain `spkrec-ecapa-voxceleb` embeddings against the full model's take of the same text and
salt; "itself, other salt" is the full model against its own take at another salt, the baseline. The built-in
voice varies from take to take, so its cosines are low for both models.

![Disk size and peak device memory](charts/footprint.png)

![Speed per arm](charts/speed.png)

![Word error rate, failed takes and speaker similarity per arm](charts/quality.png)

### Speed under The Grill

A separate streamed-speech comparison with [The Grill](https://github.com/plotarmordev/thegrill) (`grill-perf
0.11.0`, experimental): the full model with γ 1.5 (two captures, pooled) against this quant with γ 1.5, three
texts, seed 1, 6 measured requests per text.

| Text | First audio, median, full (two captures) / quant (s) | End-to-end, median, full (two captures) / quant (s) | Real-time factor, full / quant | Quant audio length / full |
|---|---:|---:|---:|---:|
| short-07 | 2.486–2.498 / 1.893 | 5.572–5.606 / 4.333 | 1.267 / 1.197 | 0.821 |
| medium-01 | 2.524 / 1.918 | 9.157–9.187 / 8.410 | 0.963 / 0.871 | 1.015 |
| long-01 | 2.602–2.605 / 1.984 | 20.811–20.888 / 17.419 | 0.797 / 0.761 | 0.876 |

The overall outcome is INCONCLUSIVE, by design: the two sides serve different models, and on that axis The Grill
reports each gate but never an overall pass. First audio and end-to-end time passed their 5 % gates in all three
texts. The duration-ratio gate (bound 0.100) failed in two: the quant's audio was 0.821 and 0.876 of the full
model's length for short-07 and long-01. Shorter audio also lowers end-to-end time, so part of the quant's lower
end-to-end time there is less audio, not faster synthesis. The Grill measures speed only, not quality.

### What these results do and do not show

- **No significant quality difference.** Every quality difference between the quant and the full model is inside
  its 95 % interval or has McNemar p ≥ 0.34. That is not proof of equal quality: 24 texts and 4 salts cannot detect
  small differences.
- **Slow cold start.** The quant takes about 26 s from process start to ready against about 6 s for the original.
  Warm, it is about as fast or slightly faster.
- **Built-in voice fails takes.** Without a reference voice both models collapsed on some takes (full 9/96, quant
  5/96; the difference is not significant). Use a reference voice.
- **Narrow test.** English only, 24 texts, 4 salts, one GPU model, one engine build, one request at a time, and
  greedy sampling as that engine reports it. Fish Audio's own examples use temperature 1.0, top-p 0.9 and top-k 30;
  this quant has not been measured that way.

## Limitations

- Measured only in English. Drama 3 lists many more languages; this quant inherits them untested, and pruning
  experts by English routing statistics may hurt other languages more than English.
- Multi-speaker scenes (`<|speaker:N|>`), long texts beyond 1,024 audio frames per reply, Drama 3's natural-language
  and tag controls, and concurrent requests were not measured.
- The pruned experts are gone. Inputs that route to them use the remaining experts instead; the effect on unusual
  voices, styles or languages is unknown.
- Only one engine loads it so far: branch `drama-nvfp4` of plotarmordev/TensorFold at commit `b1f4b9b` (see
  [Running it](#running-it)).

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

I found no audio watermark or provenance signal in Drama 3's open-weights inference, and none in the engine this
quant was measured on:

- **Fish Audio's reference.** In [SGLang-Omni](https://github.com/sgl-project/sglang-omni) at commit
  `ab095ef9`, the runtime Drama 3's card points to, the codec decodes the audio tokens (`quantizer.decode`, then
  the decoder) and the samples are written as 16-bit PCM or a plain WAV with only the `RIFF`, `fmt ` and `data`
  chunks. Its code and dependencies contain no watermarking or provenance step (I searched for watermark,
  AudioSeal, SilentCipher, WavMark, PerTh and C2PA). Drama 3's model card, config and file list mention none, and
  `codec.pth` holds only encoder, quantizer and decoder weights.
- **The engine used here.** It decodes with the same unchanged `codec.pth`, converts the samples to 16-bit PCM the
  same way, and writes the same minimal WAV header. It adds no mark and removes none.
- **What this does not cover.** Fish Audio's hosted service may differ. A signal learned into the codec's weights
  would not show up in code or tensor names; I ran no watermark detectors. This quant keeps `codec.pth`
  byte-identical, so it carries whatever the original codec produces.

The license forbids removing, disabling or obscuring any watermark or provenance signal the Fish Audio Materials
apply; nothing here was changed to do so.
