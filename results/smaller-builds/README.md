# The smaller builds of the 5,900-expert quant on the Mac

Four cuts of `t5900-cover` (published as `plotarmordev/drama-3-NVFP4-Pruned-5900`), the first three each on the one
before, measured on the M5 Ultra with the Mac NVFP4 engine (`scripts/build-engine.sh --weights nvfp4`). The winner is
the smallest build whose mean word error rate is at most twice `t5900`'s and whose failed takes are not more than
`t5900`'s, in both voices ([`pick.py`](pick.py)).

| Build | What changes | Weights + codec (GiB) |
|---|---|---:|
| `t5900` | the published quant, the baseline | 18.34 |
| `trim` | lm_head trimmed to the rows Drama emits (`recipe/trim_head.py`) | 17.39 |
| `trim-bf16` | and the codec in BF16 (`recipe/codec_bf16.py`) | 16.66 |
| `trim-bf16-int3` | and the 5,900 routed experts in int3 (3.5 bits), GPTQ from the FP8 originals | 14.50 |
| `trim-bf16-exl3-3.0` | `trim-bf16` with the 5,900 routed experts in exllamav3's EXL3 3.0 ([`../exl3-build`](../exl3-build/README.md)) | 13.51 |

## Result: `trim-bf16-exl3-3.0`

The 24 corpus texts, salts 0 and 1: 48 takes per build and voice.

| Build | Reference voice WER mean / pooled | Failed | Built-in voice WER mean / pooled | Failed |
|---|---:|---:|---:|---:|
| `t5900` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim-bf16` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim-bf16-int3` | 1.78 % / 0.97 % | 0/48 | 5.78 % / 3.73 % | 4/48 |
| `trim-bf16-exl3-3.0` | 1.61 % / 0.90 % | 0/48 | 9.47 % / 7.16 % | 6/48 |

- `trim-bf16-exl3-3.0` passes both bars in both voices: 1.08× the baseline's reference WER and 1.37× its built-in
  WER, no failed reference take and 6 failed built-in takes against 6. At 13.51 GiB it is 0.99 GiB under int3 and
  4.83 GiB (26 %) under `t5900`.
- `trim-bf16-int3` also passes (1.19× and 0.84×, 4 failed built-in takes); the rule takes the smaller build.
- `trim` speaks the same audio as `t5900`, byte for byte; `trim-bf16`'s differs, with the same word errors on every
  take.
- The three NVFP4 builds fail the same six built-in takes. `trim-bf16-int3` fails two of them (`short-01` s0,
  `long-01` s1) and one text of its own: built-in `medium-05` runs to the 1024-frame cap (47.55 s) on both salts,
  speaking only its first sentence (WER 0.68), where the NVFP4 builds say it all in about 8 s.
- Patch 0008 fixes both: eos stays in every nucleus (both truncated draws excluded it under a silence-dominant
  distribution, so the stop could not be drawn), serve fills an unset guard at −40 dBFS / 3.5 s, and a take the
  guard silenced is redrawn with the next prompt-derived seed (the int3 collapse after a sentence is per-seed: 5
  of 8 `medium-05` seeds speak the whole text). After it, [`stop-decision-20261121.json`](stop-decision-20261121.json):
  built-in failed takes 4 → 1 of 48 (`long-01` s1's 3.1 s stall inside speech stays, byte-identical), no take
  caps, `medium-05` and `short-01` s0 speak the whole text and end on eos, the 44 untouched built-in takes and 47
  of 48 reference takes are byte-identical, and the reference holds 0 failed and 1.78 % / 0.97 %.

| File | What it holds |
|---|---|
| [`builds.json`](builds.json) | Per build and voice: bytes, takes, mean and pooled WER, failed takes and which, and the winner |
| [`measure.py`](measure.py) | speak (one build, one voice, the salts), score (Whisper large-v3 on MPS), report, listen |
| [`gb10-int3.sh`](gb10-int3.sh) | The GB10 job (`job start gpu bottom`, run `bottom-d8d4f4428fba`) that wrote the int3 experts |
| [`gb10-start.sh`](gb10-start.sh) | Starts a GB10 job once a GPU is free: retries while no window is open, never pauses DeepSeek |
| [`int3.sha256`](int3.sha256) | SHA-256 of its 40 `layer-LL.safetensors` |
| [`splice_int3.py`](splice_int3.py) | `trim-bf16` with its kept experts taken from those files |
| [`hub-release.json`](hub-release.json) | The published revision downloaded fresh from the Hub, checked against its `MANIFEST.sha256` and spoken on the Mac |
| [`mac-speed.json`](mac-speed.json) | Warm real-time factor per take of `trim-bf16-int3` and `t5900` on the M5 Ultra, one engine binary, every run with the clean ones marked |
| [`release_charts.py`](release_charts.py) | The Hugging Face card's `footprint.png`, `quality.png` and `speed.png` from `builds.json` and `mac-speed.json` |
| [`gb10-speed-stage.sh`](gb10-speed-stage.sh) | `bottom shell`: the published int3 revision from the Hub, checked, with the pilot's FP32 codec swapped in and its `SHA256SUMS` |
| [`gb10-speed.sh`](gb10-speed.sh), [`spark_speed.py`](spark_speed.py) | The GB10 job for the card's Spark rows: both builds, both voices, one engine binary (`engine/nvfp4`, built by [`prebuild-spark.sh`](../exl3-spark/prebuild-spark.sh) with `PIN=engine/nvfp4 OUT=drama3-int3-engine`), warm p50 RTF, load and peak memory |
| [`spark-speed.json`](spark-speed.json) | Its run on TOP (job `top-1fa2dfce6839`): the four rows and what was checked |
| [`stop-decision-20261121.json`](stop-decision-20261121.json) | The card-55 stop fix: per-frame cause traces (eos excluded from both nuclei), the two engine changes, before/after arms and byte-identity, and the issue-32 hypothesis |
| [`trace-20261121/`](trace-20261121/), [`trace_eos.py`](trace_eos.py) | The card-55 per-frame evidence: prompt rows, chosen tokens and eos standing per frame for the failing and passing takes, and the tool that derives it |

Same protocol and failure rule as [`full-vs-quant-01`](../full-vs-quant-01/README.md): `--max-frames 1024`, the
`eq-3ref` reference voice with guidance γ 1.5 batched or the built-in voice; failed = capped, WER > 0.2 or a quiet
run of 3 s or more inside speech. The listening page (`measure.py listen`: salt 0, one text of each length and
`medium-05`, both voices, every build) and the audio stay on the Mac and are not published: open
`/Users/Shared/agent-work/drama-assets/drama3-smaller-builds/listening.html` in a browser there. No
speaker-similarity claim; speed and memory are `mac-speed.json`'s and `spark-speed.json`'s alone.

## Published

`trim-bf16-int3` replaced the NVFP4 build in
[`plotarmordev/drama-3-NVFP4-Pruned-5900`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900) at
revision `0a0afd5d` on 15 Nov 2026; the NVFP4 build stays at revision `4bfe2c32`, which `--weights nvfp4-5900`
pins. On one engine binary, [`mac-speed.json`](mac-speed.json)'s clean runs (every warm take under 0.65 RTF on a
shared machine) put the int3 build at 0.511 median warm RTF per take and `t5900` at 0.520: equally fast, the gap
within the run-to-run spread; the three contended runs are kept in its `runs`. The CUDA pin `b1f4b9b` reads its
BF16 codec through [patch 0007](../../engine/nvfp4/patches/0007-drama-read-BF16-codec-checkpoints.patch), exactly
the original FP32 codec rounded to BF16. With the FP32 codec swapped in, [`spark-speed.json`](spark-speed.json) puts
both builds at the same warm p50 RTF on a GB10 (int3 0.614 built-in and 0.809 reference voice, `t5900` 0.619 and
0.831), the int3 build at 14.77 GiB peak device memory against 16.93.

[`hub-card.md`](hub-card.md) is the Hugging Face card's next README.md: the default build on a Mac and a GB10, served
with its BF16 codec.
