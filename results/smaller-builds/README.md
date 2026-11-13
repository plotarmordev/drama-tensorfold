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

## Result: `trim-bf16-int3`

The 24 corpus texts, salts 0 and 1: 48 takes per build and voice.

| Build | Reference voice WER mean / pooled | Failed | Built-in voice WER mean / pooled | Failed |
|---|---:|---:|---:|---:|
| `t5900` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim-bf16` | 1.50 % / 0.75 % | 0/48 | 6.91 % / 6.27 % | 6/48 |
| `trim-bf16-int3` | 1.78 % / 0.97 % | 0/48 | 5.78 % / 3.73 % | 4/48 |
| `trim-bf16-exl3-3.0` | 645 % / 450 % | 48/48 | 397 % / 243 % | 48/48 |

- `trim-bf16-int3` passes both bars in both voices: 1.19× the baseline's reference WER and 0.84× its built-in WER,
  no failed reference take and 4 failed built-in takes against 6. It is 3.84 GiB (21 %) smaller than `t5900`.
- `trim` speaks the same audio as `t5900`, byte for byte; `trim-bf16`'s differs, with the same word errors on every
  take.
- The three NVFP4 builds fail the same six built-in takes. `trim-bf16-int3` fails two of them (`short-01` s0,
  `long-01` s1) and one text of its own: built-in `medium-05` runs to the 1024-frame cap (47.55 s) on both salts,
  speaking only its first sentence (WER 0.68), where the NVFP4 builds say it all in about 8 s. The rule counts
  failed takes, not which ones; with two salts, half the four of [`quant-15gb`](../quant-15gb/README.md), that
  runaway is the thing to hear before shipping.
- `trim-bf16-exl3-3.0` is 0.99 GiB smaller than int3 but fails every take in both voices: each runs to the
  1024-frame cap, and Whisper hears hallucinated filler ("Thank you for watching") or repeated clicks, not the text.
  The rule rejects it; int3 stays the winner.

| File | What it holds |
|---|---|
| [`builds.json`](builds.json) | Per build and voice: bytes, takes, mean and pooled WER, failed takes and which, and the winner |
| [`measure.py`](measure.py) | speak (one build, one voice, the salts), score (Whisper large-v3 on MPS), report, listen |
| [`gb10-int3.sh`](gb10-int3.sh) | The GB10 job (`job start gpu bottom`, run `bottom-d8d4f4428fba`) that wrote the int3 experts |
| [`gb10-start.sh`](gb10-start.sh) | Starts a GB10 job once a GPU is free: retries while no window is open, never pauses DeepSeek |
| [`int3.sha256`](int3.sha256) | SHA-256 of its 40 `layer-LL.safetensors` |
| [`splice_int3.py`](splice_int3.py) | `trim-bf16` with its kept experts taken from those files |

Same protocol and failure rule as [`full-vs-quant-01`](../full-vs-quant-01/README.md): `--max-frames 1024`, the
`eq-3ref` reference voice with guidance γ 1.5 batched or the built-in voice; failed = capped, WER > 0.2 or a quiet
run of 3 s or more inside speech. The listening page (`measure.py listen`: salt 0, one text of each length and
`medium-05`, both voices, every build) and the audio stay on the Mac and are not published: open
`/Users/Shared/agent-work/drama-assets/drama3-smaller-builds/listening.html` in a browser there. No speed, memory or
speaker-similarity claim.
