# The winning smaller build end to end on the Spark

`trim-bf16-exl3-3.0` (13.51 GiB, [`pick.py`](../smaller-builds/pick.py)'s winner) loaded and spoken on the same GB10
that measured `t5900-cover`, through the Desk's GPU door. Everything the job needs is staged on BOTTOM before any
window opens — nothing is built inside the job — and BOTTOM's own files are reused: the assembled checkpoint comes
from the pilot's `drama-3-NVFP4-Pruned-5900` model and its `exl3-done` EXL3 layers, never from the Mac.

| Piece | What |
|---|---|
| [`stage.sh`](stage.sh) | `bottom shell`: verifies the pilot's 5,900-expert model and `exl3-done` shards against the repo's pins, then `trim_head.py`, `codec_bf16.py` and [`assemble.py`](../exl3-build/assemble.py) build `drama3-exl3-5900` in the pilot folder, with its `SHA256SUMS` |
| [`prebuild-spark.sh`](prebuild-spark.sh) | `bottom shell`: builds the Spark engine binary (the patch series below) in the job's own image with g++ and nvcc added, into the pilot folder, before any window |
| [`engine/nvfp4/spark/`](../../engine/nvfp4/spark/) | TensorFold main `132bd12383ba` plus the patch that loads the winning build on CUDA: EXL3 3.0 mul1 routed experts through the existing grouped CUDA kernels, the compact `drama_head_rows` lm_head and the BF16 codec |
| [`gb10-exl3-speak.sh`](gb10-exl3-speak.sh) | The one-window job: checks the staged checksums, then [`speak.py`](speak.py) — card 37's 24 texts, both voices (`eq-3ref` with γ 1.5 batched, and built-in), salts 0 and 1 through `TENSORFOLD_SEED_SALT` (the engine's speech seed is its prompt keyed by that salt) — and writes load time, peak device memory (`GET /memory`) and seconds per take |
| [`speak.py`](speak.py) | The client: a server per arm and salt, every take timed send-to-last-byte, WAVs kept for the listening page |

The `eq-3ref` voice (sha256 `86ccbe54…`) went to BOTTOM through the `put` door; it is the only file from the Mac.
It is the fork's `drama3-voice/2` file, which the pinned main engine refuses (`BadVoice`), so
[`http_voice.py`](http_voice.py) rewrites it on BOTTOM, through `bottom shell`, in main's `drama3-http-voice/1`
format: the same codes, checked against their sha256, frame-major. In the pilot folder, after `stage.sh`:
`python3 http_voice.py voices-eq3ref/eq-3ref.json voices-eq3ref-http` (that script fetched at the job's sha).

## Expected window length, stated before it starts

Card 37's takes on this GB10 are 623 s of audio with the reference voice and 746 s with the built-in voice (the
EXL3 build's built-in runaways at the 47.55 s cap included). At the RTF the same engine measured on this Spark with
the same weights (`pruned-5900`: 2.233 guided, 1.43 unguided) that is 41 min of speech plus four loads and four
repeats of the first (short) text (a server per arm and salt, about 4 min); if EXL3 decodes 1.3× slower than NVFP4
experts, 57 min. The last window took 52 min without the repeats. **The job asks for a one-hour window.**

## Result

Window `bottom-e36eec0f871a` (13 Nov 22:31–23:25, 54 min, inside the stated hour) at `85daacdf`, engine
`tensorfold-native` sha256 `1fd72f18…` built from `5aef6075`'s series (every engine owns its EXL3 scratch; the
current 0001 adds only `owner_test.zig`): all 96 takes spoken, numbers in [`summary.json`](summary.json). Every
arm and salt spoke its first text twice and got the same bytes both times. RTF is wall seconds over the WAV's
length. `t5900-cover`'s rows are the same GB10's published numbers ([`quant-15gb`](../quant-15gb/compare.json), the
fork engine) and the same engine base and weights
([`drama3-pruned-main`](https://github.com/plotarmordev/TensorFold/tree/main/docs/receipts/drama3-pruned-main),
main `132bd1238`).

| Build | Voice | Load (s) | Peak device (GiB) | Seconds per take mean / p50 | RTF all / p50 | Capped of 48 |
|---|---|---:|---:|---|---|---:|
| `trim-bf16-exl3-3.0` | reference | 10.0 | 16.18 | 52.2 / 49.7 | 4.08 / 4.68 | 0 |
| `trim-bf16-exl3-3.0` | built-in | 12.5 | 15.68 | 11.0 / 7.4 | 0.79 / 0.83 | 4 |
| `t5900-cover` (quant-15gb) | reference | 20.1 | 16.93 | — / 8.45 | — / 0.828 | 0 |
| `t5900-cover` (quant-15gb) | built-in | 20.1 | 16.93 | — / 5.03 | — / 0.615 | 0 |
| `t5900-cover` (main `132bd1238`) | reference | 26.0 | 19.29 | — | 2.233 / — | — |

The winning build loads in 10–15 s, against `t5900-cover`'s 20.1, and peaks 0.75 GiB below the fork and 3.1 GiB
below main. Built-in speech runs 1.35× the fork's NVFP4 RTF (p50). Every reference-voice take costs a fixed 36 s
plus 1.26 s per second of speech (least squares over the 48), so guided speech is 5.7× the fork's p50 RTF and 1.8×
main's. The fixed cost points at the ~1,300-token reference prompt each take prefills through the EXL3 experts
(built-in takes carry none); the window did not time prefill apart from decode. The four capped built-in takes
stopped at the 47.55 s cap.
