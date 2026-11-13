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
the same weights (`pruned-5900`: 2.233 guided, 1.43 unguided) that is 41 min of speech plus four loads (a server
per arm and salt, about 2 min); if EXL3 decodes 1.3× slower than NVFP4 experts, 55 min. **The job asks for a
one-hour window.**

## Result

Filled in from `/work/out/exl3-spark/summary.json` when the window has run; `t5900-cover`'s columns are the same
GB10's published numbers ([`quant-15gb`](../quant-15gb/compare.json), the fork engine) and the same engine base and
weights ([`drama3-pruned-main`](https://github.com/plotarmordev/TensorFold/tree/main/docs/receipts/drama3-pruned-main),
main `132bd1238`).

| Build | Voice | Load (s) | Peak device (GiB) | Seconds per take mean / p50 | Capped of 48 |
|---|---|---:|---:|---|---:|
| `trim-bf16-exl3-3.0` | reference | pending | pending | pending | pending |
| `trim-bf16-exl3-3.0` | built-in | pending | pending | pending | pending |
| `t5900-cover` (quant-15gb) | reference | 20.1 | 16.93 | RTF p50 0.828 | 0 |
| `t5900-cover` (quant-15gb) | built-in | 20.1 | 16.93 | RTF p50 0.615 | 0 |
| `t5900-cover` (main `132bd1238`) | reference | 26.0 | 19.29 | RTF 2.233 | — |
