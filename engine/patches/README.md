# Engine patch series

The full-model series is empty: TensorFold v1.1.9, pinned in [`../UPSTREAM`](../UPSTREAM), carries every
feature the old patches 0001–0015 added, so `scripts/build-engine.sh --weights full` builds the release as is.
The empty `SHA256SUMS` declares that empty series; deleting it, or listing a patch that is not here, fails the
build before checkout. The features and their upstream pull requests are in the
[README](../../README.md#full-model-engine).

New patches are made with `git format-patch` against the pinned commit, start again at 0001 and are applied in
file-name order with `git am`. All patches are Apache-2.0 ([`../../LICENSE`](../../LICENSE)). The Drama 3 model
weights and codec checkpoint the engine loads are under the Fish Audio Community License v1.0 and are downloaded
separately; see [`NOTICE`](../../NOTICE).

`SHA256SUMS` pins every patch.

Linux `--weights nvfp4` uses the separate [CUDA NVFP4 series](../nvfp4/patches/README.md).
All profiles require an exact manifest-to-patch match before applying anything.

## Native macOS NVFP4 series

On Darwin, `--weights nvfp4` uses the separate pin in
[`../nvfp4/macos/UPSTREAM`](../nvfp4/macos/UPSTREAM) and the ordered, checksummed series below.
It does not change the full-model series above or the CUDA NVFP4 pin.

The pin is TensorFold main, which carries what the old Mac patches 0001–0006 added: packed NVFP4 experts,
compact routing and reference CFG ([#1268](https://github.com/ashhart/TensorFold/pull/1268)), request
sampling controls ([#1277](https://github.com/ashhart/TensorFold/pull/1277)) and dialogue speakers
([#1278](https://github.com/ashhart/TensorFold/pull/1278)).

| Patch | Scope | Upstream |
|---|---|---|
| [`0001-gb10-voice-files.patch`](../nvfp4/macos/patches/0001-gb10-voice-files.patch) | `--voices` loads the CUDA engine's `drama3-voice/1` and `/2` files, `codes_sha256` checked | this repo |
| [`0002-speech-no-audio-is-an-error.patch`](../nvfp4/macos/patches/0002-speech-no-audio-is-an-error.patch) | A take that ends speech before its first audio frame is a 500, whole or streamed | this repo |
| [`0003-drama3-compact-lm-head.patch`](../nvfp4/macos/patches/0003-drama3-compact-lm-head.patch) | Load a compact lm_head listed by `drama_head_rows` ([`recipe/trim_head.py`](../../recipe/trim_head.py)) | this repo |
| [`0004-drama-read-BF16-codec-checkpoints.patch`](../nvfp4/macos/patches/0004-drama-read-BF16-codec-checkpoints.patch) | Read a BF16 `codec.pth`, widened exactly to FP32 | this repo |
| [`0005-int3-routed-experts.patch`](../nvfp4/macos/patches/0005-int3-routed-experts.patch) | Int3 routed experts (3.5 bits) unpacked in `d3_nvfp4_up/down`, with `quant_check` device cases; NVFP4 layers unchanged | this repo |

The [Mac qualification receipt](../../results/macos-nvfp4.json) separates observed GPU/model results,
the initial failed fixture and focused correction, and the final CPU-only CLI integration build.
