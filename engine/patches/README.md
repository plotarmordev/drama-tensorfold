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

| Patch | Scope | Upstream |
|---|---|---|
| [`0001-drama3-metal-nvfp4-cfg.patch`](../nvfp4/macos/patches/0001-drama3-metal-nvfp4-cfg.patch) | Packed NVFP4 experts, compact routing, native reference CFG and registered numerical contracts | [TensorFold #1268](https://github.com/ashhart/TensorFold/pull/1268) |
| [`0002-shared-gate-stress-regression.patch`](../nvfp4/macos/patches/0002-shared-gate-stress-regression.patch) | Test-only correction that distinguishes reduced-precision accumulation after sigmoid; no kernel or tolerance change | [TensorFold #1268](https://github.com/ashhart/TensorFold/pull/1268) |
| [`0003-shared-cli-extension-point.patch`](../nvfp4/macos/patches/0003-shared-cli-extension-point.patch) | Group CFG flags beside voice selection without occupying shared CLI-table extension points; no parsing or numerical change | [TensorFold #1268](https://github.com/ashhart/TensorFold/pull/1268) |
| [`0004-speech-sampling-controls.patch`](../nvfp4/macos/patches/0004-speech-sampling-controls.patch) | Request sampling parameters and exact seeds, retaining CFG rows | [TensorFold #1277](https://github.com/ashhart/TensorFold/pull/1277) |
| [`0005-sampling-json-boundaries.patch`](../nvfp4/macos/patches/0005-sampling-json-boundaries.patch) | Parsed-JSON floating boundary documentation and regressions | [TensorFold #1277](https://github.com/ashhart/TensorFold/pull/1277) |
| [`0006-dialogue-speakers.patch`](../nvfp4/macos/patches/0006-dialogue-speakers.patch) | Canonical speaker IDs in request/catalog transcripts and the native writer | [TensorFold #1278](https://github.com/ashhart/TensorFold/pull/1278) |

The [Mac qualification receipt](../../results/macos-nvfp4.json) separates observed GPU/model results,
the initial failed fixture and focused correction, and the final CPU-only CLI integration build.
