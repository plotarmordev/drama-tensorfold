# Engine patch series

The `*.patch` files here are opt-in patches made with `git format-patch` against the upstream TensorFold commit
pinned in [`../UPSTREAM`](../UPSTREAM). `scripts/build-engine.sh` applies them in file-name order with `git am`.
Each one is a feature branch's net diff over the patches before it; the list, the flag each adds and the upstream
pull request that carries it are in the [README](../../README.md#the-patches).

All patches are Apache-2.0 ([`../../LICENSE`](../../LICENSE)). The Drama 3 model weights and codec checkpoint the
engine loads are under the Fish Audio Community License v1.0 and are downloaded separately; see
[`NOTICE`](../../NOTICE).

`SHA256SUMS` pins every patch.

Linux `--weights nvfp4` uses the separate [CUDA NVFP4 series](../nvfp4/patches/README.md).
All profiles require an exact manifest-to-patch match before applying anything.

Full-model patches 0013/0014 add [request sampling controls](../../recipe/README-sampling.md)
and document the existing JSON floating-point boundary. They backport
[TensorFold #1277](https://github.com/ashhart/TensorFold/pull/1277) without moving the pin.

Full-model patch 0015 adds [speaker-tagged dialogue](../../recipe/README-dialogue.md)
from [TensorFold #1278](https://github.com/ashhart/TensorFold/pull/1278).
The full pin predates the native voice writer; its Python catalog writer accepts
the same canonical speaker IDs without adding an engine tool absent from that pin.

Patch 0012 is corrected in place: it owns one reusable Metal recurrent-state snapshot per lane,
invalidates incompatible live KV before prefill, and uses the prompt builder's exact invariant
boundary including tower lookahead. It preserves uncached chunk cuts and CUDA execution.
Its CPU regressions cover identity, cancellation/error invalidation, allocation ownership and
prompt boundaries; real-model transition/corpus/stream checks are described in
[`recipe/README-macos.md`](../../recipe/README-macos.md#correctness-gates).

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
