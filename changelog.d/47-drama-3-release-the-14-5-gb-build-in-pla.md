### Changed

- `plotarmordev/drama-3-NVFP4-Pruned-5900` now holds the 14.50 GiB `trim-bf16-int3` build (int3 routed experts,
  trimmed head, BF16 codec) with a new card, NOTICE, `MANIFEST.sha256` and charts; the NVFP4 build stays at revision
  `4bfe2c32`, which `--weights nvfp4-5900` still pins. `recipe/checksums/trim-bf16-int3.sha256` pins the weights,
  codec, config and tokenizer files only, as the other pins do, so a download of the repo passes it.

### Added

- [`results/smaller-builds/mac-speed.json`](results/smaller-builds/mac-speed.json): on one engine binary on the
  M5 Ultra, the int3 build's warm RTF is 0.511 median against `t5900`'s 0.620 on a shared machine (no slower);
  [`release_charts.py`](results/smaller-builds/release_charts.py) draws the card's charts.
