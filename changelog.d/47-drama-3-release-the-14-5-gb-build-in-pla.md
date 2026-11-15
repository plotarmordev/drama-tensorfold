### Changed

- `plotarmordev/drama-3-NVFP4-Pruned-5900` now holds the 14.50 GiB `trim-bf16-int3` build (int3 routed experts,
  trimmed head, BF16 codec) with a new card, NOTICE, `MANIFEST.sha256` and charts; the NVFP4 build stays at revision
  `4bfe2c32`, which `--weights nvfp4-5900` still pins. `recipe/checksums/trim-bf16-int3.sha256` pins the weights,
  codec, config and tokenizer files only, as the other pins do, so a download of the repo passes it.

### Added

- [`results/smaller-builds/mac-speed.json`](results/smaller-builds/mac-speed.json): on one engine binary on the
  M5 Ultra, clean runs put the int3 build at 0.511 median warm RTF per take and `t5900` at 0.520 (equally fast);
  [`release_charts.py`](results/smaller-builds/release_charts.py) draws the card's charts.
- [`results/smaller-builds/hub-release.json`](results/smaller-builds/hub-release.json): revision `0a0afd5d`
  downloaded fresh from the Hub matches all 21 `MANIFEST.sha256` entries and speaks a line through
  `recipe/pilot_macos.py --weights int3-5900` on the Mac.
