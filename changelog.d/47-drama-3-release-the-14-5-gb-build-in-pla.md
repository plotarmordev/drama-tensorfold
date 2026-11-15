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
- [`results/smaller-builds/spark-speed.json`](results/smaller-builds/spark-speed.json): on one GB10 and one
  `engine/nvfp4` binary the int3 build (with the FP32 codec) and `t5900` run at the same warm p50 RTF (0.614 / 0.809
  against 0.619 / 0.831, built-in / reference voice), the int3 build at 14.77 GiB peak device memory against 16.93;
  the card's Spark rows and CUDA how-to carry it. [`gb10-speed.sh`](results/smaller-builds/gb10-speed.sh) runs on
  TOP from folders sent with `put top`; `results/exl3-spark/prebuild-spark.sh` takes `PIN`/`OUT` for another engine
  pin.
