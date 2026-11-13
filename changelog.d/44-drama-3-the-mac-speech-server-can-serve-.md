### Added

- `--weights int3-5900` (and `weights = int3-5900` in `speech-server.ini`) serves the 14.5 GB `trim-bf16-int3`
  build with the Mac NVFP4 engine, whose Metal patch 0005 reads int3 experts. It is verified against
  `recipe/checksums/trim-bf16-int3.sha256` on every start and never downloaded: a missing or changed checkpoint is
  refused, naming `results/smaller-builds/splice_int3.py`. `nvfp4-5900` stays the default.

### Changed

- `recipe/speech_server.py`: `checkpoint` is optional; unset, the selected build's own folder in the repository root
  is served, so switching builds is the `weights` line alone.
