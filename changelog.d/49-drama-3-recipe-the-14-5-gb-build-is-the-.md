### Changed

- The 14.5 GB int3 build is the Mac default: `recipe/pilot_macos.py` without `--weights` or `--checkpoint` serves
  `--weights int3-5900`, and `scripts/build-engine.sh` without `--weights` builds its engine (`nvfp4`) on Darwin
  (`full` elsewhere). `--weights int3-5900` now downloads `plotarmordev/drama-3-NVFP4-Pruned-5900` at `8c6c3d79`
  (`recipe/sources.json` `int3_5900_checkpoint`, the weights of release `0a0afd5d`) on first start and checks it
  against `recipe/checksums/trim-bf16-int3.sha256`; no splice step. `nvfp4-5900` and `nvfp4` keep their own
  revisions and pins. `recipe/speech-server.example.ini` serves `int3-5900`.
- `README.md` names what card 47 measured on a GB10: the int3 build on the `engine/nvfp4` binary with the FP32 codec
  swapped in, as fast as the 5,900 quant in 2.16 GiB less device memory, and how to serve it that way; the 20.8 GB
  quant stays the GB10 default.
