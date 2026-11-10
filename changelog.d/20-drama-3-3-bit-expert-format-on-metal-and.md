### Added

- Int3 routed experts (3.5 bits): three bit planes per 16 inputs for E2M1 magnitudes 0, 2, 4, 6, under NVFP4's
  E4M3 block-16 scales and FP32 global. Metal patch 0011 unpacks them in `d3_nvfp4_up/down` and CUDA NVFP4 patch
  0006 in the NVFP4 expert kernel; a layer whose gate codes are `[n, 3k/8]` bytes is read as int3. The CUDA
  patch's `tools/zig/drama_nvfp4_gptq.py --mode int3 --provenance nvfp4-provenance.json` quantizes the FP8
  originals with each layer's recorded GPTQ rows, kappa and damping, and `tf-drama-quant --encoded` assembles its
  files into a checkpoint, recording each layer's format. 4-bit NVFP4 checkpoints decode unchanged.
