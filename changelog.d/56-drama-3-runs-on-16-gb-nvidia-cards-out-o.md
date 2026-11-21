### Changed

- `scripts/build-engine.sh` builds GPU 0's compute capability as the CUDA engine's SASS target, read from `nvidia-smi`
  (sm_121 on a GB10, sm_120 on an RTX 5060 Ti); `--sm` overrides it, a build machine without a GPU must pass it, and
  the Metal engine refuses it. A GB10 still passes `-Dsm=121`, the engine's own default, so its build is unchanged.
- `recipe/pilot.py` sizes the host-memory floor, `TENSORFOLD_MEMORY_RESERVE_GIB` and the CUDA allocator cap from the
  machine instead of the fixed 88 / 34 / 54 GiB: a GB10's unified pool keeps those measured figures, while a discrete
  card takes its CUDA cap from device memory (total less 1 GiB for the driver) and its host floor from RAM (a quarter,
  at least 8 GiB), with a 0.5 GiB reserve. `--reserve-gib` still overrides the reserve.

### Fixed

- The 14.5 GB int3 build loads and serves out of the box on a 16 GB NVIDIA card: the engine is built for the card's
  compute capability and the launcher no longer asks a 16 GB host for 88 GiB or keeps 34 GiB back from the card
  (addresses [#68](https://github.com/plotarmordev/drama-tensorfold/issues/68), reported by @Nipale-ai).
