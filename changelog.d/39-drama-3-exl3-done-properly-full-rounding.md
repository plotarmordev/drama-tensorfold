### Added
- `results/exl3-ldlq/`: exllamav3 `bd1f519`'s own `quantize_exl3` (full-Hessian LDLQ rounding, vendored, run on a
  GB10 by `gb10-exllamav3.sh`) on card 35's six experts: EXL3 at 3.0 bits has 0.15× the int3 build's Hessian error at
  0.87× its bytes and 0.78× NVFP4's (2.5 bits: 0.31× int3, 1.6× NVFP4). `measure.py parity` scores it against ours,
  NVFP4 and int3, and the README gives the 13.51 GiB, ~3 GPU-hour build plan on exllamav3.
- `results/exl3-ldlq/ldlq.patch` makes TensorFold's `tf-exl3-encode` round by exllamav3's LDLQ feedback; with its own
  search and scales it is still 2.7× exllamav3's error, so it is kept as a measurement, not the build route.
