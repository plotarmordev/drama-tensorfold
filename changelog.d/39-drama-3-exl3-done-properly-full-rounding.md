### Added
- `results/exl3-ldlq/`: `ldlq.patch` makes TensorFold's `tf-exl3-encode` round by exllamav3's LDLQ over the full
  Hessian (rotated, damped, Cholesky feedback from the later 16-row blocks). On card 35's six experts EXL3 at 3.0 bits
  has 0.40× the int3 build's Hessian error at 0.87× its bytes (2.1× NVFP4's, down from 63×); 2.5 bits is 0.82× at
  0.73×. `measure.py` scores each expert against NVFP4 and int3, and the README gives the 13.51 GiB build plan.
- `results/exl3-ldlq/gb10-exllamav3.sh`: one Spark GPU job runs exllamav3 `bd1f519`'s own `quantize_exl3` (vendored)
  on the same six experts' W and Hessians (`hessians/`, from `measure.py hessians`) at 3.0 and 2.5 bits;
  `measure.py parity` then shows whether ours is within 1.1× of it on every projection.
