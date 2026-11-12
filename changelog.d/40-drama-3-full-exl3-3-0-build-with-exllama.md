### Added

- [`results/exl3-build/`](results/exl3-build/README.md): the GB10 job (`gb10-exl3.sh`, ~3.5 GPU-hours, not yet
  run) that encodes all 5,900 kept routed experts of `t5900-cover` at EXL3 3.0 bits with exllamav3 `bd1f519` from
  the FP8 originals and the NVFP4-mix rows, one layer file and record at a time, so a stopped job resumes.
  `results/exl3-ldlq/exllamav3-env.sh` holds the exllamav3 setup both GB10 jobs share.
- Finished layer records carry shard checksums; resuming rejects incomplete/corrupt pairs and changed recipes.
  The report records the closed GPU window and missing staged rows; no build, winner or audio is claimed yet.
- `assemble.py` validates completed EXL3 layers before replacing routed tensors, preserving dense weights,
  codec and pruning/head metadata; the final index contains only the new EXL3 expert roles.
- The Mac patch series now loads mul1 EXL3 3.0 routed experts and dispatches indexed Metal gate/up and down
  projections through compact router slots. Synthetic decode/prompt operator checks pass beside int3 and NVFP4;
  full-model speech and exllamav3-produced checkpoint verification still require the GPU output.
