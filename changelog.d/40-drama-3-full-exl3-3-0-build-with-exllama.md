### Added

- [`results/exl3-build/`](results/exl3-build/README.md): the GB10 job (`gb10-exl3.sh`, ~3.5 GPU-hours, not yet
  run) that encodes all 5,900 kept routed experts of `t5900-cover` at EXL3 3.0 bits with exllamav3 `bd1f519` from
  the FP8 originals and the NVFP4-mix rows, one layer file and record at a time, so a stopped job resumes.
  `results/exl3-ldlq/exllamav3-env.sh` holds the exllamav3 setup both GB10 jobs share.
- Finished layer records carry shard checksums; resuming rejects incomplete/corrupt pairs and changed recipes.
  The report records the closed GPU window and missing staged rows; no build, winner or audio is claimed yet.
