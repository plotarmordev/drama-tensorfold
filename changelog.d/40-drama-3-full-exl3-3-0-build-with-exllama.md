### Added

- [`results/exl3-build/`](results/exl3-build/README.md): the GB10 job (`gb10-exl3.sh`, ~3.5 GPU-hours planned)
  that encodes all 5,900 kept routed experts of `t5900-cover` at EXL3 3.0 bits with exllamav3 `bd1f519` from
  the FP8 originals and the NVFP4-mix rows, one layer file and record at a time, so a stopped job resumes.
  `results/exl3-ldlq/exllamav3-env.sh` holds the exllamav3 setup both GB10 jobs share.
- Finished layer records carry shard checksums; resuming rejects incomplete/corrupt pairs and changed recipes.
  Resumes hash the actual row and FP8 shard bytes, rejecting changed input even when the rows manifest and
  checkpoint index are unchanged.
  The job takes only the staged v3 rows; BOTTOM's older capture is not admitted (its saved report misses
  layer 36 experts 188 and 243 beyond the approved exceptions).
- `assemble.py` validates completed EXL3 layers before replacing routed tensors, preserving dense weights,
  codec and pruning/head metadata; the final index contains only the new EXL3 expert roles.
- The Mac patch series now loads mul1 EXL3 3.0 routed experts and dispatches indexed Metal gate/up and down
  projections through compact router slots. Synthetic decode/prompt operator checks pass beside int3 and NVFP4.
- Job `bottom-ccd34101e181` encoded all 5,900 experts in 1.78 GPU-hours. The assembled `trim-bf16-exl3-3.0`
  (13.51 GiB) passes the card 37 protocol's bars in both voices, so `pick.py` names it over `trim-bf16-int3`
  (14.50 GiB); `builds.json`, both READMEs and the Mac listening page carry the EXL3 build.
- Patch 0006 has the mail header `build-engine.sh`'s `git am` needs, keeps the per-frame candidate-head
  `rmsnorm` its `forward.zig` hunk had dropped, and its EXL3 admission tests (scalar `mul1` markers) run in
  `test-drama-metal-quant-host`.
