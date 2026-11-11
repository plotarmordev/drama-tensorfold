### Changed

- `recipe/calibrate_experts.py` selects the calibration rows from the full capture in the NVFP4 build's class mix
  (`stats_pairs`), keeping at least 64 rows of every kept expert (all when it has fewer), into
  `exl3-calibration-rows-v3`; the full capture stays in `exl3-calibration-rows-v2`.
- The command fails when a kept expert has no rows, when a class's share leaves its NVFP4 share by more than 0.01
  or when the `err_h` ratio to the NVFP4 build leaves 1 ± 0.05. `report.json` gives each row-less kept expert's best
  rank among the router's 256 logits. The 12 layer-36 experts that never place above 171st over the 11 Nov capture
  are a named exception (`NEVER_ROUTED`, approved by the Drama lead on 11 Nov); any other kept expert without rows
  still fails.
- `recipe/exl3_experts.py` records its Hessian recipe and a hash of every rows file in `work/hessian.json` and does
  not resume a work folder encoded with another recipe or other rows.
