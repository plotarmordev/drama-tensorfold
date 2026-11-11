### Changed

- `recipe/calibrate_experts.py` weights each calibration row by its class (the class's share of the NVFP4 build's
  `stats_pairs` over its share of the rows) and records the weights in `report.json`; the `err_h` check and
  `recipe/exl3_experts.py`'s Hessians, H = Xᵀ diag(c) X / Σc, use them (one shared recipe; `down_proj` from the
  original gate and up, an expert with no rows on all of its layer's rows).
- The command fails when any kept expert has no rows, when a class weight leaves ½–2 or when the `err_h` ratio to
  the NVFP4 build leaves 1 ± 0.05. `report.json` gives each row-less kept expert's best rank among the router's 256
  logits: the 12 in layer 36 never place above 171st over the 11 Nov capture, which therefore fails.
- `recipe/exl3_experts.py` records its Hessian recipe in `work/hessian.json` and does not resume a work folder
  encoded with another one.
