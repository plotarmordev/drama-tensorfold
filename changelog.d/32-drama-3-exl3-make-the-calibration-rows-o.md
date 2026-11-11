### Changed

- `recipe/calibrate_experts.py`'s `report.json` gives, for each kept expert without rows, its best rank among the
  router's 256 logits over its layer's rows (0 is the top): the 12 in layer 36 never place above 171st. It records
  the Hessian recipe `recipe/exl3_experts.py` uses (one shared definition; `down_proj` from the original gate and
  up, an expert with no rows on all of its layer's rows), and fails when the `err_h` ratio to the NVFP4 build
  leaves 1 ± 0.05.
