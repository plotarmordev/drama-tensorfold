### Changed

- `recipe/exl3_experts.py --workers N` (default 8) converts N layers at once, one process per layer writing only
  that layer's work folder; it resumes from the existing `work/` experts. The shards, index and
  `exl3-provenance.json` are written once, after every worker is done, and not at all if a layer fails. A layer
  from the parallel run is byte-identical to the single-process one.
  Measured on 11 Nov: 58 experts/h single-process, 78/h with `--workers 12` on a Mac other jobs held at load
  average 300–490. An expert's record is written through a temporary file, so a killed run never leaves one
  that reads as done; `--workers` below 1 and `--layers` outside 0–39 are rejected up front.
