### Changed

- `recipe/exl3_experts.py --workers N` (default 8) converts N layers at once, one process per layer writing only
  that layer's work folder; it resumes from the existing `work/` experts. The shards, index and
  `exl3-provenance.json` are written once, after every worker is done, and not at all if a layer fails. A layer
  from the parallel run is byte-identical to the single-process one.
