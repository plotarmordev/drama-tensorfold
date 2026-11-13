### Added

- [`results/exl3-spark/`](results/exl3-spark/README.md): the winning smaller build (`trim-bf16-exl3-3.0`) runs on
  the Spark end to end. `stage.sh` assembles the checkpoint on BOTTOM from the pilot's own 5,900-expert model and
  `exl3-done` layers (nothing copied from the Mac but the `eq-3ref` voice, through the `put` door);
  `prebuild-spark.sh` builds the engine binary before any window; `gb10-exl3-speak.sh` speaks card 37's texts in
  both voices in one window and reports load time, peak memory and seconds per take against `t5900-cover` on the
  same GB10, with the expected window length stated before it starts.
- [`engine/nvfp4/spark/`](engine/nvfp4/spark/): TensorFold main `132bd12383ba` pinned for the Spark, plus the
  patch that loads EXL3 3.0 mul1 routed experts, the compact `drama_head_rows` lm_head and the BF16 codec in the
  drama3 CUDA engine.
- [`results/exl3-spark/summary.json`](results/exl3-spark/summary.json): the GB10 window's numbers. The winning
  build loads in 10 s (`t5900-cover` 20.1 s) and peaks at 16.14 GiB (16.93 GiB); built-in speech runs RTF p50 0.84
  (0.615), reference-voice speech 4.64 (0.828), a fixed 36 s per take. `http_voice.py` rewrites the fork's
  `eq-3ref` voice in the `drama3-http-voice/1` format the pinned main engine loads.
