### Fixed

- The built-in Drama 3 voice no longer runs on to the 1024-frame cap when the model wants to stop. Two engine
  changes in Metal patch 0008 ([`engine/nvfp4/macos/patches`](engine/nvfp4/macos/patches/SHA256SUMS)): the
  main-token sampler keeps the eos slot in every nucleus — under a silence-dominant distribution (one token near
  the whole mass, eos second at 2–22 % for ~700 frames) both truncated draws excluded eos, so a stop the model
  asked for could never be drawn — and serve fills an unset silence guard with Drama's default (3.5 s below
  −40 dBFS), ending the trailing quiet or murmur the same collapse produces. Measured on the Mac int3 build (24
  texts, salts 0/1, both voices, per-frame traces in
  [`results/smaller-builds/trace-20261121`](results/smaller-builds/README.md)): `medium-05` ends naturally on
  both salts instead of capping at 47.55 s — with the sampler fix alone it draws eos at frames 357/350 — no take
  reaches the cap, all 44 previously passing built-in takes and 47 of 48 reference takes stay byte-identical (one
  ends a frame early by the exemption), and the reference voice holds 0 failed and WER 1.78 % / 0.97 %. Built-in
  failed takes stay 4 of 48: the traces show the rest is outside stopping — `medium-05` speaks only its first
  sentence (WER 0.68) on both salts, `short-01` s0 the same (0.71), and `long-01` s1 stalls 3.1 s mid-speech (a
  quiet-run failure whose audio is byte-identical before and after). Issue #32's full-model caps are consistent
  with the same mechanism, untraced there: hypothesis, with the same two changes porting if confirmed.
