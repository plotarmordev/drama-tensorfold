### Fixed

- The built-in Drama 3 voice no longer runs on to the 1024-frame cap or stops short of the text. Three engine
  changes in Metal patch 0008 ([`engine/nvfp4/macos/patches`](engine/nvfp4/macos/patches/SHA256SUMS)): the
  main-token sampler keeps the eos slot in every nucleus — under a silence-dominant distribution (one token near
  the whole mass, eos second at 2–22 % for ~700 frames) both truncated draws excluded eos, so a stop the model
  asked for could never be drawn; serve fills an unset silence guard with Drama's default (3.5 s below −40 dBFS),
  ending the trailing quiet the same collapse produces; and serve redraws a buffered take the guard silenced, up
  to twice, with the next prompt-derived seed — on the int3 build the collapse can come after a sentence, a
  per-seed degeneracy (5 of 8 `medium-05` seeds speak the whole text and stop). Measured on the Mac int3 build
  (24 texts, salts 0/1, both voices, per-frame logit traces in
  [`results/smaller-builds/trace-20261121`](results/smaller-builds/README.md)): built-in failed takes go 4 → 1
  of 48 and WER 5.78 % → 1.46 % (`medium-05` on both salts and `short-01` s0 now speak the whole text and end on
  eos), no take reaches the cap, the 44 untouched built-in takes stay byte-identical, and the reference voice
  holds 0 failed and WER 1.78 % / 0.97 % (47 of 48 byte-identical; one ends a frame early by the exemption). The
  remaining failure, `long-01` s1, is a 3.1 s quiet run inside speech whose audio is byte-identical before and
  after. `--silence-redraws 0` keeps the first take. Issue #32's full-model caps are consistent with the same
  mechanism, untraced there: hypothesis, with the same changes porting if confirmed.
