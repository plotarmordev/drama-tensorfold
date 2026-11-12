### Fixed

- [`results/capri-clean-long/`](results/capri-clean-long/README.md): Capri's cleaned long voice
  (`capri-clean-long`, 47.5 s across two references) ended speech early on about 3 in 7 Capri notes: ~0.5 s of
  speech, then 10 s of silence (`X-Finish-Reason: silence`). The fixed catalog keeps the voice name and cuts the
  same cleaned recording to 7.2 s / 156 frames, one reference; it speaks all 20 test lines to `stop`, as
  `shell-nolaugh` does on the same int3 build. The catalog and the per-line results are in the results folder;
  the WAVs stay on the Mac.
