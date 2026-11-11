### Changed

- The Mac NVFP4 engine pins TensorFold main at `fb8fdf3d` (10 Nov 2026), up from `3d219d21` (24 Oct), so the
  Mac builds on the same engine as everyone else. Main already carries what Metal patches 0001–0006 added
  (packed NVFP4 experts and CFG, request sampling, dialogue speakers), so those are gone; the other five are
  rebased and renumbered: 0001 GB10 voice files, 0002 a take with no audio is a 500, 0003 compact lm_head,
  0004 BF16 codec, 0005 int3 routed experts. Patch 0002 now edits `speech_stream.zig`, where main moved the
  streamed reply. The CUDA NVFP4 pin and the full-model pin are unchanged.
