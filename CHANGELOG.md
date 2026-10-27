# Changelog

## Unreleased

### Fixed

- CUDA NVFP4 long-form splitting preserves paired delivery tags, bracketed cues and special tokens,
  including topic-shift handling, and recognizes CJK sentence/clause punctuation. An indivisible
  protected span may exceed the character target; existing prompt and frame limits still apply.
- Metal reference-prefill reuse now invalidates before incompatible/default prefill, compares the
  complete invariant prefix plus tower lookahead, preserves original chunk boundaries, and reuses
  one GPU-owned snapshot buffer pair per lane instead of allocating on every voice switch.
- The macOS pilot defaults to loopback, uses one conservative free+inactive memory snapshot,
  validates admission controls, clears reused readiness under the shared lock, and settles its
  owned process group before reaping its leader. The server inherits the lock; unexpected clean
  exits and observer/cleanup failures are reported as failures, not successful completion.
- Voice catalog creation requires explicit synthetic/self/consented provenance and enforces the
  pinned server's name, UTF-8, transcript, reference-count, codec-row and size limits.
- Reference-cache regression checks now compare PCM across voice/default/text transitions.
  Corpus and streaming identity checks accept an explicit catalog voice.


### Added

- A checksummed CUDA NVFP4 engine patch series without changing its pinned upstream or weights.
  Model-free build tests exercise real patch application, selection on Linux/macOS, corruption
  refusal, repeat builds and preservation of modified checkouts. Manifests must cover each
  selected patch exactly once; extra, missing and duplicate entries fail before checkout.
- Native Apple-silicon support for the pinned NVFP4-pruned checkpoint: packed experts, compact
  original-ID routing and reference-conditioned semantic CFG, with independent backbone states.
  The Mac engine uses `engine/nvfp4/macos/UPSTREAM` and its checksum-verified patches; the existing
  full-model and CUDA engine pins are unchanged. [Mac qualification](results/macos-nvfp4.json)
  records actual speech, streaming and guidance checks, including the initially failed stress fixture.
- `--weights nvfp4|full` on `scripts/build-engine.sh`, `recipe/pilot.py` and `recipe/pilot_macos.py`.
  Selection verifies the pinned checkpoint on every start, including explicit checkpoint overrides,
  and downloads only an absent default directory. Linux verifies before its GPU lock; Mac verifies
  under its cooperative lock, before model spawn. NVFP4 defaults to `--cfg-gamma 1.5 --cfg-batched`
  unless an explicit guidance flag follows `--`; legacy explicit-path mode retains operator-owned provenance.
- Every speed or quality claim in a pull request carries before/after evidence from the recipe's bench and
  qualification scripts, measured against the recipe baseline (`CONTRIBUTING.md`, pull request template).
- A speed claim also carries Grill v0.11.0 speech evidence: A, B and A2 captures of the recipe server and a
  `grill-perf media compare` that passes on first-audio time and duration ratio (`CONTRIBUTING.md`, pull request
  template). Quality claims keep the WER and speaker-cosine evidence, which the Grill doesn't measure yet.
