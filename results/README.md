# Results

> **Legacy:** the speed-stage and qualification sections below measure the old patched engine
> (commit `0f54000`, 59 patches on TensorFold `c7137bb3`), not the current opt-in recipe.
> Each newer result identifies its own pinned source and platform.

Summaries of the measurements and correctness checks behind this recipe. Raw audio and weights are not
published; the JSON files retain the measured results and synthetic-input hashes.

| File | What it holds |
|---|---|
| [`dialogue-stopping-20261030.json`](dialogue-stopping-20261030.json) | GB10 CUDA speaker-tagged stopping diagnostic: full-model frame-cap failures, untagged controls, reference/label variants and NVFP4 comparison; no listening or quality claim |
| [`macos-nvfp4.json`](macos-nvfp4.json) | Native Metal NVFP4/CFG support on M5 Ultra: 28 HTTP checks, paired/separate PCM identity, registered operator gates with the initial fixture failure retained; no speed or quality-equivalence claim |
| [`macos-pr17-repair.json`](macos-pr17-repair.json) | Current full-model Metal cache/lifecycle repair: 48/48 paired corpus captures identical, transition/cancellation/multilane parity, bounded voice-switch RSS; no speed claim |
| [`before-after-02/`](before-after-02/README.md) | Reference against native: latency, RTF, first audio, cold start, frame-cap replies |
| [`speed-stages.json`](speed-stages.json) | Warm RTF and load time through the five speed stages, with the audio-identity gate |
| [`qualification.json`](qualification.json) | Numerical gate status, the reference self-control, the revised worst-row bar, clipped replies and the listening test |
| [`long-text-tail.json`](long-text-tail.json) | The silent tails of replies that reach the 1024-frame cap, on both engines |
| [`backbone-attention-tiles.json`](backbone-attention-tiles.json) | The backbone against the reference before and after patch 0040's 64-key attention tiles, with its speed cost |
| [`stream-silence-cut.json`](stream-silence-cut.json) | A collapsing reply streamed and whole before and after patches 0041–0043, and the default-voice corpus gate |
| [`native-voice-encoder.json`](native-voice-encoder.json) | The native voice encoder's codes, memory and time against the reference encoder's, and clones made from each |
| [`clip-speaker-check.json`](clip-speaker-check.json) | The clip speaker check's calibration on model-invented voices and mixes of them: every case's split and decision |
| [`silence-guard-floor.json`](silence-guard-floor.json) | The silence guard's -40 dBFS floor replayed over renders of the built-in and model-invented voices, and checked on whispered and soft speech |
| [`full-vs-quant-01/`](full-vs-quant-01/README.md) | Full FP8 against the NVFP4 quant, with and without guidance and voice: size, speed, word error rate, failed takes, speaker similarity, and a model-card section |

## Speed stages

Fourteen patches in five stages, each gated on one GB10 host with `recipe/pilot.py`. Every stage kept the audio of
all 46 captured normal requests byte-identical to the pre-speed build, the two capped long-05 replies matched the
reviewed clipped capture, and `bench/stream_check.py` passed 9/9.

| Stage | Adds | Warm RTF |
|---|---|---:|
| pre-speed build | streaming | 1.087 |
| 02 | one-row GEMV ring, backbone glue fusions, tower decode attention (rows == 1 only), radix sampler, clipped replies | 0.741 |
| 03 | direct-read weight stager, codec init thread (load 53.3 s → 6.2 s) | — |
| 04 | tower one-row GEMV reproducing the cuBLASLt summation, checked per projection at init (fail-closed) | 0.678 |
| 05 | backbone programmatic dependent launch (only when both kernel images carry the `drama_pdl` marker) | 0.655 |

On the final binary of the speed work, before patch 0040, the warm RTF is 0.636, first audio arrives at 0.132 s and
a 4.18 s utterance completes in 2.86 s. Stage 01 (the staged backbone GEMM) has no separate RTF entry. Patch 0040
costs about 5% on our GB10, our corpus: [`backbone-attention-tiles.json`](backbone-attention-tiles.json).

## Qualification status: not achieved

| Gate | Bar | Native | Status |
|---|---|---:|---|
| Codec, 1 frame | SNR ≥ 40 dB | 33.086 dB | fail (known limitation) |
| Codec, 2 / 7 / 128 / 1024 frames | SNR ≥ 40 dB, exact prefixes | pass | pass |
| Backbone hidden relative L2 | ≤ 0.05 | 0.0528 | fail |
| Backbone worst-row cosine | ≥ 0.8712518909203973 (revised from 0.995) | 0.9169 | pass |
| Backbone all-row cosine | ≥ 0.999 | 0.99860 | fail |
| Backbone last-row cosine | ≥ 0.999 | 0.99809 | fail |
| Backbone semantic KL | ≤ 0.01 | 0.0062 | pass |
| Backbone semantic top-30 overlap | ≥ 0.9 | 0.933 | pass |
| Backbone semantic top-1, cosine, relative L2 | equal / ≥ 0.999 / ≤ 0.03 | — | pass |
| Tower, 7-row streaming head top-1 | equal | 600 vs 463 | fail (the reference logits tie) |
| Tower real-input and teacher-forced replays | component gates | — | pass |
| long-05 end of speech | reached | never | fail |

- **Exact backbone parity is infeasible.** The reference's own GDN b/a projection gives different bits for the
  same rows at different call shapes: on 3,932,160 outputs, M=1 vs 182 differ in 618 and M=2 vs 182 in 4,696.
- **The reference fails its own worst-row gate.** Co-batched with a second request, which it supports by
  default, the reference's worst-row cosine against its solo output is 0.871–0.982 in every tested view. The
  revised bar is the lowest of those minima, set from reference data alone and registered before native was
  re-scored ([method](../qualification/README.md#how-the-worst-row-bar-was-set)).
- **Native is still outside that envelope.** Relative L2 is 0.0528 against the reference's own maximum of 0.045,
  and KL 0.0062 against 0.0049.
- **The backbone rows are after patch 0040.** Before it they read relative L2 0.124, worst row 0.8006, all-row
  0.99233, last row 0.99670, KL 0.0243 and top-30 0.867, all failing. The codec, tower and long-05 rows were
  measured before it.

## Long text

Both engines can fail to emit an end of speech and pad to the 1024-frame (47.55 s) cap. All six reference
1024-frame replies show the signature of native long-05: speech stops early (reference after 3–6 s, native after
about 11 s), then about −95 dBFS until the cap. Native returns the clipped audio as HTTP 200 with
`X-Finish-Reason: length`; the reference returns it as a plain 200. long-05 is not fixed.

## Listening

In a paired A/B test, blind except for one pair, the maintainer preferred native on 4 of 5 same-text pairs
(short-03, the backbone fixture prompt, long-03, long-05) and the reference on medium-04. The long-05 pair was
not blind: the rater knew native's behaviour on it. There, the reference resumes speech after its silence and
native never does. This is one rater, one take per engine and no pre-registered bar: it changes no numerical
gate.

## Where the numbers come from

Measured results come from the files in this directory. Configuration constants and prerequisites come from the
recipe, and they are limits and budgets, not measurements.

| Number in the top-level README | Kind | Source |
|---|---|---|
| RTF, whole-reply p50, first audio p50, per-length p50 and RTF, cap replies, 72 warm attempts and no failures per engine | measured | [`before-after-02/tables.json`](before-after-02/tables.json): `overall`, `per_bucket.{short,medium,long}` |
| "About 8 s of audio" for a typical clip | measured | `tables.json` `overall.reference_audio_s_p50` 8.54 s, `native_audio_s_p50` 7.87 s |
| Process start to ready, 254 s and 6.4 s | measured, one start each, reused caches | [`before-after-02/startup.json`](before-after-02/startup.json); conditions in [`before-after-02/README.md`](before-after-02/README.md) |
| Speedups (1.48x, 1.62x, 3.7x, 39x, 1.35x/1.62x/1.57x) | derived | reference value / native value from the two files above |
| Character ranges 19–47, 114–144, 283–313 | corpus property | [`../bench/corpus/drama-bench-en-v1.jsonl`](../bench/corpus/drama-bench-en-v1.jsonl) |
| 46 normal replies byte-identical, 14 speed patches, warm RTF 1.087 → 0.636–0.655 | measured (retained summary) | [`speed-stages.json`](speed-stages.json); the patches are in [`../engine/patches/`](../engine/patches/README.md) |
| Accuracy bars and native values before patch 0040 | measured | [`qualification.json`](qualification.json) `qualification_01`, `backbone_worst_row_bar_revision.rescore` |
| Backbone values after patch 0040, voice-prompt values, RTF 0.642–0.650 → 0.682, silence finishes 2 → 0 of 40 | measured | [`backbone-attention-tiles.json`](backbone-attention-tiles.json) `text_fixture_182_tokens`, `voice_prompt_449_rows`, `serving` |
| Silence cut at 16.579 s streamed and whole (before 29.77 s / 17.09 s), 46 normal replies byte-identical, streams 9 of 9, long-05 at 20.2 s | measured | [`stream-silence-cut.json`](stream-silence-cut.json) |
| Native encoder codes differ on 2.46–2.95% of semantic codes, speaker similarity −0.0046 to +0.0061 against a 0.023 take spread | measured | [`native-voice-encoder.json`](native-voice-encoder.json) |
| Long-text cap behaviour | measured | [`long-text-tail.json`](long-text-tail.json) |
| Speaker check threshold 5.6: one voice up to 4.87, full-band two voices from 6.40, 2 of 12 two-voice takes low-passed at 8 kHz kept | measured | [`clip-speaker-check.json`](clip-speaker-check.json) |
| Silence guard at -40 dBFS: 11 of 11 collapses, none of the other 267 of 278 renders, one hum missed at -60 dBFS | measured | [`silence-guard-floor.json`](silence-guard-floor.json) |
| 4 of 5 listening pairs, one not blind | rating | [`qualification.json`](qualification.json) `listening_ab_01` |
| 88 GiB `MemAvailable` at load, 54 GiB cap, 34 GiB abort floor | configuration | [`../recipe/pilot.py`](../recipe/pilot.py) constants; [`../recipe/README.md`](../recipe/README.md) |
| About 39 GiB of disk | prerequisite: checkpoint storage only | `weight_and_codec_bytes` 41474674072 B (38.63 GiB) in [`../recipe/sources.json`](../recipe/sources.json). The engine checkout, build output and runtime caches need more; their size is not measured here. |

This directory holds neutralized summaries, not the original captures (audio bodies, per-request samples and
server logs). `bench/` runs the same comparison on your hardware; your numbers will differ with the host, cache
state and admission conditions. `bench/before-after.sh` also refuses a failed admission check, which the
recorded reference run went past (see the deviations in [`before-after-02/README.md`](before-after-02/README.md)).
