# The EXL3 3.0 build of the 5,900-expert quant

Card 39 found exllamav3 `bd1f519`'s EXL3 at 3.0 bits at 0.15× the `trim-bf16-int3` build's Hessian error on six
experts ([`../exl3-ldlq`](../exl3-ldlq/README.md)). This build encodes every kept routed expert of `t5900-cover`
the same way, as `trim-bf16-exl3-3.0`, and measures it next to int3 with
[`../smaller-builds`](../smaller-builds/README.md)' protocol and pick rule.

## Result: `trim-bf16-int3` stays the winner

`trim-bf16-exl3-3.0` is **13.51 GiB** (weights + codec, 0.99 GiB under int3's 14.50) and fails all 96 takes:
every one of the 24 texts, salts 0 and 1, in both voices runs to the 1024-frame cap, and Whisper hears filler
("Thank you for watching", repeated clicks) instead of the text: mean WER 645 % reference, 397 % built-in, against
int3's 1.78 % and 5.78 %. `pick.py` over [`builds.json`](../smaller-builds/builds.json) names `trim-bf16-int3`.
The listening page (`/Users/Shared/agent-work/drama-assets/drama3-smaller-builds/listening.html`, on the Mac, not
published) has the EXL3 build next to int3 and `t5900`, including built-in `medium-05`.

Why it fails is not settled. What was checked:

- The encode matches card 39: its six probe experts' errors in the job's records are within 2 % of card 39's.
- The stored tensors decode correctly: a numpy port of exllamav3's unpack, tile permutation, mul1 codebook and
  Hadamard/`suh`/`svh` rebuilds real projections at cosine 0.97–0.99 against the originals (layers 5, 38, 39). A
  read of patch 0006 against exllamav3's sources found the same conventions on every axis.
- The error grows with depth. Per projection, `tr((W−Q)H(W−Q)ᵀ)/tr(WHWᵀ)` has median 0.015–0.02 in layers 1–17,
  0.5 by layer 34 and above 1 in layers 38–39, where 107/162 and 114/225 projections are worse than zero weights;
  outliers reach 729 (layer 8 expert 172) and 3.1×10⁵ (layer 39 expert 108). Int3 has no matching per-layer record
  to compare. Whether the deep-layer error alone breaks the model, or the Metal kernel also differs on real data,
  needs one check: Metal's projection of a real expert against the reference decode.

The v3 rows were staged on BOTTOM and passed `sha256sum -c` against [`rows.sha256`](rows.sha256). BOTTOM's older
`eq/runs/fp8-calib-rows` capture is **not admitted** (Desk decision, 12 November; its
[saved report](bottom-rows-preflight.json) misses layer 36 experts 188 and 243 beyond the twelve approved exceptions).

## The GPU job: ~3.5 hours on one GB10 planned, 1.78 hours run

[`gb10-exl3.sh`](gb10-exl3.sh) runs [`encode.py`](encode.py) on BOTTOM: 5,900 experts × gate, up and down = 17,700
projections at card 39's 0.6 s each (FP8 load and float64 scoring included) is 2.95 h, plus the 40 layers' rows
(500 MB each) and their Hessians on the GPU: **allow 0.1 h for row hashing, then a 3.5-hour GPU window**, opened
by the Desk once the rows are staged.

The local Metal operator check (synthetic routed projections, no model conversion) is budgeted at **0.02 GPU-hours**
on the Mac before it starts. It does not request or open a Spark window and does not replace full-build measurement.

- W is each projection's FP8 original from the pilot checkpoint; H its expert's Hessians from the NVFP4-mix rows
  (`exl3-calibration-rows-v3`, `recipe/exl3_experts.py`'s recipe; the twelve never-routed layer-36 experts take
  their layer's rows), built in float32 on the GPU, where float64 on the CPU would take hours a layer.
- Each projection goes through `exllamav3_job.quantize`, card 39's call (mul1 codebook, automatic output scales,
  seed 0, scales refit), and keeps exllamav3's `trellis`, `suh`, `svh` and `mul1` under the projection's name.
- Each layer is written atomically as `layer-LL.safetensors`, then `layer-LL.json` (rows, seconds, shard SHA-256
  and each projection's error per expert). A valid pair is done: **a closed window resumes finished layers.**
  The unfinished layer is recomputed, not the finished layers. Resume rejects missing/corrupt committed shards
  and mismatched expert/projection records rather than silently skipping them.
  Copy the stopped job's `recipe.json` and finished pairs from `/work/out/exl3` into the pilot folder's
  `exl3-done/` before starting the next window. `recipe.json` binds every actual row file's bytes, checkpoint
  config/index, kept expert set, encoder, quantizer and hash-checking setup; a different recipe cannot reuse layers.
- [`kept.json`](kept.json) lists the 5,900 kept experts (from the published quant's index, checked by
  `test_encode.py` on the Mac); [`rows.sha256`](rows.sha256) pins the staged rows, which the launcher checks
  before loading CUDA.

Once the rows are staged and the Desk opens the window:

```sh
results/smaller-builds/gb10-start.sh <sha> results/exl3-build/gb10-exl3.sh   # /work/out/exl3/layer-LL.*
```

Job `bottom-ccd34101e181` ran `gb10-exl3.sh` at `398b1a0` on 13 November 08:37–10:24 HKT and exited 0: 40 layers,
5,900 experts, 1.78 hours of encoding ([`encode.log`](encode.log); [`exl3.sha256`](exl3.sha256) has its 40 shards).
Its pairs and `recipe.json` are in BOTTOM's pilot `exl3-done/`. `job fetch` stops at 1 GiB, so the shards came to
the Mac as base64 through `bottom shell` in 12-character lines (longer lines get secret-like runs redacted), each
checked against `exl3.sha256`.

## On the Mac

1. `python3 results/exl3-build/assemble.py TRIM_BF16 ENCODED_LAYERS NEW_CHECKPOINT` requires every saved
   layer/record pair, validates the EXL3 3.0 tensor geometry and mul1 marker, preserves dense tensors, codec and
   pruning/head metadata, removes the old routed-expert tensors, and writes the new index last. On this job's
   layers it printed 5,900 experts, 13.5106 GiB.
2. `scripts/build-engine.sh --weights nvfp4` applies patch `0006-exl3-routed-experts.patch`, which admits mul1
   EXL3 3.0 expert groups and runs gate/up, SwiGLU and down through core's indexed Metal projection.
3. `../smaller-builds/measure.py` speak (both voices, salts 0 and 1), score, report and listen, as card 37.

## Verification

- Card 39's vendored source and extension match the SHA-256 values in `exllamav3-env.sh`.
- All 123 local calibration files pass `rows.sha256`; the staged copy on BOTTOM passed the same check.
- CPU checkpoint smoke exercises reuse of a finished layer, completion of an interrupted layer, and rejection
  of a missing committed shard. Regression cases also reject changed shard bytes, changed expert sets and
  missing projection records. These checks do not claim a CUDA encode.
- On the pinned Mac engine plus its patch series, `zig build drama3-metal-quant-check test-drama-metal-quant-host
  native -Dcpu=apple_m1 -j4` passed (85 build steps, seven host tests). The actual `drama3-metal-quant-check`
  executable passed NVFP4, int3 and EXL3 checks for decode and three-row prompt shapes. EXL3 checked 16,384
  values each for gate, up and SwiGLU and 65,536 down values against scalar trellis decode; maximum absolute
  difference was zero. This synthetic gate is not an exllamav3-produced full-model speech measurement.
- The engine at `fb8fdf3` + 6 patches built (`tensorfold-native` `727b6a8a…`), served the assembled checkpoint and
  spoke all 96 takes; `measure.py report` rewrote `builds.json` with the earlier builds' figures unchanged except
  `t5900`/`trim`/`trim-bf16` built-in WER mean 0.06911 → 0.0691 (rounding of the same scores).
- `nice -n 10 scripts/test.sh`: passed (91 recipe tests, two skipped; nine smaller-build tests, one checking
  `builds.json`'s winner against `pick.py`; four LDLQ tests; ten full-build tests; streaming and GPU-harness
  self-tests). Job shell syntax checks passed.
  Input-identity smoke rejects mismatched calibration/checkpoint config before creating output.
- Changed-row resume regression failed before the fix (no rejection) and passes with actual-row hashing.
- `talia health` after the CPU work reported gateway active, zero failed units and DeepSeek listed; its iMessage
  bridge separately reported down. No DeepSeek pause/resume or service change was made.
