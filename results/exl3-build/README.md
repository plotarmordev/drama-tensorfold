# The EXL3 3.0 build of the 5,900-expert quant

Card 39 found exllamav3 `bd1f519`'s EXL3 at 3.0 bits at 0.15× the `trim-bf16-int3` build's Hessian error on six
experts ([`../exl3-ldlq`](../exl3-ldlq/README.md)). This build encodes every kept routed expert of `t5900-cover`
the same way, for `trim-bf16-exl3-3.0` (13.51 GiB planned, against int3's 14.50 GiB), to be measured next to int3
with [`../smaller-builds`](../smaller-builds/README.md)' protocol and pick rule.

**Status: BOTTOM-local preflight fails card 32's coverage gate; no EXL3 build or measurements yet.**
On 12 November 2026, BOTTOM's existing `eq/runs/fp8-calib-rows` capture was exercised through
[`rows_bottom.py`](rows_bottom.py), reusing card 32's selection (seed 0) and unchanged validation.
The [observed report](bottom-rows-preflight.json) records 174,529 rows: selection is the identity, class shares
match NVFP4 exactly, and the 1,208 checked tensors' err_h ratio is 0.99887–0.99999 (p05–p95).
However, only **5,886/5,900** kept experts have rows. Layer 36 experts **188 and 243** are missing in addition
to the twelve approved exceptions. Their best router ranks are 14 and 15, unlike the approved exceptions'
182–219 here. No exception was added and the failed gate was not bypassed. This is the older CUDA capture,
not a regeneration of card 32's 193-request Metal capture (seeds 1 and 2, Capri references).

The exact Capri source recording and transcript do exist in the pilot, with SHA-256 matching the Mac inputs
(`2d7986b798aca51f695417e6fc51c03d87cfe1814ef9941d1d7e9b2b51a65bd9` and
`28f5be03a3926c2483bbdb783395dbf37381963677da5ff0bb321b02d74a52a0`). The existing CUDA
`tf-drama-probe` exposes salt-based requests, while card 32's `--calibration-rows` implementation is Metal-only;
a same-seed CUDA capture is not implemented by this preflight. The next decision is whether the owner permits
the older capture and two additional fallback experts, or requires a CUDA collector matching card 32 first.
The GPU window was closed at 10:53; no GPU job was launched, DeepSeek was not paused, and no assets were uploaded.

## The GPU job: ~3.5 hours on one GB10

[`gb10-exl3.sh`](gb10-exl3.sh) runs [`encode.py`](encode.py) on BOTTOM: 5,900 experts × gate, up and down = 17,700
projections at card 39's 0.6 s each (FP8 load and float64 scoring included) is 2.95 h, plus the 40 layers' rows
(500 MB each for Mac rows, 715 MB for BOTTOM's capture) and their Hessians on the GPU; **ask the Desk for a
3.5-hour window only after calibration passes**. BOTTOM CPU preflight took about one minute; allow 0.1 h for
preflight and row hashing, then 3.5 GPU-hours for encoding. This is not a runtime estimate for a new capture.

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
  `test_encode.py` on the Mac); [`rows.sha256`](rows.sha256) pins only the staged Mac rows.

The launcher checks staged Mac rows against `rows.sha256` when present. Otherwise it runs `rows_bottom.py`
against the read-only pilot capture, writes `/work/out/rows-report.json`, and fails before loading CUDA if the
unchanged card 32 gates fail. Its selection seed is 0; an identity selection reads the capture in place instead
of copying 29 GB. The current BOTTOM capture is **not admitted**, for the coverage reason above.

After the calibration prerequisite is resolved and the Desk opens the window:

```sh
results/smaller-builds/gb10-start.sh <sha> results/exl3-build/gb10-exl3.sh   # /work/out/exl3/layer-LL.*
```

`job fetch <id>` returns `/work/out` as a tar stream. Preserve completed pairs and `recipe.json` in the host's
pilot `exl3-done/` before a new job; `/work` is container-local, not a host directory. The launcher copies those
pairs into the new job, and the encoder validates the recipe and checksums before skipping finished layers.

## After the job, on the Mac

1. Run `python3 results/exl3-build/assemble.py TRIM_BF16 ENCODED_LAYERS NEW_CHECKPOINT`. It requires every saved
   layer/record pair, validates the EXL3 3.0 tensor geometry and mul1 marker, preserves dense tensors, codec and
   pruning/head metadata, removes the old routed-expert tensors, and writes the new index last. Its printed GiB
   is weights plus codec; the planned full-build size above is not a measurement.
2. Build the Mac engine with `scripts/build-engine.sh --weights nvfp4`. Patch `0006-exl3-routed-experts.patch`
   admits mul1 EXL3 3.0 expert groups, preserves compact router slots, and runs gate/up, SwiGLU and down through
   core's indexed Metal projection. The full checkpoint still needs an end-to-end speech run after encoding.
3. Speak and score the 24 texts, salts 0 and 1, both voices, with `../smaller-builds/measure.py`, add the build to
   `builds.json` and name the winner with `pick.py`; add it to the listening page next to int3 and `t5900`, with
   built-in `medium-05` (int3's runaway).

## Verification before scheduling

- Card 39's vendored source and extension match the SHA-256 values in `exllamav3-env.sh`.
- All 123 local calibration files pass `rows.sha256`; they are not staged on BOTTOM.
- CPU checkpoint smoke exercises reuse of a finished layer, completion of an interrupted layer, and rejection
  of a missing committed shard. Regression cases also reject changed shard bytes, changed expert sets and
  missing projection records. These checks do not claim a CUDA encode.
- On the pinned Mac engine plus its patch series, `zig build drama3-metal-quant-check test-drama-metal-quant-host
  native -Dcpu=apple_m1 -j4` passed (85 build steps, seven host tests). The actual `drama3-metal-quant-check`
  executable passed NVFP4, int3 and EXL3 checks for decode and three-row prompt shapes. EXL3 checked 16,384
  values each for gate, up and SwiGLU and 65,536 down values against scalar trellis decode; maximum absolute
  difference was zero. This synthetic gate is not an exllamav3-produced full-model speech measurement.
- No `builds.json` entry, winner, measured size or listening sample has been invented for the unbuilt model.
- `nice -n 10 scripts/test.sh`: passed (91 recipe tests, two skipped; eight smaller-build tests; four LDLQ
  tests; nine full-build tests; streaming and GPU-harness self-tests). Job shell syntax checks passed.
  Input-identity smoke rejects mismatched calibration/checkpoint config before creating output.
- The BOTTOM CPU smoke ran the real `rows_bottom.prepare` over all 40 layers, saved the report, and rejected
  exactly layer 36 experts 188 and 243. No coverage/statistics tolerance changed.
- Changed-row resume regression failed before the fix (no rejection) and passes with actual-row hashing.
- `talia health` after the CPU work reported gateway active, zero failed units and DeepSeek listed; its iMessage
  bridge separately reported down. No DeepSeek pause/resume or service change was made.
