# The EXL3 3.0 build of the 5,900-expert quant

Card 39 found exllamav3 `bd1f519`'s EXL3 at 3.0 bits at 0.15× the `trim-bf16-int3` build's Hessian error on six
experts ([`../exl3-ldlq`](../exl3-ldlq/README.md)). This build encodes every kept routed expert of `t5900-cover`
the same way, as `trim-bf16-exl3-3.0`, and measures it next to int3 with
[`../smaller-builds`](../smaller-builds/README.md)' protocol and pick rule.

## Result: `trim-bf16-exl3-3.0` is the winner

`trim-bf16-exl3-3.0` is **13.51 GiB** (weights + codec), 0.99 GiB under int3's 14.50. Over the 24 texts, salts 0
and 1, both voices, its mean WER is 1.61 % reference and 9.47 % built-in (int3: 1.78 % and 5.78 %; `t5900`:
1.50 % and 6.91 %), with no failed reference take and 6 failed built-in takes, `t5900`'s count. `pick.py` over
[`builds.json`](../smaller-builds/builds.json) names it; int3 also qualifies but is larger. It keeps int3's
built-in `medium-05` runaway on both salts and adds `medium-02` s1
([`../smaller-builds`](../smaller-builds/README.md#result-trim-bf16-exl3-30)). The listening page
(`/Users/Shared/agent-work/drama-assets/drama3-smaller-builds/listening.html`, on the Mac, not published) has it
next to int3 and `t5900`, including built-in `medium-05`.

A first measurement had every EXL3 take running to the 1024-frame cap. The cause was patch 0006, not the encode:
its `forward.zig` hunk dropped the per-frame `rmsnorm` into `e.hlast`, so each decode frame's candidate head read
the prompt-final state. With it restored, the int3 control (built-in, salt 0) on the fixed engine is byte-identical
to the recorded int3 takes (24/24 WAVs), so the earlier builds' figures stand; the 96 EXL3 takes were spoken again.

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
  `exl3-done/` before starting the next window. `recipe.json` binds every actual row file's bytes, the FP8 shard
  bytes, checkpoint config/index, kept expert set, encoder, quantizer and hash-checking setup; a different recipe
  cannot reuse layers. Job `bottom-ccd34101e181`'s recipe predates the shard binding, so a re-encode starts fresh.
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
- On the pinned Mac engine plus its patch series, `zig build test-drama-metal-quant-host` runs nine host tests,
  including both EXL3 admission tests on scalar `mul1` markers (a `[1]` marker is refused); with the old `[1]`
  fixture the admission test fails. `drama3-metal-quant-check` passed NVFP4, int3 and EXL3 checks for decode and
  three-row prompt shapes, EXL3 at zero maximum difference against scalar trellis decode; it does not run the frame
  loop, which the int3 control above covers.
- The engine at `fb8fdf3` + 6 patches (`tensorfold-native` `e4b7f35a…`) served the assembled checkpoint and spoke
  all 96 takes; `measure.py report` rewrote `builds.json` with the earlier builds' figures unchanged except
  `t5900`/`trim`/`trim-bf16` built-in WER mean 0.06911 → 0.0691 (rounding of the same scores).
- `nice -n 10 scripts/test.sh`: passed (91 recipe tests, two skipped; nine smaller-build tests, one checking
  `builds.json`'s winner against `pick.py`; four LDLQ tests; ten full-build tests; streaming and GPU-harness
  self-tests). Job shell syntax checks passed.
  Input-identity smoke rejects mismatched calibration/checkpoint config before creating output.
- Changed-row and changed-FP8-shard resume regressions each failed before their fix (no rejection) and pass now.
- `talia health` after the CPU work reported gateway active, zero failed units and DeepSeek listed; its iMessage
  bridge separately reported down. No DeepSeek pause/resume or service change was made.
