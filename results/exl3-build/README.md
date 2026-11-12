# The EXL3 3.0 build of the 5,900-expert quant

Card 39 found exllamav3 `bd1f519`'s EXL3 at 3.0 bits at 0.15× the `trim-bf16-int3` build's Hessian error on six
experts ([`../exl3-ldlq`](../exl3-ldlq/README.md)). This build encodes every kept routed expert of `t5900-cover`
the same way, for `trim-bf16-exl3-3.0` (13.51 GiB planned, against int3's 14.50 GiB), to be measured next to int3
with [`../smaller-builds`](../smaller-builds/README.md)' protocol and pick rule.

**Status: blocked before the GPU run; no EXL3 build or measurements yet.** On 12 November 2026,
`ssh top-menu deepseek status` reported the window closed at 06:31 and BOTTOM serving DeepSeek.
A read-only `bottom shell` check found neither `exl3-rows-v3/` nor `exl3-done/` in the pilot folder.
No GPU job was started, DeepSeek was not paused, and no assets were uploaded. The Desk must stage the rows and
open the 3.5-hour window below; the existing int3 result remains unchanged, not a verdict against EXL3.

## The GPU job: ~3.5 hours on one GB10

[`gb10-exl3.sh`](gb10-exl3.sh) runs [`encode.py`](encode.py) on BOTTOM: 5,900 experts × gate, up and down = 17,700
projections at card 39's 0.6 s each (FP8 load and float64 scoring included) is 2.95 h, plus the 40 layers' rows
(500 MB each) and their Hessians on the GPU; **ask the Desk for a 3.5-hour window**.

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
  `exl3-done/` before starting the next window. `recipe.json` binds the rows manifest, checkpoint config/index,
  kept expert set, encoder, quantizer and hash-checking setup; a different recipe cannot reuse these layers.
- [`kept.json`](kept.json) lists the 5,900 kept experts (from the published quant's index, checked by
  `test_encode.py` on the Mac); [`rows.sha256`](rows.sha256) the rows' files, which the job checks first.

Before it starts, the Desk stages the rows on BOTTOM (19 GB; the job has no network):

```sh
rsync -a /Users/Shared/agent-work/drama-assets/exl3-calibration-rows-v3/ \
  BOTTOM:/home/noideahowthisworks/Documents/drama3-pilot-20261014/exl3-rows-v3/
results/smaller-builds/gb10-start.sh <sha> results/exl3-build/gb10-exl3.sh   # /work/out/exl3/layer-LL.*
```

## After the job, on the Mac

1. Splice the 40 layers into `trim-bf16` as `splice_int3.py` does for int3.
2. Load them in the Mac engine: no patch in `engine/` reads EXL3 routed experts yet. TensorFold has a Metal EXL3
   projection (`zig/kernels/metal/core/exl3_projection.metal`) for other models; Drama's expert path needs it
   wired in and checked against exllamav3's own decode of the same trellis.
3. Speak and score the 24 texts, salts 0 and 1, both voices, with `../smaller-builds/measure.py`, add the build to
   `builds.json` and name the winner with `pick.py`; add it to the listening page next to int3 and `t5900`, with
   built-in `medium-05` (int3's runaway).

## Verification before scheduling

- Card 39's vendored source and extension match the SHA-256 values in `exllamav3-env.sh`.
- All 123 local calibration files pass `rows.sha256`; they are not staged on BOTTOM.
- CPU checkpoint smoke exercises reuse of a finished layer, completion of an interrupted layer, and rejection
  of a missing committed shard. Regression cases also reject changed shard bytes, changed expert sets and
  missing projection records. These checks do not claim a CUDA encode or Metal decode.
- The Mac runtime remains a required integration, not an available EXL3 engine. No `builds.json` entry,
  winner, measured size or listening sample has been invented for the unbuilt model.
- `nice -n 10 scripts/test.sh`: passed (91 recipe tests, two skipped; eight smaller-build tests; four LDLQ
  tests; six full-build tests; streaming and GPU-harness self-tests). Job shell syntax checks passed.
  Input-identity smoke rejects mismatched calibration/checkpoint config before creating output.
