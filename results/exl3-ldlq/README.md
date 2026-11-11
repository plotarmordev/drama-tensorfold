# EXL3 with exllamav3's LDLQ rounding, against NVFP4 and int3

Card 35 found EXL3 at 3.0 bits 63× worse than the published NVFP4: TensorFold's `tf-exl3-encode` weighed each tile by
H's diagonal only, with no LDLQ feedback. Here exllamav3 `bd1f519`'s own quantizer encodes card 35's six experts on a
GB10, and EXL3 at 3.0 bits has **0.15× the `trim-bf16-int3` build's error at 0.87× its bytes**, under NVFP4's
(0.78×). [`ldlq.patch`](ldlq.patch), our encoder with exllamav3's LDLQ, closes most of card 35's gap but is still
2.7× exllamav3, so the build should use exllamav3 itself.

## Result

`trace((W − Q) H (W − Q)ᵀ)` summed over gate, up and down of the six experts (layers 0 and 2: fewest, median, most
rows), W the FP8 original, H each expert's Hessians from the NVFP4-mix rows (`exl3-calibration-rows-v3`), every quant
on the same H ([`probes.json`](probes.json); exllamav3 in [`exllamav3.json`](exllamav3.json), verdict in
[`parity.json`](parity.json)):

| quant | bytes per expert | vs int3 | summed error | vs int3 | vs NVFP4 |
|---|---:|---:|---:|---:|---:|
| NVFP4 (published) | 1,769,484 | 1.29 | 1.129 | 0.19 | 1 |
| int3 (3.5 bits) | 1,376,268 | 1 | 5.975 | 1 | 5.3 |
| **EXL3 3.0, exllamav3** | 1,195,020 | **0.87** | 0.878 | **0.15** | **0.78** |
| EXL3 2.5, exllamav3 | 998,412 | 0.73 | 1.850 | 0.31 | 1.6 |
| EXL3 3.0, `ldlq.patch` | 1,195,020 | 0.87 | 2.391 | 0.40 | 2.1 |
| EXL3 2.5, `ldlq.patch` | 998,412 | 0.73 | 4.919 | 0.82 | 4.4 |

exllamav3's EXL3 over int3, per expert (gate / up / down → the three summed), and over NVFP4 (summed):

| layer/expert | rows | 3.0 bits | 2.5 bits | 3.0 / 2.5 vs NVFP4 |
|---|---:|---|---|---|
| 0/237 | 61 | 0.01 / 0.01 / 0.13 → 0.04 | 0.01 / 0.01 / 0.28 → 0.09 | 0.24 / 0.50 |
| 0/16 | 3,802 | 0.33 / 0.35 / 0.20 → 0.21 | 0.69 / 0.71 / 0.42 → 0.45 | 1.14 / 2.43 |
| 0/117 | 11,185 | 0.50 / 0.48 / 0.17 → 0.22 | 1.03 / 0.98 / 0.36 → 0.46 | 1.18 / 2.47 |
| 2/42 | 346 | 0.01 / 0.01 / 0.03 → 0.01 | 0.01 / 0.01 / 0.07 → 0.02 | 0.04 / 0.08 |
| 2/222 | 4,029 | 0.07 / 0.08 / 0.11 → 0.09 | 0.15 / 0.16 / 0.22 → 0.19 | 0.46 / 0.95 |
| 2/5 | 8,357 | 0.11 / 0.11 / 0.17 → 0.13 | 0.23 / 0.23 / 0.35 → 0.27 | 0.67 / 1.36 |

- **3.0 bits beats int3 on every expert and every projection**, at 0.87 of its bytes, and NVFP4 in sum at 0.68 of
  its bytes. 2.5 bits beats int3 in sum but loses on layer 0 expert 117's `gate_proj` (1.03), so 3.0 is the build.
- The int3 and NVFP4 GPTQ ran on the Spark's own statistics (other text and audio, per-expert shrinkage toward a
  prior), EXL3 on these rows; each is scored here on these rows' H, which favours EXL3 by an amount not measured
  here (no held-out rows). The sparse experts' 0.01–0.04× mostly reflect it. These are proxy errors, not listening
  tests.

## The encoder change

[`ldlq.patch`](ldlq.patch) applies to TensorFold `6b01e7307` (`zig/src/core/exl3_encode.zig` is the same on main
`630390e6e`). Checked against exllamav3 `bd1f519` (`exllamav3/modules/quant/exl3_lib/quantize.py`):

| step | exllamav3 | `ldlq.patch` |
|---|---|---|
| H damping | `sigma_reg` 0.025 × mean diagonal | the same |
| LDLQ | `block_ldl` + `ldlq`, 16-row blocks last to first | Cholesky form; equal to `block_ldl`'s compensation to 1e-9 (patch test `feedback equals exllamav3's block-LDL compensation`) |
| tile search | exact tail-biting Viterbi (CUDA `quantize_tiles`) | 128-wide beam over the closed ring |
| input scales `su` | random signs × per-row RMS, then a searched global scale (`regularize`, `g_scale_gss`) | random signs only |
| output scales `sv` | random signs × per-column RMS | per-column RMS / 1.2437 |
| after LDLQ | `refit_scales` refits `su`, `sv` against H | none |

So the feedback is exllamav3's; the search and the scales are not, and they cost it **2.7× exllamav3's error** in sum
at both bitrates (1.7–6.5× per expert, worst on the sparse experts; 0.69–6.7× per projection, worst on `down_proj`):
far from the 1.1× per projection that `measure.py parity` asks of a match ([`parity.json`](parity.json)).

[`gb10-exllamav3.sh`](gb10-exllamav3.sh) ran exllamav3 `bd1f519`'s own `quantize_exl3` ([vendored](exllamav3-bd1f519.tar.gz),
its CUDA extension built for sm_121) on the same six experts at 3.0 and 2.5 bits, W rebuilt from the pilot's FP8
checkpoint and checked by sha256 against [`hessians/`](hessians) (`measure.py hessians`: each expert's two Hessians as
float32 upper triangles, which reproduce `probes.json`'s NVFP4 errors to 2e-7). [`exllamav3_job.py`](exllamav3_job.py)
scores exllamav3's returned Q (after `refit_scales`, original basis) on the undamped H in float64; `measure.py parity`
divides ours by it and scores exllamav3 against int3 and NVFP4 on our bytes (its trellis and `su`/`sv` take the same
bytes at each bitrate). It ran on the Spark host in the 12 Nov 04:51 window, since the job image has no C++ compiler
for the extension: building it took 2.5 min, the 36 encodes and their scoring 22 s.

```sh
# on the Spark host, from a checkout of this repo
PILOT=<drama3-pilot folder> OUT=<scratch> results/exl3-ldlq/gb10-exllamav3.sh   # OUT/exl3-ref/parity.json
```

```sh
git -C TENSORFOLD checkout 6b01e7307 && git -C TENSORFOLD am results/exl3-ldlq/ldlq.patch
(cd TENSORFOLD && zig build tf-exl3-encode -Doptimize=ReleaseFast)
python3 results/exl3-ldlq/measure.py probe TENSORFOLD/zig-out/bin/tf-exl3-encode INT3_DIR 2/222 OUT_DIR   # each expert
python3 results/exl3-ldlq/measure.py report OUT_DIR results/exl3-ldlq/probes.json
```

`INT3_DIR` holds the int3 job's `layer-LL.safetensors` ([`int3.sha256`](../smaller-builds/int3.sha256)). The encoder
that made `probes.json` is sha256 `4e987037…47c4ae`, from the patch without its parity test (the test adds no code to
the encoder). The patch's unit tests pass, including LDLQ beating the same search without feedback under a low-rank H,
and its written trellis still reloads and decodes bit for bit.

## Build plan

All 5,900 kept experts at 3.0 bits through exllamav3 `bd1f519`'s `quantize_exl3` on one GB10, as
`exllamav3_job.py` does here:

| build | routed experts | weights + codec | vs `trim-bf16-int3` |
|---|---:|---:|---:|
| `trim-bf16-int3` | 8.12 GB | 14.50 GiB | 1 |
| `trim-bf16-exl3-3.0` | 7.05 GB | **13.51 GiB** | 0.93 |

1. **Hessians** on this Mac: `measure.py hessians`' recipe over every kept expert from `exl3-calibration-rows-v3`,
   ~46 GB of compressed upper triangles (7.8 MB per expert here), staged by the Desk into the Spark's pilot assets;
   the Spark has no network in a job and the repo cannot carry them.
2. **Encode** in one Spark window: `exllamav3_job.py` turned into a writer that keeps exllamav3's trellis, `su` and
   `sv` per projection in `recipe/exl3_experts.py`'s shard names and records each error. At this run's 0.6 s per
   projection (FP8 load and float64 scoring included) the 17,700 projections take **~3 h of GPU**, plus 2.5 min to
   build the extension on the host.
3. **Then:** splice the shards into `trim-bf16` and speak the 48 takes per voice with `../smaller-builds`' harness.
   No patch in `engine/` loads EXL3 routed experts yet; that loader, and checking that it decodes exllamav3's
   trellis as exllamav3 does, is the step between the conversion and the listening test.

Our encoder at Mac speed (43 experts/h with `ldlq.patch`, 5.7 days for both bitrates) is no longer the route: it
costs 2.7× the error and ~20× the time (84 s an expert at both bitrates, against 3.7 s).
