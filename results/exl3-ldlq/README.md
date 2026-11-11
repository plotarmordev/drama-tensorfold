# EXL3 with exllamav3's LDLQ rounding, against NVFP4 and int3

Card 35 found EXL3 at 3.0 bits 63× worse than the published NVFP4: TensorFold's `tf-exl3-encode` weighed each tile by
H's diagonal only, with no LDLQ feedback. [`ldlq.patch`](ldlq.patch) makes it round the way exllamav3's quantizer
does, and on card 35's six experts EXL3 then beats the `trim-bf16-int3` build ([`../smaller-builds`](../smaller-builds/README.md))
at both bitrates.

## Result

`trace((W − Q) H (W − Q)ᵀ)` summed over gate, up and down of the six experts (layers 0 and 2: fewest, median, most
rows), W the FP8 original, H each expert's Hessians from the NVFP4-mix rows (`exl3-calibration-rows-v3`), all four
quants on the same H ([`probes.json`](probes.json)):

| quant | bytes per expert | vs int3 | summed error | vs int3 | vs NVFP4 |
|---|---:|---:|---:|---:|---:|
| NVFP4 (published) | 1,769,484 | 1.29 | 1.129 | 0.19 | 1 |
| int3 (3.5 bits) | 1,376,268 | 1 | 5.975 | 1 | 5.3 |
| **EXL3 3.0, LDLQ** | 1,195,020 | **0.87** | 2.391 | **0.40** | 2.1 |
| EXL3 2.5, LDLQ | 998,412 | 0.73 | 4.919 | 0.82 | 4.4 |

EXL3 over int3, per expert (gate / up / down → the three summed):

| layer/expert | rows | 3.0 bits | 2.5 bits |
|---|---:|---|---|
| 0/237 | 61 | 0.03 / 0.03 / 0.85 → 0.27 | 0.08 / 0.06 / 1.72 → 0.55 |
| 0/16 | 3,802 | 0.34 / 0.34 / 0.66 → 0.62 | 0.69 / 0.71 / 1.35 → 1.28 |
| 0/117 | 11,185 | 0.35 / 0.33 / 0.53 → 0.50 | 0.74 / 0.70 / 1.10 → 1.04 |
| 2/42 | 346 | 0.01 / 0.01 / 0.08 → 0.01 | 0.02 / 0.02 / 0.16 → 0.03 |
| 2/222 | 4,029 | 0.14 / 0.16 / 0.22 → 0.18 | 0.29 / 0.31 / 0.42 → 0.36 |
| 2/5 | 8,357 | 0.21 / 0.21 / 0.38 → 0.27 | 0.41 / 0.41 / 0.70 → 0.52 |

- **3.0 bits beats int3 on every expert and every projection**, at 0.87 of its bytes. 2.5 bits beats it in sum but
  loses on layer 0's two busiest experts, so 3.0 is the build to make.
- The int3 and NVFP4 GPTQ ran on the Spark's own statistics (other text and audio, per-expert shrinkage toward a
  prior), EXL3 on these rows; each is scored here on these rows' H, which favours EXL3 by an amount not measured
  here (no held-out rows). The sparse experts' 0.01–0.03× mostly reflect it. These are proxy errors, not listening
  tests.
- Card 35's old encoder summed 63× NVFP4's error at 3.0 bits over 841 experts; with LDLQ it is 2.1× here. On layer 2
  expert 222, the old encoder gave 48× NVFP4 on `gate_proj` and 13× on `down_proj` with these rows, LDLQ 0.71× and 1.14×.

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

So the feedback is exllamav3's; the search and the scales are not. **Parity is not shown**: exllamav3's quantizer
runs only on CUDA, and no encode of these experts by it exists yet. The check that closes it: exllamav3 `bd1f519`'s
`quantize_exl3` on the same six experts' W and H at 3.0 and 2.5 bits, on the Spark; ours matches when every
projection's `trace((W − Q) H (W − Q)ᵀ)` is within 1.1× of exllamav3's: its returned `proxy_err` (after
`refit_scales`: original basis, undamped H, relative) × `trace(W H Wᵀ)`.

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

All 5,900 kept experts through `recipe/exl3_experts.py --encoder` with the patched encoder, into a new `--output`
(the 841 records in `drama3-exl3-experts/work` came from the old encoder and v2 rows and must not be resumed):

| build | routed experts | weights + codec | vs `trim-bf16-int3` |
|---|---:|---:|---:|
| `trim-bf16-int3` | 8.12 GB | 14.50 GiB | 1 |
| `trim-bf16-exl3-3.0` | 7.05 GB | **13.51 GiB** | 0.93 |

- **Time:** the six experts at both bitrates took 497 s as six parallel processes on this Mac at load average
  180–250, 43 experts/h; at that rate the 5,900 take **5.7 days** for both bitrates. 3.0 alone is somewhat over half
  of that; a quieter machine or more workers shortens it, not measured.
- **Then:** splice the EXL3 shards into `trim-bf16` and speak the 48 takes per voice with `../smaller-builds`'
  harness. No patch in `engine/` loads EXL3 routed experts yet; that loader is the step between the conversion and
  the listening test.
