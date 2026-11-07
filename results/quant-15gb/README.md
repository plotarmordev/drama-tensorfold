# The 5,900-expert quant against the released quant and the full model

Drama 3 NVFP4 with 5,900 of the 10,240 routed experts kept (`t5900-cover`, manifest
[`recipe/checksums/nvfp4-t5900cover.sha256`](../../recipe/checksums/nvfp4-t5900cover.sha256)), published as
[`plotarmordev/drama-3-NVFP4-Pruned-5900`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900). Measured
on one GB10 host with the protocol of [`full-vs-quant-01`](../full-vs-quant-01/README.md): the 24 English texts of
the benchmark corpus one request at a time, 4 seed salts, a reference voice with guidance γ 1.5 batched (arm Q) and
the built-in voice (arm QB, guidance inactive).

| Column | Checkpoint |
|---|---|
| `v0-release`, `release` | the released 20.8 GB quant, `plotarmordev/drama-3-NVFP4-Pruned` (7,369 experts); `release` is run-01's Q and QB arms |
| `t5900-cover` | this quant |
| `v2-trim5892`, `t5400-cover`, `t6400-cover` | unpublished candidates, measured in the same sweep |

| File | What it holds |
|---|---|
| [`compare.md`](compare.md), [`compare.json`](compare.json) | Per-variant size, memory, load, speed, word error rate, failed takes, speaker cosine, UTMOS, length and pitch against the full model, and each variant paired with the release |
| [`null-control.md`](null-control.md), [`null-control.json`](null-control.json) | The release at salts 4–7 against itself at 0–3, the noise floor of the paired comparisons, and every variant against the release pooled over 8 salts |
| [`audio/t5900-cover.json`](audio/t5900-cover.json) | Per-take UTMOS, audio length, pitch and spectral measures of this quant and the full model (F, FB) |

The full model's numbers are run-01's (`results/full-vs-quant-01/tables.json`). These files are copies of the
GPU host's; nothing was re-run.
