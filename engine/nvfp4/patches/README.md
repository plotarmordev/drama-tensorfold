# CUDA NVFP4 patch series

These native patches apply in filename order to the unchanged
[`../UPSTREAM`](../UPSTREAM) pin. The builder checks that `SHA256SUMS` lists every
selected patch exactly once, verifies its digest, then applies the series.

| Patch | Scope | Source PR |
|---|---|---|
| `0001-protected-speech-chunks.patch` | Protected markup and CJK long-form boundaries; test fixture drain ordering | [TensorFold #5](https://github.com/plotarmordev/TensorFold/pull/5) |
| `0002-speech-sampling-controls.patch` | Request sampling through whole/streamed, reference, CFG and long-form redraw paths | [TensorFold #6](https://github.com/plotarmordev/TensorFold/pull/6) |

Patch 0002 adds [request sampling](../../../recipe/README-sampling.md). Each
long-form segment starts with the supplied seed; its retries add their own
attempt number. Omitted seeds retain the existing per-segment prompt hash.

Protected constructs are complete model special tokens, balanced bracketed
spans and correctly nested, exact-name ASCII tags without attributes. Unclosed
or mismatched markup does not protect unrelated following text. Adjacent
controls remain separate atoms. Segment slices retain their original backing
buffer, and only a whole protected atom may exceed the code-point target.

CPU validation: 364 passed, four skipped; goldens have zero unexpected
differences. Independent adversarial testing exercised 1,680 combinations of
markup, Unicode and character targets. Six existing socket-fixture ordering
repairs let the macOS CPU suite drain replies before joining server threads;
production transport is unchanged. CUDA host compilation is still blocked on
macOS by the base's `O.DIRECT` use. No device/model or listening qualification
is claimed by these CPU checks.

The [public MLX chunker](https://github.com/Sogni-AI/drama3-mlx/blob/ecc235039ccd5f982f66a6982184f57432354ab4/drama3_mlx/chunking.py)
is a behavior reference. This is an independent native implementation under
the recipe's Apache-2.0 license; model weights keep their separate license.
