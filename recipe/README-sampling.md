# Per-request speech sampling

All three pinned engine profiles accept `temperature`, `top_p`, `top_k` and `seed`
on `POST /v1/audio/speech`, for whole and streamed responses. For example:

```json
{
  "model": "fishaudio/drama-3",
  "input": "The train leaves at seven fifteen.",
  "response_format": "pcm",
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 30,
  "seed": 42
}
```

| Field | Accepted values | Omitted or `null` |
|---|---|---|
| `temperature` | Finite nonnegative number representable by the native `f32` sampler | `1.0` |
| `top_p` | Finite number in `(0, 1]` that remains positive as `f32` | `0.9` |
| `top_k` | JSON integer from 1 through 64 | `30` |
| `seed` | JSON integer from 0 through 18446744073709551615 | Existing prompt-derived seed |

Booleans, numeric strings, fractional integer fields and out-of-range values are
refused before submitting to the engine. Seeds are parsed as integers without a
floating-point conversion. Clients that cannot represent all 64-bit integers
exactly should use a smaller seed, such as the example above.

Floating fields are validated after the existing JSON parser converts them to
`f64`. Extremely small decimals such as `1e-400` and `-1e-400` become zero there:
both request temperature 0, while both are refused for `top_p`. A positive
parsed value such as `1e-100` is refused when narrowing to `f32` would make it zero.

An explicit seed is used unchanged for the first take. Redraws add the attempt
number with unsigned 64-bit wraparound. The selected sampling parameters survive
reference-prompt rebuilding and silence/register redraws. Without an explicit
seed, prompt-derived seeding retains its existing behavior.

Legacy CUDA long-form requests apply that policy separately to each segment:
each segment starts with the supplied seed, and retries within that segment add
their attempt number. Omitted seeds continue to hash each segment's own prompt.

The HTTP limit of 64 for `top_k` is shared across the CUDA and Metal profiles.
Sampling still uses each engine's native algorithm, including its existing
repetition-aware behavior. Equal settings do not promise equal audio across
backends, checkpoints or engine versions. The full-model and Metal NVFP4 profiles
use their native greedy path for zero temperature or `top_k: 1`; the legacy CUDA
profile retains its existing repetition-aware sampler.

The public [Sogni-AI/drama3-mlx API](https://github.com/Sogni-AI/drama3-mlx/blob/ecc235039ccd5f982f66a6982184f57432354ab4/drama3_mlx/model.py#L79)
is the behavioral reference for exposing these controls. These patches are
independent native Zig implementations; no Python inference dependency is added.

CPU regressions exercise validation and the values actually submitted through
the HTTP route. Device/model determinism and audio-quality qualification are
separate checks; this change makes no performance or quality claim.
