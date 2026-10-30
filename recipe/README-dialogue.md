# Speaker-tagged dialogue

The patched speech routes accept canonical tags `<|speaker:0|>` through
`<|speaker:15|>` in request text and reference transcripts. Untagged requests
still start as speaker 0. Tags are preserved in the model prompt:

```json
{
  "model": "fishaudio/drama-3",
  "input": "<|speaker:1|> Is the train here? <|speaker:15|> It arrives at seven fifteen.",
  "voice": "dialogue",
  "response_format": "pcm"
}
```

The example assumes a catalog named `dialogue` is loaded. On the full-model and
Metal NVFP4 profiles, label each encoded reference transcript with its matching
speaker ID when building the catalog:

```sh
python3 recipe/build_voice.py --source synthetic dialogue 'Model-generated reference voices.' '<|speaker:1|> First reference transcript.:first.codes' '<|speaker:15|> Second reference transcript.:second.codes'
```

Use transcripts that actually match the reference clips. The CUDA NVFP4 profile
keeps its existing encoder catalog and request-code-reference formats; their
transcripts accept the same tags. No reference-audio upload route is added.

The full-model and Metal NVFP4 catalogs, and the Python writer, retain their
four-reference limit. Legacy CUDA keeps its existing body, file and prompt-size
admission bounds; it has no fixed four-reference count cap.
A tag selects a model speaker label; it is not a guarantee of a distinct or
faithfully cloned voice. Listening and reference-conditioned device tests remain
necessary to assess speaker identity.

Malformed tags such as `<|speaker:01|>`, IDs outside 0..15, structural prompt
tokens and the existing size/reference limits remain rejected. In the legacy
CUDA long-form path, continuation segments retain the active speaker across
automatic cuts, forced paragraph boundaries and redraws.

The legacy CUDA splitter keeps its existing character, segment, prompt and
frame limits. It does not adopt the MLX chunker's policy of at most three
distinct speakers per chunk. Accepting sixteen valid speaker IDs does not
qualify the acoustic quality of sixteen speakers in a single chunk. The
full-model and Metal profiles do not gain a long-form splitter from this patch.

## Known full-model stopping limitation

On one GB10 with the pinned CUDA full-model profile, this page's two-speaker
example reached the 1024-frame cap (47.55 seconds, `X-Finish-Reason: length`)
in three of five seeds with a matching reference catalog. The same text without
speaker tags stopped normally in all five. Relabeling both references and input
to speakers 0/1 still reached the cap in two of five seeds; using the tagged
input without a reference catalog did so in one of five. Neither variation is a
reliable workaround.

The CUDA NVFP4 profile, with its default guidance, stopped normally on all five
tagged and all five untagged requests in this small check. The engine pins,
weights and guidance differ, so these observations do not establish the cause
or a guarantee for NVFP4. Metal was not tested in this follow-up. See the
[settings and per-seed results](../results/dialogue-stopping-20261030.json).

Treat `X-Finish-Reason: length` as an incomplete generation. The cause remains
unresolved; accepting a speaker tag does not guarantee natural termination.
The investigation is tracked in [issue #32](https://github.com/plotarmordev/drama-tensorfold/issues/32).

The public [Sogni-AI/drama3-mlx prompt tests](https://github.com/Sogni-AI/drama3-mlx/blob/ecc235039ccd5f982f66a6982184f57432354ab4/tests/test_prompt.py#L38)
provide the behavioral reference. These changes extend TensorFold's existing
native prompt builder without copying Fish-derived Python implementation code.
