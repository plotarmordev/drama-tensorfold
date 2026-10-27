# Recipe: the guarded launcher

`pilot.py` starts one Drama 3 server, native or reference, as a transient `systemd --user` unit, and stops it
cleanly. It needs Linux, user systemd and Python 3 (standard library only).

On Apple silicon use [`pilot_macos.py`](README-macos.md), with the same explicit `--weights`
selection and checkpoint manifests. It uses native Metal and owned process groups, not systemd.
Its NVFP4 engine pin is separate from CUDA; the full-model pin remains unchanged.

The Linux NVFP4 build applies `engine/nvfp4/patches` after its pinned base. Its long-form
splitter keeps delivery markup intact; see [CUDA NVFP4 long speech](../README.md#cuda-nvfp4-long-speech).

```sh
python3 recipe/pilot.py --weights nvfp4 \
  --runtime-dir "$RUNTIME_DIR" \
  --gpu-lock "$GPU_LOCK" \
  --evidence-dir "$EVIDENCE_DIR"
```

`--weights nvfp4` or `--weights full` picks the pinned checkpoint, `checksums/nvfp4.sha256` or
`checksums/checkpoint.sha256`, and its engine build in `build/tensorfold-nvfp4` or `build/tensorfold` from
`scripts/build-engine.sh --weights …`. On first use the launcher downloads the checkpoint with `hf download` at the
revision in `sources.json`, into `drama-3-NVFP4-Pruned/` or `drama-3/` at the repository root. Every start then
hashes every pinned file and refuses, before it takes the GPU lock, if one is missing or changed. With `nvfp4` it adds
`--cfg-gamma 1.5 --cfg-batched`, the guidance the quant was measured with, unless a `--cfg-gamma`,
`--cfg-allow-request-gamma` or `--cfg-batched` of your own follows `--`. `--checkpoint` and `--native-bin` override
the directory and executable, but a selected `--weights` manifest is still checked against an overridden
checkpoint. Only an absent default checkpoint is downloaded; explicit paths and existing incomplete directories
are refused rather than repaired. Without `--weights`, pass both paths; nothing is downloaded or added,
and no pinned-checkpoint verification is claimed.

For the SGLang-Omni reference, replace `--native-bin …` with `--backend sglang-omni`. The reference also needs
`venv/` (with `sgl-omni` installed) and an `sglang-omni/` checkout at the revision in `sources.json`, both inside
the runtime directory. Add `--library-dir` (repeatable) when the CUDA libraries are not on the loader's search
path. `CUDA_HOME` (default `/usr/local/cuda`) puts the toolkit's `bin/` on the server's `PATH`.

| Argument | Meaning |
|---|---|
| `--runtime-dir` | An existing working directory; holds every Hugging Face, Triton, TorchInductor, CUDA and FlashInfer cache |
| `--checkpoint` | A downloaded checkpoint directory, verified beforehand; with `--weights`, default its download |
| `--gpu-lock` | The host's existing shared lock file, the one every GPU job on the machine takes, so two models never load at once. Only on a host without one, create it with `touch` and have the other GPU jobs take it too. |
| `--evidence-dir` | New directory for `session.json`, `ready.json`, `memory.jsonl`, `server.log` and `completion.json` |
| `--port` | Loopback port, default 18793 |

## The guard

| Quantity | GiB |
|---|---:|
| Checkpoint weights plus codec, on disk (not runtime memory): full / NVFP4 | 38.63 / 20.76 |
| Unit memory cap (`MemoryMax`), swap disabled | 54 |
| Host headroom floor | 30 |
| Abort threshold on `MemAvailable` | 34 |
| `MemAvailable` required before loading | 88 |

The launcher:

- takes the shared lock without blocking and refuses to start if another job holds it;
- refuses to start below the admission figure or on an occupied port;
- samples `MemAvailable` every 100 ms and kills the whole unit if it falls below 34 GiB;
- on exit, keeps the lock until systemd confirms the unit is inactive.

The native server caps tracked CUDA and pinned-host buffers together at 54 GiB
(`TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54`) and keeps 34 GiB of host headroom at admission
(`TENSORFOLD_MEMORY_RESERVE_GIB=34`). The reference's PyTorch allocator budgets are 4 GiB for preprocessing,
44 GiB for the engine and 6 GiB for the vocoder (`reference/drama3-gb10.yaml`). Neither the caps nor the cgroup
count every driver or library allocation, which is why the launcher also watches `MemAvailable`.

On GB10, reclaimable page cache can make `MemAvailable` look healthy while CUDA reports little free memory
([NVIDIA known issue](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).
`bench/admit.py` checks CUDA free memory and `MemFree` directly before a load.

## Server settings

Both backends serve `fishaudio/drama-3` on loopback with the built-in `default` voice (no reference audio),
concurrency 1 and at most 1024 frames (47.55 s) per reply. Readiness is `GET /health`; synthesis is
`POST /v1/audio/speech`.

The reference YAML limits the KV cache by tokens (8192) rather than bytes, because the hybrid DeltaNet cache
rejects a byte budget. Its 44 GiB engine allocator cap is unchanged.

## Files

- `pilot.py`: the launcher.
- `reference/drama3-gb10.yaml`: the sglang-omni pipeline config used for every reference measurement.
- `checksums/checkpoint.sha256`: SHA-256 of the codec, the eight weight shards and the tokenizer at checkpoint
  revision `e0a9218a`.
- `checksums/nvfp4.sha256`: SHA-256 of the codec, the eight weight shards, their index, the configs and the
  tokenizer files of `plotarmordev/drama-3-NVFP4-Pruned` at revision `494db084`.
- `sources.json`: the pinned reference and checkpoint revisions.
