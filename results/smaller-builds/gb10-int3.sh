#!/usr/bin/env bash
# GPTQ int3 routed experts for t5900-cover on a GB10: `job start gpu bottom` with the Drama pilot folder at
# /assets/drama3-pilot (read-only). Writes /work/out/int3/layer-LL.safetensors for every layer.
set -euo pipefail

pilot=/assets/drama3-pilot
tensorfold=b1f4b9b4830c9bd43bf7c432cffe7dea21313537  # engine/nvfp4/UPSTREAM
tool=tools/zig/drama_nvfp4_gptq.py

work=$(mktemp -d)
mkdir -p "$work/tools/zig" /work/out/int3
curl -fsSL "https://raw.githubusercontent.com/plotarmordev/TensorFold/$tensorfold/$tool" -o "$work/$tool"
git -C "$work" init -q
git -C "$work" apply --include="$tool" "$PWD/engine/nvfp4/patches/0006-int3-routed-experts.patch"

nvidia-smi -L
cd /work/out
/usr/bin/time -v "$pilot/venv/bin/python" "$work/$tool" "$pilot/checkpoint" "$pilot/eq/runs/fp8-calib-rows" int3 \
  --mode int3 --provenance "$pilot/models/eq-nvfp4-v3/nvfp4-provenance.json" 2>&1 | tee int3.log
