#!/usr/bin/env bash
# GPTQ int3 routed experts for t5900-cover on a GB10: `job start gpu bottom` with the Drama pilot folder at
# /assets/drama3-pilot (read-only). Writes /work/out/int3/layer-LL.safetensors for every layer.
set -euo pipefail

pilot=/assets/drama3-pilot
tool=tools/zig/drama_nvfp4_gptq.py
# the pinned TensorFold's (engine/nvfp4/UPSTREAM b1f4b9b4) tool, as the pilot keeps it; the job has no network
pinned=$pilot/eq/bin/fallback-cb17869b/drama_nvfp4_gptq.py
echo "93e5eff8efe49746dbe9710b9de07ed478827f6f1cc0c5eb385d67f20c533b53  $pinned" | sha256sum -c

work=$(mktemp -d)
mkdir -p "$work/tools/zig" /work/out/int3
cp "$pinned" "$work/$tool"
git -C "$work" init -q
git -C "$work" apply --include="$tool" "$PWD/engine/nvfp4/patches/0006-int3-routed-experts.patch"

nvidia-smi -L
cd /work/out
"$pilot/venv/bin/python" "$work/$tool" "$pilot/checkpoint" "$pilot/eq/runs/fp8-calib-rows" int3 \
  --mode int3 --provenance "$pilot/models/eq-nvfp4-v3/nvfp4-provenance.json" 2>&1 | tee int3.log
