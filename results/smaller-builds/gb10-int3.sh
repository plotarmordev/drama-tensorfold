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
# the venv's interpreter links into the host's uv store, outside the job; its packages need a Python 3.12
python=$(command -v python3.12 || true)
if [ -z "$python" ]; then
  ls /usr/bin/python* /usr/local/bin/python* 2>&1
  exit 1
fi
# The FP8 checkpoint is still downloading into the pilot folder. Wait for its index and for every shard to stop
# growing, holding a CUDA context so the window does not close idle (30 minutes with nothing on the GPU).
PYTHONPATH="$pilot/venv/lib/python3.12/site-packages" "$python" - "$pilot/checkpoint" <<'EOF'
import json, sys, time
from pathlib import Path
import torch

held, src = torch.ones(1, device="cuda"), Path(sys.argv[1])
while True:
    index = src / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else None
    if shards and all((src / s).exists() and time.time() - (src / s).stat().st_mtime > 120 for s in shards):
        break
    time.sleep(60)
print("checkpoint complete:", time.strftime("%T"), flush=True)
EOF
cd /work/out
PYTHONPATH="$pilot/venv/lib/python3.12/site-packages" "$python" "$work/$tool" "$pilot/checkpoint" \
  "$pilot/eq/runs/fp8-calib-rows" int3 --mode int3 --provenance "$pilot/models/eq-nvfp4-v3/nvfp4-provenance.json" \
  2>&1 | tee int3.log
