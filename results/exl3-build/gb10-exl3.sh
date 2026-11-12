#!/usr/bin/env bash
# All 5,900 kept routed experts of t5900-cover at EXL3 3.0 bits by exllamav3 bd1f519, on a GB10: ~3.5 GPU-hours.
# `results/smaller-builds/gb10-start.sh <sha> results/exl3-build/gb10-exl3.sh`, with the Drama pilot folder at
# /assets/drama3-pilot (read-only) and no network. It needs the NVFP4-mix rows staged at $pilot/exl3-rows-v3
# (checked against rows.sha256). Writes /work/out/exl3/layer-LL.safetensors and layer-LL.json per layer, and the log.
# A job that stops resumes in the next one: copy its recipe.json and finished layer pairs into $pilot/exl3-done.
# The encoder validates every resumed shard against its record before skipping it.
set -euo pipefail

pilot=/assets/drama3-pilot
out=/work/out/exl3
here=$PWD/results/exl3-build
rows=$pilot/exl3-rows-v3
(cd "$rows" && sha256sum --quiet -c "$here/rows.sha256")
. results/exl3-ldlq/exllamav3-env.sh
mkdir -p "$out"
if [ -d "$pilot/exl3-done" ]; then
  cp "$pilot"/exl3-done/recipe.json "$out"/
  shopt -s nullglob
  for record in "$pilot"/exl3-done/layer-*.json; do
    cp "${record%.json}.safetensors" "$out"/
    cp "$record" "$out"/
  done
fi
nice -n 10 python3.12 "$here/encode.py" "$pilot/checkpoint" "$rows" "$out" 2>&1 | tee -a "$out/encode.log"
