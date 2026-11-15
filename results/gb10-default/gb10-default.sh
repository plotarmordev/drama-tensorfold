#!/usr/bin/env bash
# The default build on a GB10 in one job: scripts/build-engine.sh's default engine, the int3 weights checked against
# recipe/checksums/trim-bf16-int3.sha256, served without systemd and asked for speech once:
# `ssh top-menu "job start gpu top plotarmordev/drama-tensorfold <sha> results/gb10-default/gb10-default.sh"`.
# No network: the Desk first stages two folders in the Drama pilot folder, each sent with `put top <folder>`:
#   drama-3-trim-bf16-int3  hf download plotarmordev/drama-3-NVFP4-Pruned-5900 --revision <recipe/sources.json's
#                           int3_5900_checkpoint revision> --local-dir drama-3-trim-bf16-int3
#   tensorfold              git clone https://github.com/plotarmordev/TensorFold.git tensorfold
# Prints the weights revision, the HTTP status, the seconds of audio and the time to first audio; any failure exits
# non-zero with its error.
set -euo pipefail
pilot=/assets/drama3-pilot
weights=$pilot/drama-3-trim-bf16-int3
for path in "$weights/config.json" "$pilot/tensorfold/.git"; do
    [ -e "$path" ] || { echo "missing $path; /assets holds:" >&2; find /assets -maxdepth 2 >&2; exit 1; }
done
work=$(mktemp -d)
# a copy the job owns: build-engine.sh checks out and patches it in place, and finds the pinned commit already there
cp -R "$pilot/tensorfold" "$work/tensorfold"
chmod -R u+w "$work/tensorfold"
# the image has no C++ compiler for nvcc but zig's
printf '#!/bin/sh\nexec zig c++ "$@"\n' > "$work/clang++"
chmod +x "$work/clang++"
export NVCC_PREPEND_FLAGS="-ccbin $work/clang++ -allow-unsupported-compiler"
export ZIG_GLOBAL_CACHE_DIR=$work/zig ZIG_LOCAL_CACHE_DIR=$work/zig
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh "$work/tensorfold"
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 results/gb10-default/speak_once.py "$work/tensorfold/zig-out/native/bin/tensorfold-native" "$weights"
