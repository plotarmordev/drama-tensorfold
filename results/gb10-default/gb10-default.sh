#!/usr/bin/env bash
# The default build on a GB10 in one job: scripts/build-engine.sh's default engine, the int3 weights checked against
# recipe/checksums/trim-bf16-int3.sha256, served without systemd and asked for speech once:
# `ssh top-menu "job start gpu top plotarmordev/drama-tensorfold <sha> results/gb10-default/gb10-default.sh"`.
# No network: the Desk first stages two folders in the Drama pilot folder, each sent with `put top <folder>`:
#   drama-3-trim-bf16-int3  hf download plotarmordev/drama-3-NVFP4-Pruned-5900 --revision <recipe/sources.json's
#                           int3_5900_checkpoint revision> --local-dir drama-3-trim-bf16-int3
#   tensorfold              git clone https://github.com/plotarmordev/TensorFold.git tensorfold
#   nvcc-host-bundle        nvcc-host.tar.gz: Debian bookworm arm64's g++-12, libstdc++-12-dev, libgcc-12-dev,
#                           libc6-dev, linux-libc-dev, libgmp10, libmpfr6, libmpc3, libisl23, zlib1g and libzstd1;
#                           nvcc cannot parse the libc++ headers that the image's zig c++ supplies
# Prints the weights revision, the HTTP status, the seconds of audio and the time to first audio; any failure exits
# non-zero with its error.
set -euo pipefail
pilot=/assets/drama3-pilot
weights=$pilot/drama-3-trim-bf16-int3
bundle=$pilot/nvcc-host-bundle/nvcc-host.tar.gz
for path in "$weights/config.json" "$pilot/tensorfold/.git" "$bundle"; do
    [ -e "$path" ] || { echo "missing $path; /assets holds:" >&2; find /assets -maxdepth 2 >&2; exit 1; }
done
work=$(mktemp -d)
host=$work/nvcc-host
tar -xzf "$bundle" -C "$work"
# a copy the job owns: build-engine.sh checks out and patches it in place, and finds the pinned commit already there
cp -R "$pilot/tensorfold" "$work/tensorfold"
chmod -R u+w "$work/tensorfold"
# The image has no C++ compiler for nvcc. -B finds cc1plus; --sysroot finds GCC's and glibc's staged headers.
printf '#!/bin/sh\nexport LD_LIBRARY_PATH="%s/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec "%s/usr/bin/g++-12" -B"%s/usr/lib/gcc/aarch64-linux-gnu/12/" --sysroot="%s" "$@"\n' \
    "$host" "$host" "$host" "$host" > "$work/g++"
chmod +x "$work/g++"
echo '#include <vector>' | "$work/g++" -xc++ -E - >/dev/null
export NVCC_PREPEND_FLAGS="-ccbin $work/g++"
export ZIG_GLOBAL_CACHE_DIR=$work/zig ZIG_LOCAL_CACHE_DIR=$work/zig
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh "$work/tensorfold"
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 results/gb10-default/speak_once.py "$work/tensorfold/zig-out/native/bin/tensorfold-native" "$weights"
