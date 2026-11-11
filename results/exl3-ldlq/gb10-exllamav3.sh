#!/usr/bin/env bash
# exllamav3 bd1f519's own quantize_exl3 on card 35's six experts at 3.0 and 2.5 bits, on a GB10, from the repo root.
# exllamav3 builds its CUDA extension on first import, which needs a host C++ compiler: the job image has none, so
# this runs on the Spark host itself (`bottom shell`) with PILOT=<the pilot folder> OUT=<a scratch folder>; the
# defaults are the job door's mounts. Writes OUT/exl3-ref/exllamav3.json (per-projection Hessian error), parity.json
# (ours over exllamav3's, from probes.json) and the log.
set -euo pipefail

pilot=${PILOT:-/assets/drama3-pilot}
here=$PWD/results/exl3-ldlq
out=${OUT:-/work/out}/exl3-ref
sha256sum -c <<EOF
15cfc9a4984d04c7d40acaa98ea9cbe55b4b6cce6c35bf2da6bca5e6ca42f833  $here/exllamav3-bd1f519.tar.gz
EOF

work=$(mktemp -d)
mkdir -p "$out"
tar -xzf "$here/exllamav3-bd1f519.tar.gz" -C "$work"

nvidia-smi -L
# the venv's python links into the host's uv store; its packages need a Python 3.12
python=$(command -v python3.12)
# exllamav3 compiles its CUDA extension on first import (torch.utils.cpp_extension), for the GB10 only; that needs a
# host C++ compiler and ninja (the venv's, a native binary), so the run stops here, before any work, without either
packages=$pilot/venv/lib/python3.12/site-packages
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 TORCH_EXTENSIONS_DIR=$work/ext
export PATH=$CUDA_HOME/bin:$pilot/venv/bin:$PATH PYTHONPATH=$work/exllamav3-bd1f519:$here:$packages
for tool in c++ ninja nvcc; do command -v "$tool" || { echo "no $tool to build exllamav3's extension" >&2; exit 1; }; done
{
  "$python" "$here/exllamav3_job.py" "$pilot/checkpoint" "$here/hessians" "$out/exllamav3.json"
  "$python" "$here/measure.py" parity "$here/probes.json" "$out/exllamav3.json" "$out/parity.json"
} 2>&1 | tee "$out/exllamav3.log"
