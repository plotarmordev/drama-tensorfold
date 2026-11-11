#!/usr/bin/env bash
# exllamav3 bd1f519's own quantize_exl3 on card 35's six experts at 3.0 and 2.5 bits, on a GB10:
# `job start gpu middle|bottom plotarmordev/<repo> <sha> results/exl3-ldlq/gb10-exllamav3.sh`, with the Drama pilot
# folder at /assets/drama3-pilot (read-only) and no network. Writes /work/out/exl3-ref/exllamav3.json (per-projection
# Hessian error), parity.json (ours over exllamav3's, from probes.json, and exllamav3's verdict against int3 and
# NVFP4) and the log.
set -euo pipefail

pilot=/assets/drama3-pilot
here=$PWD/results/exl3-ldlq
out=/work/out/exl3-ref
sha256sum -c <<EOF
15cfc9a4984d04c7d40acaa98ea9cbe55b4b6cce6c35bf2da6bca5e6ca42f833  $here/exllamav3-bd1f519.tar.gz
f2d768df28b8521d0756ac3bbfb9bd19d002c9b39a43b7eb2fd7af70869db075  $here/exllamav3_ext-bd1f519-sm121.so.gz
EOF

work=$(mktemp -d)
mkdir -p "$out"
tar -xzf "$here/exllamav3-bd1f519.tar.gz" -C "$work"
# the job image has no C++ compiler to build exllamav3's extension; build-exllamav3-ext.sh prebuilt it, and exllamav3
# imports an exllamav3_ext found on the path instead of compiling one
gunzip -c "$here/exllamav3_ext-bd1f519-sm121.so.gz" > "$work/exllamav3_ext.so"

nvidia-smi -L
# the venv's interpreter links into the host's uv store, outside the job; its packages need a Python 3.12
export PYTHONPATH=$work:$work/exllamav3-bd1f519:$here:$pilot/venv/lib/python3.12/site-packages
{
  python3.12 "$here/exllamav3_job.py" "$pilot/checkpoint" "$here/hessians" "$out/exllamav3.json"
  python3.12 "$here/measure.py" parity "$here/probes.json" "$out/exllamav3.json" "$out/parity.json"
} 2>&1 | tee "$out/exllamav3.log"
