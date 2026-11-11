#!/usr/bin/env bash
# Builds exllamav3 bd1f519's CUDA extension for the GB10 (sm_121) into results/exl3-ldlq/exllamav3_ext-bd1f519-sm121.so.gz,
# which gb10-exllamav3.sh loads: the job image has no C++ compiler, so the extension is built ahead, on a Spark host
# with docker and network, in python:3.12.12-bookworm (the job image's base: same Python, glibc and libstdc++, plus
# g++ 12), against the pilot venv's torch. Run from the repo root with PILOT=<the drama3-pilot folder>.
set -euo pipefail

pilot=${PILOT:?the drama3-pilot folder}
docker run --rm -v /usr/local/cuda-13.0:/usr/local/cuda:ro -v "$pilot:/assets/drama3-pilot:ro" -v "$PWD:/src" -w /src \
  python:3.12.12-bookworm sh -euc '
work=$(mktemp -d)
tar -xzf results/exl3-ldlq/exllamav3-bd1f519.tar.gz -C "$work"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 TORCH_EXTENSIONS_DIR=$work/ext
export PATH=$CUDA_HOME/bin:/assets/drama3-pilot/venv/bin:$PATH
export PYTHONPATH=$work/exllamav3-bd1f519:/src/results/exl3-ldlq:/assets/drama3-pilot/venv/lib/python3.12/site-packages
python3.12 -c "import exllamav3_job; exllamav3_job.import_quantize()"
gzip -9 -c "$work/ext/exllamav3_ext/exllamav3_ext.so" > results/exl3-ldlq/exllamav3_ext-bd1f519-sm121.so.gz'
