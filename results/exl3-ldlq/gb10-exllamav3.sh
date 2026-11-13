#!/usr/bin/env bash
# exllamav3 bd1f519's own quantize_exl3 on card 35's six experts at 3.0 and 2.5 bits, on a GB10:
# `job start gpu middle|bottom plotarmordev/<repo> <sha> results/exl3-ldlq/gb10-exllamav3.sh`, with the Drama pilot
# folder at /assets/drama3-pilot (read-only) and no network. Writes /work/out/exl3-ref/exllamav3.json (per-projection
# Hessian error), parity.json (ours over exllamav3's, from probes.json, and exllamav3's verdict against int3 and
# NVFP4) and the log.
set -euo pipefail

pilot=/assets/drama3-pilot
out=/work/out/exl3-ref
. results/exl3-ldlq/exllamav3-env.sh
mkdir -p "$out"
{
  python3.12 "$exl3/exllamav3_job.py" "$pilot/checkpoint" "$exl3/hessians" "$out/exllamav3.json"
  python3.12 "$exl3/measure.py" parity "$exl3/probes.json" "$out/exllamav3.json" "$out/parity.json"
} 2>&1 | tee "$out/exllamav3.log"
