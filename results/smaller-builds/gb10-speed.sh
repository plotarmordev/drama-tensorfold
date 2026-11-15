#!/usr/bin/env bash
# The int3 build's and t5900's speed on a GB10, one engine binary:
# `ssh top-menu "job start gpu top plotarmordev/drama-tensorfold <sha> results/smaller-builds/gb10-speed.sh"`, with
# the pilot folder holding the int3 checkpoint with its FP32 codec and SHA256SUMS (gb10-speed-stage.sh) and the engine
# (`PIN=engine/nvfp4 OUT=drama3-int3-engine results/exl3-spark/prebuild-spark.sh`), each sent with `put top <folder>`.
# No network: the job checks and runs what is staged, and writes /work/out/spark-speed/speed.json.
set -euo pipefail
pilot=/assets/drama3-pilot
out=/work/out/spark-speed
int3=$pilot/drama3-trim-bf16-int3
t5900=$pilot/models/drama-3-NVFP4-Pruned-5900
engine=$pilot/drama3-int3-engine
for path in "$int3/SHA256SUMS" "$engine/SHA256SUMS" "$pilot/voices-eq3ref/eq-3ref.json"; do
    [ -e "$path" ] || { echo "missing $path; /assets holds:" >&2; find /assets -maxdepth 2 >&2; exit 1; }
done
(cd "$int3" && sha256sum --quiet -c SHA256SUMS)
pins=$PWD/recipe/checksums/nvfp4-t5900cover.sha256
(cd "$t5900" && sha256sum --quiet -c "$pins")
(cd "$engine" && sha256sum --quiet -c SHA256SUMS)
echo "engine $(sha256sum < "$engine/SHA256SUMS" | cut -d' ' -f1)"
echo "expected window: 2 builds x 2 voices x 2 passes of 24 texts, ~45 min of speech at quant-15gb's RTF + 4 loads"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$out"
python3 results/smaller-builds/spark_speed.py "$engine/tensorfold-native" "$int3" "$t5900" "$pilot/voices-eq3ref" \
    "$out" 2>&1 | tee -a "$out/speed.log"
