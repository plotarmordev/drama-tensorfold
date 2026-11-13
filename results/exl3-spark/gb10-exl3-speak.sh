#!/bin/sh
# The winning smaller build (trim-bf16-exl3-3.0, 13.51 GiB) end to end on a GB10, in one window:
# `results/smaller-builds/gb10-start.sh <sha> results/exl3-spark/gb10-exl3-speak.sh`, with everything it needs
# already staged in the Drama pilot folder by stage.sh (the assembled checkpoint from the pilot's own t5900-cover
# model and exl3-done experts, the eq-3ref voice) and prebuild-spark.sh (the engine binary). No network: the job
# only checks and runs what is staged. It speaks card 37's 24 texts in both voices at salts 0 and 1 and writes
# /work/out/exl3-spark: load time, peak device memory and seconds per take, next to every WAV.
set -eu
pilot=/assets/drama3-pilot
out=/work/out/exl3-spark
here=$PWD/results/exl3-spark
ckpt=$pilot/drama3-exl3-5900
engine=$pilot/drama3-exl3-engine
for path in "$ckpt/config.json" "$ckpt/codec.pth" "$engine/tensorfold-native" "$engine/SHA256SUMS" \
            "$pilot/voices-eq3ref/eq-3ref.json" "$pilot/drama3-exl3-5900.sha256"; do
    [ -e "$path" ] || { echo "missing $path; /assets holds:" >&2; find /assets -maxdepth 3 >&2; exit 1; }
done
(cd "$ckpt" && sha256sum --quiet -c "$pilot/drama3-exl3-5900.sha256")
(cd "$engine" && sha256sum --quiet -c SHA256SUMS)
echo "engine $(sha256sum < "$engine/SHA256SUMS" | cut -d' ' -f1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$out"
nice -n 10 python3 "$here/speak.py" "$engine/tensorfold-native" "$ckpt" "$pilot/voices-eq3ref" "$out" 2>&1 | tee -a "$out/speak.log"
