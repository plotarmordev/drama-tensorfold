#!/bin/sh
# Stages the Spark speed job's checkpoint in the Drama pilot folder, on BOTTOM through `bottom shell`, before any
# window: the published int3 revision fetched from the Hub and checked against recipe/checksums/trim-bf16-int3.sha256,
# then its BF16 codec swapped for the pilot t5900-cover model's FP32 one, which the CUDA pin reads (the card's CUDA
# recipe). Run from anywhere on BOTTOM:
#   ssh top-menu "bottom shell" < results/smaller-builds/gb10-speed-stage.sh   # env: SHA (default origin/main), PILOT
# The engine is built by results/exl3-spark/prebuild-spark.sh with PIN=engine/nvfp4 OUT=drama3-nvfp4-engine.
set -eu
repo_sha=${SHA:-origin/main}
pilot=${PILOT:-"$HOME/Documents/drama3-pilot-20261014"}
revision=0a0afd5d0825d39443e522b27aab6eb2f85287bd
stage=$pilot/drama3-trim-bf16-int3
[ ! -e "$stage" ] || { echo "$stage exists" >&2; exit 1; }
work=$(mktemp -d ~/drama3-int3-stage.XXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet https://github.com/plotarmordev/drama-tensorfold.git "$work/repo"
git -C "$work/repo" checkout --quiet "$repo_sha"
pins=$work/repo/recipe/checksums

t5900=$pilot/models/drama-3-NVFP4-Pruned-5900
(cd "$t5900" && sha256sum --quiet -c "$pins/nvfp4-t5900cover.sha256")
mkdir "$work/ckpt"
cut -c67- "$pins/trim-bf16-int3.sha256" | while read -r file; do
  curl -fsSL --retry 3 -o "$work/ckpt/$file" \
    "https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned-5900/resolve/$revision/$file"
done
(cd "$work/ckpt" && sha256sum --quiet -c "$pins/trim-bf16-int3.sha256")
cp "$t5900/codec.pth" "$work/ckpt/codec.pth"
mv "$work/ckpt" "$stage"
(cd "$stage" && sha256sum $(cut -c67- "$pins/trim-bf16-int3.sha256")) > "$stage.sha256"
echo "staged $stage: $(sha256sum < "$stage.sha256" | cut -d' ' -f1)"
