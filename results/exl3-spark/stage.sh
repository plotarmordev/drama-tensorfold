#!/bin/sh
# Stages everything the GB10 speak job needs into the Drama pilot folder, on BOTTOM through `bottom shell`, before
# any window opens: the assembled trim-bf16-exl3-3.0 checkpoint (13.51 GiB) from the pilot's own t5900-cover model
# and its own exl3-done experts, nothing copied from the Mac. Run from anywhere on BOTTOM:
#   ssh top-menu "bottom shell" < results/exl3-spark/stage.sh   # env: SHA (default origin/main), PILOT (see below)
# The engine binary is staged separately by prebuild-spark.sh once the patch series is in.
set -eu
repo_sha=${SHA:-origin/main}
pilot=${PILOT:-"$HOME/Documents/drama3-pilot-20261014"}
work=$(mktemp -d ~/drama3-exl3-stage.XXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet https://github.com/plotarmordev/drama-tensorfold.git "$work/repo"
git -C "$work/repo" checkout --quiet "$repo_sha"
repo=$work/repo

# The pilot's inputs: the published 5,900-expert quant and the EXL3 job's finished layers, both verified against the
# repo's pins before anything is built from them.
(cd "$pilot/models/drama-3-NVFP4-Pruned-5900" && sha256sum --quiet -c "$repo/recipe/checksums/nvfp4-t5900cover.sha256")
(cd "$pilot/exl3-done" && sha256sum --quiet -c "$repo/results/exl3-build/exl3.sha256")
[ -f "$pilot/exl3-done/recipe.json" ]

py=$pilot/venv/bin/python
stage=$pilot/drama3-exl3-5900
[ ! -e "$stage" ] || { echo "$stage exists" >&2; exit 1; }
trim=$work/trim
$py "$repo/recipe/trim_head.py" "$pilot/models/drama-3-NVFP4-Pruned-5900" "$trim"
$py "$repo/recipe/codec_bf16.py" "$trim/codec.pth" "$trim/codec-bf16.pth" && mv "$trim/codec-bf16.pth" "$trim/codec.pth"
cd "$repo/results/exl3-build"
$py assemble.py "$trim" "$pilot/exl3-done" "$stage"
cd "$stage" && find . -name '*.safetensors' -o -name codec.pth | sort | xargs sha256sum > "$stage.sha256"
echo "staged $stage: $(sha256sum < "$stage.sha256" | cut -d' ' -f1)"
