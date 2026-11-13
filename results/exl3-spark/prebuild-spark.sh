#!/bin/bash
# Builds the EXL3 speak job's engine binary for the Spark (GB10, sm_121) and stages it in the Drama pilot folder,
# BEFORE any GPU window: the Desk runs this on BOTTOM through `bottom shell` (BOTTOM has network and docker; the
# GPU job has neither). It takes the engine pin (TensorFold main @ 132bd123 + the EXL3 patch) from the pushed
# drama-tensorfold commit $SHA, applies it in a clean checkout, and builds `native` inside the receipts' job image
# with nvcc from the CUDA 13.0 mount — the same image and driver the window will run. Output:
#   $PILOT/drama3-exl3-engine/{tensorfold-native,SHA256SUMS}   (refused if it exists)
# Usage:  SHA=<pushed sha> bash results/exl3-spark/prebuild-spark.sh
set -euo pipefail

sha=${SHA:?set SHA to the pushed drama-tensorfold commit}
repo=${REPO:-https://github.com/plotarmordev/drama-tensorfold.git}
pilot=${PILOT:-$HOME/Documents/drama3-pilot-20261014}
image=${IMAGE:-macbots:49eafd253777}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
out=$pilot/drama3-exl3-engine
[ ! -e "$out" ] || { echo "$out exists; move it away before rebuilding" >&2; exit 1; }

git clone -q "$repo" "$work/drama"
git -C "$work/drama" checkout -q "$sha"
pin=$work/drama/engine/nvfp4/spark
# shellcheck disable=SC1091
. "$pin/UPSTREAM"
[ -n "${TENSORFOLD_REPOSITORY:-}" ] && [ -n "${TENSORFOLD_COMMIT:-}" ] || { echo "$pin/UPSTREAM is incomplete" >&2; exit 1; }
(cd "$pin/patches" && sha256sum -c SHA256SUMS)

git clone -q "$TENSORFOLD_REPOSITORY" "$work/tf"
git -C "$work/tf" checkout -q "$TENSORFOLD_COMMIT"
git -C "$work/tf" am -q --committer-date-is-author-date "$pin"/patches/*.patch
echo "engine: $TENSORFOLD_COMMIT + $(ls "$pin"/patches/*.patch | wc -l) patch(es) = $(git -C "$work/tf" rev-parse HEAD)"

mkdir -p "$out"
docker run --rm --user 0 --entrypoint /bin/sh -e OWNER="$(id -u):$(id -g)" \
    -v /usr/local/cuda-13.0:/usr/local/cuda:ro -v "$work/tf":/src:rw -v "$out":/out -w /src "$image" -euc '
trap "chown -R $OWNER /out" EXIT
[ "$(zig version)" = "$(cat /src/.zig-version)" ] || { echo "zig $(zig version), the engine pins $(cat /src/.zig-version)" >&2; exit 1; }
apt-get update -qq && apt-get install -y -qq g++ >/dev/null
work=$(mktemp -d)
ZIG_GLOBAL_CACHE_DIR=$work ZIG_LOCAL_CACHE_DIR=$work \
    zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix "$work/zig-out"
cp "$work"/zig-out/native/bin/tensorfold-native /out/
cd /out && sha256sum tensorfold-native > SHA256SUMS
' || { status=$?; rm -rf "$out"; exit $status; }

[ -s "$out/tensorfold-native" ] && [ -s "$out/SHA256SUMS" ]
file "$out/tensorfold-native" || true
echo "$out: $(sha256sum < "$out/SHA256SUMS" | cut -d" " -f1)"
