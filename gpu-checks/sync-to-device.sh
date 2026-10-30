#!/usr/bin/env bash
set -eu
# GPU_CHECK_SSH_HOST: ssh destination of the device host; the work directory path is mirrored there.
: "${GPU_CHECK_SSH_HOST:?set GPU_CHECK_SSH_HOST to the device host ssh destination}"
here=$(cd "$(dirname "$0")" && pwd)
w=$(cd "$here/.." && pwd)
branch=$(git -C "$w" branch --show-current)
git -C "$w" bundle create "$w/pr.bundle" main "$branch" a7752b4591da9b33bf6e7f8f2b478a638bc64bd1 3f44de621379bbe20f1ae6599a26465edf9dcbce
ssh "$GPU_CHECK_SSH_HOST" mkdir -p "$w/gpu-checks" "$w/bench/corpus"
rsync -a "$here/" "$GPU_CHECK_SSH_HOST:$w/gpu-checks/"
rsync -a "$w/pr.bundle" "$GPU_CHECK_SSH_HOST:$w/pr.bundle"
# The device host has no git checkout, so the pinned corpus (runner.py CORPUS_SHA256 preflight) travels with the scripts.
rsync -a "$w/bench/corpus/drama-bench-en-v1.jsonl" "$GPU_CHECK_SSH_HOST:$w/bench/corpus/drama-bench-en-v1.jsonl"
