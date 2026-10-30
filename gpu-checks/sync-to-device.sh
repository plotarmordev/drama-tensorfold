#!/usr/bin/env bash
set -eu
# GPU_CHECK_SSH_HOST: ssh destination of the device host; the work directory path is mirrored there.
: "${GPU_CHECK_SSH_HOST:?set GPU_CHECK_SSH_HOST to the device host ssh destination}"
here=$(cd "$(dirname "$0")" && pwd)
w=$(cd "$here/.." && pwd)
branch=$(git -C "$w" branch --show-current)
# A = main f68738ea, B = PR #33's v1.1.9 tree 3f44de62 (runner.py SHAS); each tree carries its own bench corpus.
git -C "$w" bundle create "$w/pr.bundle" main "$branch" f68738ea91de74dcf8bb698a26df44b5cf078396 3f44de621379bbe20f1ae6599a26465edf9dcbce
ssh "$GPU_CHECK_SSH_HOST" mkdir -p "$w/gpu-checks"
rsync -a "$here/" "$GPU_CHECK_SSH_HOST:$w/gpu-checks/"
rsync -a "$w/pr.bundle" "$GPU_CHECK_SSH_HOST:$w/pr.bundle"
