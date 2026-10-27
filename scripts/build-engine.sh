#!/usr/bin/env bash
# Builds the TensorFold engine for the chosen Drama 3 weights.
#
# Usage: scripts/build-engine.sh [--weights full|nvfp4] [BUILD_DIR]
#
#   --weights full    (the default) upstream TensorFold at engine/UPSTREAM with the opt-in patches in
#                     engine/patches: the native server and drama3-generate, in build/tensorfold
#   --weights nvfp4   CUDA: engine/nvfp4/UPSTREAM plus its patches, native server.
#                     Metal: engine/nvfp4/macos/UPSTREAM plus its patches, native server and voice tools.
#                     Both use plotarmordev/drama-3-NVFP4-Pruned, in build/tensorfold-nvfp4.
#
# Environment:
#   ZIG          Zig 0.17.0 executable (default: zig on PATH)
#   NVCC         CUDA 13 nvcc (default: nvcc on PATH)
#   JOBS         parallel build jobs (default: Zig's choice)
#
# Every run checks out the pinned commit and re-applies the whole series, so the result does not depend on what an
# earlier run left behind. Output: BUILD_DIR/zig-out/native/bin/tensorfold-native, plus the selected voice tools
# under BUILD_DIR/zig-out/bin.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zig=${ZIG:-zig}
nvcc=${NVCC:-nvcc}
platform=$(uname -s)

die() {
  echo "build-engine: $*" >&2
  exit 1
}

weights=full
if [[ ${1:-} == --weights ]]; then
  (($# >= 2)) || die "--weights needs full or nvfp4"
  weights=$2
  shift 2
elif [[ ${1:-} == --weights=* ]]; then
  weights=${1#--weights=}
  shift
fi
(($# <= 1)) || die "usage: scripts/build-engine.sh [--weights full|nvfp4] [BUILD_DIR]"
patch_dir=
case $weights in
  full)
    pin=$repo/engine/UPSTREAM build=${1:-$repo/build/tensorfold}
    patch_dir=$repo/engine/patches
    targets=(native drama3-generate)
    ;;
  nvfp4)
    pin=$repo/engine/nvfp4/UPSTREAM build=${1:-$repo/build/tensorfold-nvfp4}
    patch_dir=$repo/engine/nvfp4/patches
    targets=(native)
    if [[ $platform == Darwin ]]; then
      pin=$repo/engine/nvfp4/macos/UPSTREAM
      patch_dir=$repo/engine/nvfp4/macos/patches
      targets+=(drama3-generate drama3-voice)
    fi
    ;;
  *) die "--weights must be full or nvfp4, not '$weights'" ;;
esac

# shellcheck source=../engine/UPSTREAM
. "$pin"
[[ -n ${TENSORFOLD_REPOSITORY:-} && -n ${TENSORFOLD_COMMIT:-} ]] || die "$pin must set TENSORFOLD_REPOSITORY and TENSORFOLD_COMMIT"

shopt -s nullglob
patches=()
if [[ -n $patch_dir ]]; then
  patches=("$patch_dir"/*.patch)
  ((${#patches[@]})) || die "$patch_dir has no *.patch files"
  [[ -f $patch_dir/SHA256SUMS ]] || die "$patch_dir has no SHA256SUMS manifest"
  manifest_pattern='^[[:xdigit:]]{64}  ([^/[:space:]]+\.patch)$'
  listed_patches=()
  while IFS= read -r manifest_line || [[ -n $manifest_line ]]; do
    [[ $manifest_line =~ $manifest_pattern ]] || die "invalid patch manifest entry"
    listed_patches+=("${BASH_REMATCH[1]}")
  done < "$patch_dir/SHA256SUMS"
  ((${#listed_patches[@]} == ${#patches[@]})) || die "patch manifest must list every selected patch exactly once"
  for patch in "${patches[@]}"; do
    matches=0
    for listed_patch in "${listed_patches[@]}"; do
      if [[ $listed_patch == "${patch##*/}" ]]; then matches=$((matches + 1)); fi
    done
    ((matches == 1)) || die "patch manifest must list every selected patch exactly once"
  done
  if command -v sha256sum >/dev/null; then
    (cd "$patch_dir" && sha256sum -c SHA256SUMS) || die "patch checksum verification failed"
  else
    (cd "$patch_dir" && shasum -a 256 -c SHA256SUMS) || die "patch checksum verification failed"
  fi
fi

command -v git >/dev/null || die "git not found"
zig=$(command -v "$zig") || die "Zig not found; set ZIG"
# macOS builds the Metal engine without requiring a CUDA compiler.
if [[ $platform == Darwin ]]; then
  build_args=()
  if [[ $weights == nvfp4 ]]; then
    build_args=(-Dcpu=apple_m1)
  fi
else
  nvcc=$(command -v "$nvcc") || die "nvcc not found; set NVCC"
  build_args=(-Dnvcc="$nvcc")
fi

if [[ -e $build ]]; then
  git -C "$build" rev-parse --git-dir >/dev/null 2>&1 || die "$build exists and is not a git checkout"
  [[ -z $(git -C "$build" status --porcelain --untracked-files=no) ]] || die "$build has local changes; refusing to discard them"
  origin=$(git -C "$build" remote get-url origin 2>/dev/null || true)
  [[ $origin == "$TENSORFOLD_REPOSITORY" ]] || die "$build is a checkout of '$origin', not $TENSORFOLD_REPOSITORY; pass another BUILD_DIR"
  git -C "$build" am --abort >/dev/null 2>&1 || true
else
  git clone --quiet --no-checkout "$TENSORFOLD_REPOSITORY" "$build"
fi
git -C "$build" cat-file -e "$TENSORFOLD_COMMIT^{commit}" 2>/dev/null || git -C "$build" fetch --quiet origin "$TENSORFOLD_COMMIT"
git -C "$build" checkout --quiet --detach "$TENSORFOLD_COMMIT"

# A fixed committer and dates taken from the patches make the patched HEAD the same on every run.
if ((${#patches[@]})); then
  GIT_COMMITTER_NAME=build-engine GIT_COMMITTER_EMAIL=build-engine@example.invalid \
    git -C "$build" am --quiet --committer-date-is-author-date "${patches[@]}" \
    || { git -C "$build" am --abort || true; die "the patch series does not apply to $TENSORFOLD_COMMIT"; }
fi

want=$(<"$build/.zig-version")
have=$("$zig" version)
[[ $have == "$want" ]] || die "Zig $have found; the engine pins $want"

cd "$build"
"$zig" build "${targets[@]}" ${build_args[@]+"${build_args[@]}"} --summary all ${JOBS:+-j"$JOBS"}

echo "engine ($weights weights): $TENSORFOLD_COMMIT + ${#patches[@]} patches = $(git rev-parse HEAD)"
artifacts=(zig-out/native/bin/tensorfold-native)
for target in "${targets[@]:1}"; do
  artifacts+=("zig-out/bin/$target")
done
if command -v sha256sum >/dev/null; then
  sha256sum "${artifacts[@]}"
else
  shasum -a 256 "${artifacts[@]}"
fi
