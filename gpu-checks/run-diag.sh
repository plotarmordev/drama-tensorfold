#!/usr/bin/env bash
set -u
# Repair round r2 (run-1 recovery) launches this script as unit gpu-checks-diag-r2 with
# --out-dir "$W/out-diag-r2"; see README 'Repair round r2'. Run-1 out-diag/ holds a summary and is refused.
here=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} == --plan ]]; then
  exec python3 "$here/runner.py" --diag --plan
fi
# The EXIT trap and ExecStopPost must recover the same output dir the runner writes to.
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ ${args[i]} == --out-dir && $((i + 1)) -lt ${#args[@]} ]]; then
    export GPU_CHECK_OUT=${args[i + 1]}
  fi
done
trap '"$here/resume-paused.sh"' EXIT
python3 "$here/runner.py" --diag "$@"
rc=$?
# 75 = memory WAIT (launch preflight or a later full-profile admission): INCONCLUSIVE summary written, no further
# pilot launched, not a unit failure. Every other code (corpus/hash/harness/code failure, 143) passes through.
if [[ $rc == 75 ]]; then
  echo "diag refused on memory WAIT (see summary); exiting cleanly" >&2
  exit 0
fi
exit "$rc"
