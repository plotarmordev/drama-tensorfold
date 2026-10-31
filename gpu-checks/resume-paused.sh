#!/usr/bin/env bash
set -u
here=$(cd "$(dirname "$0")" && pwd)
if [[ -n ${GPU_CHECK_OUT:-} ]]; then
  exec python3 "$here/runner.py" --resume
fi
status=0
# A diagnostic run pauses into out-diag; recover both records when both exist, newest behavior unchanged.
if [[ -f "$here/../out-diag/paused.txt" ]]; then
  python3 "$here/runner.py" --diag --resume || status=$?
fi
if [[ -f "$here/../out/paused.txt" ]]; then
  python3 "$here/runner.py" --resume || status=$?
fi
exit $status
