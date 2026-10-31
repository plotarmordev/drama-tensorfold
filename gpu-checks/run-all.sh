#!/usr/bin/env bash
set -u
here=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} == --plan ]]; then
  exec python3 "$here/runner.py" --plan
fi
trap '"$here/resume-paused.sh"' EXIT
python3 "$here/runner.py" "$@"
