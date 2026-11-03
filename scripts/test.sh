#!/usr/bin/env bash
# Runs every test in this repository that needs no model, GPU or network; exits non-zero on any failure.
#
# Usage: scripts/test.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# The macOS lifecycle smoke uses only short-lived dummy subprocesses; on Linux it skips itself.
PILOT_MACOS_INTEGRATION=1 python3 -m unittest discover -s recipe -p 'test_*.py'
python3 bench/stream_check.py --self-test
if [[ $(uname -s) == Linux ]]; then
  gpu-checks/selftest.sh
else
  echo "skip gpu-checks/selftest.sh pause/resume check: needs Linux user systemd"
  python3 gpu-checks/runner.py --patch-selftest
  python3 gpu-checks/runner.py --diag-selftest
fi
