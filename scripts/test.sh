#!/usr/bin/env bash
# Runs every test in this repository that needs no model, GPU or network; exits non-zero on any failure.
#
# Usage: scripts/test.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# The macOS lifecycle smoke uses only short-lived dummy subprocesses; on Linux it skips itself.
PILOT_MACOS_INTEGRATION=1 python3 -m unittest discover -s recipe -p 'test_*.py'
python3 -m unittest discover -s results/smaller-builds -p 'test_*.py'
python3 -m unittest discover -s results/exl3-ldlq -p 'test_*.py'
python3 -m unittest discover -s results/exl3-build -p 'test_*.py'
python3 -m unittest discover -s results/exl3-spark -p 'test_*.py'
python3 -m unittest discover -s results/capri-clean-long -p 'test_*.py'
python3 bench/stream_check.py --self-test
gpu-checks/selftest.sh
