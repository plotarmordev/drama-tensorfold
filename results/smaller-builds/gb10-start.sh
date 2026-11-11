#!/usr/bin/env bash
# Starts a GB10 job on BOTTOM through the TOP menu once a GPU is free, and never stops DeepSeek: a GPU job runs only in
# a window someone else opened, so while there is none (or the door is busy) it asks again every POLL seconds.
#
# Usage: results/smaller-builds/gb10-start.sh SHA SCRIPT
#   SHA     the pushed 40-hex commit of plotarmordev/drama-tensorfold the job runs
#   SCRIPT  the job script at that commit, e.g. results/smaller-builds/gb10-int3.sh
#
# Environment: POLL seconds between tries (default 300). Prints the door's `started …` line and exits 0, or its
# refusal and exits 1.
set -uo pipefail

(($# == 2)) || { sed -n '5,7p' "$0" >&2; exit 2; }
while true; do
  reply=$(ssh top-menu "job start gpu bottom plotarmordev/drama-tensorfold $1 $2" 2>&1)
  case $reply in
    *started*) echo "$reply"; exit 0 ;;
    busy:* | *'deepseek pause'*) echo "$(date +%T) waiting for a free GPU: $reply" >&2; sleep "${POLL:-300}" ;;
    *) echo "$reply" >&2; exit 1 ;;
  esac
done
