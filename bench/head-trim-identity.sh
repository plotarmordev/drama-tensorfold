#!/usr/bin/env bash
# Byte-identity check for recipe/trim_head.py: the NVFP4 engine serves the source checkpoint, then the trimmed one,
# one at a time under the platform's pilot with the NVFP4 guidance; every corpus take of the trimmed checkpoint must
# equal the source's byte for byte. Exits 0 when all match, 1 otherwise.
#
# Usage: bench/head-trim-identity.sh SOURCE TRIMMED OUT
#
# Environment:
#   ENGINE_DIR    engine built by scripts/build-engine.sh --weights nvfp4 (default: build/tensorfold-nvfp4)
#   RUNTIME_DIR   pilot runtime directory (default: runtime)
#   GPU_LOCK      the host's existing GPU lock file
#   PORT          loopback port (default 18793)
set -uo pipefail

(($# == 3)) || { sed -n '2,6p' "$0" >&2; exit 2; }
: "${GPU_LOCK:?}"
source=$1 trimmed=$2 out=$3
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
engine=${ENGINE_DIR:-$repo/build/tensorfold-nvfp4}
runtime=${RUNTIME_DIR:-$repo/runtime}
port=${PORT:-18793}
pilot=$repo/recipe/pilot.py
[[ $(uname -s) == Darwin ]] && pilot=$repo/recipe/pilot_macos.py
mkdir "$out" || exit 2

capture() {  # capture NAME CHECKPOINT: serve, capture the corpus once, stop
  python3 -B "$pilot" --native-bin "$engine/zig-out/native/bin/tensorfold-native" --checkpoint "$2" \
    --runtime-dir "$runtime" --gpu-lock "$GPU_LOCK" --evidence-dir "$out/pilot-$1" --port "$port" \
    -- --cfg-gamma 1.5 --cfg-batched > "$out/pilot-$1.stdout" 2>&1 &
  local pilot_pid=$!
  until [[ -f $out/pilot-$1/ready.json ]]; do
    kill -0 "$pilot_pid" 2>/dev/null || { echo "pilot $1 exited before ready" >&2; return 1; }
    sleep 2
  done
  python3 -B "$repo/bench/corpus_capture.py" "http://127.0.0.1:$port" "$out/$1"
  local status=$?
  kill -INT "$pilot_pid"
  wait "$pilot_pid" || status=1
  return $status
}

capture source "$source" && capture trimmed "$trimmed" || exit 2
python3 - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
takes = {name: {r['item']: (r['status'], r['sha256']) for r in json.loads((out / name / 'capture.json').read_text())['rows']}
         for name in ('source', 'trimmed')}
differ = sorted(i for i, take in takes['source'].items() if take[0] != 200 or take != takes['trimmed'].get(i))
report = {'items': len(takes['source']), 'identical': not differ, 'differ': differ}
(out / 'identity.json').write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps(report))
sys.exit(1 if differ else 0)
EOF
