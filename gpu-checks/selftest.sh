#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
export GPU_CHECK_OUT
GPU_CHECK_OUT=$(mktemp -d)
export PAUSE_UNITS=gpuchk-selftest-sleep
created=0
cleanup() {
  if [[ $created == 1 ]]; then
    "$here/resume-paused.sh"
    systemctl --user stop gpuchk-selftest-sleep
  fi
}
trap cleanup EXIT
if systemctl --user show -p LoadState --value gpuchk-selftest-sleep | grep -qx loaded; then
  echo 'Refusing to touch a pre-existing dummy unit' >&2
  exit 1
fi
python3 "$here/runner.py" --patch-selftest
python3 "$here/runner.py" --segments-selftest
python3 "$here/runner.py" --diag-selftest
# run-diag.sh exit mapping, through isolated shims (no runner, no units): 75 -> 0, other codes kept.
shim=$(mktemp -d)
cp "$here/run-diag.sh" "$shim/run-diag.sh"
printf 'import os, sys\nsys.exit(int(os.environ["FAKE_RC"]))\n' >"$shim/runner.py"
printf '#!/usr/bin/env bash\ntrue\n' >"$shim/resume-paused.sh"
chmod +x "$shim/resume-paused.sh"
for pair in 75:0 1:1 2:2 0:0 143:143; do
  set +e
  FAKE_RC=${pair%%:*} bash "$shim/run-diag.sh" --out-dir "$shim/out" 2>/dev/null
  got=$?
  set -e
  echo "wrapper rc ${pair%%:*} -> $got"
  [[ $got == "${pair##*:}" ]]
done
rm -r "$shim"
systemd-run --user --unit=gpuchk-selftest-sleep sleep 600
created=1
pid=$(systemctl --user show -p MainPID --value gpuchk-selftest-sleep)
python3 "$here/runner.py" --pause-selftest
sleep 1
state=$(ps -o state= -p "$pid")
echo "paused state=$state"
[[ $state == T ]]
"$here/resume-paused.sh"
sleep 1
state=$(ps -o state= -p "$pid")
echo "resumed state=$state"
[[ $state == S ]]
"$here/resume-paused.sh"
echo "selftest evidence: $GPU_CHECK_OUT"
