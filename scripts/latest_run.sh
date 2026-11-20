#!/usr/bin/env bash
# Print the directory of the latest completed-or-not run of a task: latest_run.sh TASK [large]
# (small outputs under $RUNS_ROOT/runs, stores under $RUNS_ROOT/large/runs).
set -euo pipefail
root="${RUNS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/runs}"
[ "${2:-}" = large ] && root="$root/large"
d=$(ls -d "$root"/runs/*_"$1"_* 2>/dev/null | sort | tail -1 || true)
[ -n "$d" ] || { echo "no run of task '$1' under $root/runs: run its recipe first" >&2; exit 1; }
echo "$d"
