#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
DEPENDENCY_PID="${DEPENDENCY_PID:-194128}"
DEPENDENCY_STATUS="$ROOT/logs/fullclass_dsdm_conv4_alexnet_pool_compare_seed0/status.tsv"
LOG_ROOT="$ROOT/logs/fullclass_slice_pat5_conv3_comparison_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"
RUN_LOG="$LOG_ROOT/run.log"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if ! [[ "$DEPENDENCY_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEPENDENCY_PID must be a positive integer" >&2
  exit 2
fi

printf 'phase\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"

record_status() {
  printf '%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$3" >>"$STATUS_FILE"
}

conv4_pair_complete() {
  [[ -f "$DEPENDENCY_STATUS" ]] || return 1
  awk -F '\t' '
    $1 == "pipeline" && $2 == "conv4" && $4 == "finished" && $6 == "0" {
      complete[$3] = 1
    }
    END { exit !(complete["pure200"] && complete["trajectory"]) }
  ' "$DEPENDENCY_STATUS"
}

record_status dependency_wait started -
while ! conv4_pair_complete; do
  if ! kill -0 "$DEPENDENCY_PID" 2>/dev/null; then
    echo "Conv-4 dependency exited without two successful pipeline completions" >>"$RUN_LOG"
    record_status dependency_wait finished 2
    exit 2
  fi
  sleep 30
done
record_status dependency_wait finished 0

record_status comparison started -
"$PYTHON" -u scripts/compare_fullclass_slice_pat5_conv3.py --seed 0 >>"$RUN_LOG" 2>&1
exit_code=$?
record_status comparison finished "$exit_code"
exit "$exit_code"
