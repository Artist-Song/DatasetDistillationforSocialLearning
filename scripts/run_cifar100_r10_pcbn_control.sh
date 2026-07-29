#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG="$ROOT/configs/fullclass_dsdm/fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml"
LOG_ROOT="$ROOT/logs/cifar100_r10_pcbn_control_seed0"
LOG="$LOG_ROOT/resnet10_standard_pcbn.log"
STATUS_FILE="$LOG_ROOT/status.tsv"
WAIT_FOR_PID="${WAIT_FOR_PID:-284624}"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=4; fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export MKL_NUM_THREADS=4; fi

printf 'phase\tstate\ttimestamp_utc\texit_code\tdetail\n' > "$STATUS_FILE"
record_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$3" "$4" >> "$STATUS_FILE"
}

available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((3 * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  record_status preflight failed 2 "less_than_3_GiB_free"
  echo "R10 PCBN control requires at least 3 GiB free" >&2
  exit 2
fi

"$PYTHON" scripts/prepare_cifar100_r10_pcbn_control.py > "$LOG_ROOT/setup.log" 2>&1 || exit $?
"$PYTHON" scripts/validate_cifar100_r10_pcbn_pair.py > "$LOG_ROOT/preflight.json" 2>&1 || exit $?

if [[ "$WAIT_FOR_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$WAIT_FOR_PID" 2>/dev/null; then
  record_status queue waiting - "pid=$WAIT_FOR_PID"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

: > "$LOG"
record_status pipeline started - "pure_R10_already_complete"
for stage in distill_packets build_communication; do
  record_status "$stage" started - -
  "$PYTHON" -u run_social_pipeline.py \
    --config "$CONFIG" --stage "$stage" --packet-method dsdm \
    --only-agent 0 --resume >> "$LOG" 2>&1
  code=$?
  record_status "$stage" finished "$code" -
  if [[ "$code" -ne 0 ]]; then
    record_status pipeline finished "$code" -
    exit "$code"
  fi
done

record_status validate started - -
"$PYTHON" -u validate_packets.py --config "$CONFIG" --packet-method dsdm >> "$LOG" 2>&1
code=$?
record_status validate finished "$code" -
record_status pipeline finished "$code" -
exit "$code"
