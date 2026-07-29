#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG="$ROOT/configs/fullclass_dsdm/fullclass_alexnet_historicale0020_ipc10_seed0.yaml"
LOG_ROOT="$ROOT/logs/fullclass_alexnet_historicale0020_seed0"
LOG="$LOG_ROOT/alexnet.log"
STATUS_FILE="$LOG_ROOT/status.tsv"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=4
fi

printf 'phase\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"
record_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$3" >> "$STATUS_FILE"
}

available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((4 * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  record_status preflight failed 2
  echo "historical AlexNet recovery requires at least 4 GiB free" >&2
  exit 2
fi

"$PYTHON" scripts/prepare_fullclass_alexnet_historical_e0020.py > "$LOG_ROOT/setup.log" 2>&1 || exit $?
: > "$LOG"
record_status pipeline started -
for stage in train_experts distill_packets build_communication; do
  record_status "$stage" started -
  "$PYTHON" -u run_social_pipeline.py \
    --config "$CONFIG" --stage "$stage" --packet-method dsdm \
    --only-agent 0 --resume >> "$LOG" 2>&1
  code=$?
  record_status "$stage" finished "$code"
  if [[ "$code" -ne 0 ]]; then
    record_status pipeline finished "$code"
    exit "$code"
  fi
done

record_status validate started -
"$PYTHON" -u validate_packets.py \
  --config "$CONFIG" --packet-method dsdm >> "$LOG" 2>&1
code=$?
record_status validate finished "$code"
record_status pipeline finished "$code"
exit "$code"
