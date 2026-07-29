#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG="$ROOT/configs/fullclass_dsdm/fullclass_conv3_trajectorye0020to0200step0020_ipc10_seed0.yaml"
LOG_ROOT="$ROOT/logs/fullclass_dsdm_conv3_trajectorye0020to0200step0020_seed0"
LOG_FILE="$LOG_ROOT/run.log"
STATUS_FILE="$LOG_ROOT/status.tsv"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

record_status() {
  local stage="$1"
  local state="$2"
  local exit_code="$3"
  printf '%s\t%s\t%s\t%s\n' \
    "$stage" "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >> "$STATUS_FILE"
}

run_stage() {
  local stage="$1"
  local exit_code
  record_status "$stage" started -
  "$PYTHON" -u run_social_pipeline.py \
    --config "$CONFIG" --stage "$stage" --packet-method dsdm \
    --only-agent 0 --resume >> "$LOG_FILE" 2>&1
  exit_code=$?
  record_status "$stage" finished "$exit_code"
  return "$exit_code"
}

printf 'stage\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"
: > "$LOG_FILE"

"$PYTHON" scripts/prepare_fullclass_conv3_trajectory.py >> "$LOG_FILE" 2>&1 || exit $?
"$PYTHON" scripts/audit_official_dsdm.py >> "$LOG_FILE" 2>&1 || exit $?

run_stage train_experts || exit $?
run_stage distill_packets || exit $?
run_stage build_communication || exit $?

record_status validate started -
"$PYTHON" -u validate_packets.py --config "$CONFIG" --packet-method dsdm >> "$LOG_FILE" 2>&1
exit_code=$?
record_status validate finished "$exit_code"
exit "$exit_code"
