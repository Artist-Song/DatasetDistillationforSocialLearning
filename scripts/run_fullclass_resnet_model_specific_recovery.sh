#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG_ROOT="$ROOT/configs/fullclass_dsdm"
LOG_ROOT="$ROOT/logs/fullclass_resnet_model_specific_recovery_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=4; fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export MKL_NUM_THREADS=4; fi

printf 'phase\tmodel\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"
record_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" >> "$STATUS_FILE"
}

available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((3 * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  record_status preflight disk failed 2
  echo "ResNet recovery requires at least 3 GiB free" >&2
  exit 2
fi

"$PYTHON" scripts/prepare_fullclass_resnet_model_specific_recovery.py > "$LOG_ROOT/setup.log" 2>&1 || exit $?

run_stage() {
  local model="$1" stage="$2" config="$3" log="$4" code
  record_status "$stage" "$model" started -
  "$PYTHON" -u run_social_pipeline.py \
    --config "$config" --stage "$stage" --packet-method dsdm \
    --only-agent 0 --resume >> "$log" 2>&1
  code=$?
  record_status "$stage" "$model" finished "$code"
  return "$code"
}

run_model() {
  local model="$1"
  local config="$CONFIG_ROOT/fullclass_${model}_modelbest_e0200_ipc10_seed0_recovery.yaml"
  local log="$LOG_ROOT/${model}.log"
  local stages=()
  local stage code
  : > "$log"
  record_status pipeline "$model" started -
  if [[ "$model" == "resnet10_standard" ]]; then
    stages=(distill_packets build_communication)
  else
    stages=(train_experts distill_packets build_communication)
  fi
  for stage in "${stages[@]}"; do
    run_stage "$model" "$stage" "$config" "$log"
    code=$?
    if [[ "$code" -ne 0 ]]; then
      record_status pipeline "$model" finished "$code"
      return "$code"
    fi
  done
  record_status validate "$model" started -
  "$PYTHON" -u validate_packets.py --config "$config" --packet-method dsdm >> "$log" 2>&1
  code=$?
  record_status validate "$model" finished "$code"
  record_status pipeline "$model" finished "$code"
  return "$code"
}

run_model resnet10_standard || exit $?
run_model resnet18_standard || exit $?
