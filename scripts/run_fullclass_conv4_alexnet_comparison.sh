#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG_ROOT="$ROOT/configs/fullclass_dsdm"
LOG_ROOT="$ROOT/logs/fullclass_dsdm_conv4_alexnet_pool_compare_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  echo "comparison queue requires at least ${MIN_FREE_GIB} GiB free" >&2
  exit 2
fi

printf 'phase\tmodel\tdesign\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"

record_status() {
  local phase="$1"
  local model="$2"
  local design="$3"
  local state="$4"
  local exit_code="$5"
  (
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$phase" "$model" "$design" "$state" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >&9
  ) 9>>"$STATUS_FILE"
}

run_job() {
  local model="$1"
  local design="$2"
  local config
  local log="$LOG_ROOT/${model}_${design}.log"
  local exit_code

  if [[ "$design" == "pure200" ]]; then
    config="$CONFIG_ROOT/fullclass_${model}_dsdmguidee0200_ipc10_seed0.yaml"
  else
    config="$CONFIG_ROOT/fullclass_${model}_trajectorye0020to0200step0020_ipc10_seed0.yaml"
  fi

  : > "$log"
  record_status pipeline "$model" "$design" started -
  for stage in train_experts distill_packets build_communication; do
    record_status "$stage" "$model" "$design" started -
    "$PYTHON" -u run_social_pipeline.py \
      --config "$config" --stage "$stage" --packet-method dsdm \
      --only-agent 0 --resume >>"$log" 2>&1
    exit_code=$?
    record_status "$stage" "$model" "$design" finished "$exit_code"
    if [[ "$exit_code" -ne 0 ]]; then
      record_status pipeline "$model" "$design" finished "$exit_code"
      return "$exit_code"
    fi
  done

  record_status validate "$model" "$design" started -
  "$PYTHON" -u validate_packets.py \
    --config "$config" --packet-method dsdm >>"$log" 2>&1
  exit_code=$?
  record_status validate "$model" "$design" finished "$exit_code"
  record_status pipeline "$model" "$design" finished "$exit_code"
  return "$exit_code"
}

wait_group() {
  local failures=0
  local pid
  for pid in "$@"; do
    wait "$pid" || failures=1
  done
  return "$failures"
}

"$PYTHON" scripts/audit_official_dsdm.py >"$LOG_ROOT/setup.log" 2>&1 || exit $?
"$PYTHON" scripts/prepare_fullclass_dsdm.py >>"$LOG_ROOT/setup.log" 2>&1 || exit $?
"$PYTHON" scripts/prepare_fullclass_conv3_trajectory.py >>"$LOG_ROOT/setup.log" 2>&1 || exit $?

jobs=(
  "conv4 pure200"
  "conv4 trajectory"
  "alexnet pure200"
  "alexnet trajectory"
)
failures=0
pids=()
for job in "${jobs[@]}"; do
  read -r model design <<<"$job"
  run_job "$model" "$design" &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "$MAX_PARALLEL" ]]; then
    wait_group "${pids[@]}" || failures=1
    pids=()
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_group "${pids[@]}" || failures=1
fi

if [[ "$failures" -ne 0 ]]; then
  echo "one or more comparison jobs failed" >&2
  exit 1
fi
echo "Conv-4/AlexNet pool comparison completed successfully"
