#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG_ROOT="$ROOT/configs/fullclass_dsdm"
LOG_ROOT="$ROOT/logs/fullclass_dsdm_model_specific_e0200_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-4}"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=4
fi

available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  echo "model-specific queue requires at least ${MIN_FREE_GIB} GiB free" >&2
  exit 2
fi
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi

printf 'phase\tmodel\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"
record_status() {
  local phase="$1" model="$2" state="$3" code="$4"
  (
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\n' "$phase" "$model" "$state" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >&9
  ) 9>>"$STATUS_FILE"
}

run_model() {
  local model="$1"
  local config="$CONFIG_ROOT/fullclass_${model}_modelbest_e0200_ipc10_seed0.yaml"
  local log="$LOG_ROOT/${model}.log"
  local code
  : > "$log"
  record_status pipeline "$model" started -
  for stage in train_experts distill_packets build_communication; do
    record_status "$stage" "$model" started -
    "$PYTHON" -u run_social_pipeline.py --config "$config" \
      --stage "$stage" --packet-method dsdm --only-agent 0 --resume >>"$log" 2>&1
    code=$?
    record_status "$stage" "$model" finished "$code"
    if [[ "$code" -ne 0 ]]; then
      record_status pipeline "$model" finished "$code"
      return "$code"
    fi
  done
  record_status validate "$model" started -
  "$PYTHON" -u validate_packets.py --config "$config" --packet-method dsdm >>"$log" 2>&1
  code=$?
  record_status validate "$model" finished "$code"
  record_status pipeline "$model" finished "$code"
  return "$code"
}

wait_group() {
  local failures=0 pid
  for pid in "$@"; do wait "$pid" || failures=1; done
  return "$failures"
}

"$PYTHON" scripts/prepare_fullclass_dsdm_model_specific.py >"$LOG_ROOT/setup.log" 2>&1 || exit $?

failures=0
pids=()
for model in alexnet resnet10_standard resnet18_standard; do
  run_model "$model" &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "$MAX_PARALLEL" ]]; then
    wait_group "${pids[@]}" || failures=1
    pids=()
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then wait_group "${pids[@]}" || failures=1; fi

if [[ "$failures" -ne 0 ]]; then
  echo "one or more model-specific full-class jobs failed" >&2
  exit 1
fi
echo "model-specific full-class DSDM queue completed successfully"
