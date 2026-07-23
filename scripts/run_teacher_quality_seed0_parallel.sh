#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG_ROOT="configs/teacher_quality"
LOG_ROOT="$ROOT/logs/teacher_quality_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"
MAX_DSDM_PARALLEL="${MAX_DSDM_PARALLEL:-3}"
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if ! [[ "$MAX_DSDM_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_DSDM_PARALLEL must be a positive integer" >&2
  exit 2
fi
available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  echo "teacher-quality queue requires at least ${MIN_FREE_GIB} GiB free" >&2
  exit 2
fi

printf 'phase\texperiment_id\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_FILE"

record_status() {
  local phase="$1"
  local experiment_id="$2"
  local state="$3"
  local exit_code="$4"
  (
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$phase" "$experiment_id" "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >&9
  ) 9>>"$STATUS_FILE"
}

run_pipeline() {
  local phase="$1"
  local experiment_id="$2"
  local config_path="$3"
  local stage="$4"
  local agent_id="$5"
  local log_path="$LOG_ROOT/${experiment_id}.log"
  local exit_code

  record_status "$phase" "$experiment_id" started -
  "$PYTHON" -u run_social_pipeline.py \
    --config "$config_path" \
    --stage "$stage" \
    --packet-method dsdm \
    --only-agent "$agent_id" \
    --resume \
    >"$log_path" 2>&1
  exit_code=$?
  record_status "$phase" "$experiment_id" finished "$exit_code"
  return "$exit_code"
}

run_candidate() {
  local config_path="$1"
  local agent_id="$2"
  local basename
  local experiment_id
  local exit_code

  basename="$(basename "$config_path" .yaml)"
  experiment_id="${basename#packet_}"
  run_pipeline distill "$experiment_id" "$config_path" distill_packets "$agent_id"
  exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    return "$exit_code"
  fi
  run_pipeline logits "${experiment_id}_logits" "$config_path" attach_logits "$agent_id"
}

agent_for_config() {
  case "$(basename "$1")" in
    *conv3*) printf '0\n' ;;
    *conv4*) printf '1\n' ;;
    *alexnet*) printf '2\n' ;;
    *resnet10_standard*) printf '3\n' ;;
    *resnet18_standard*) printf '4\n' ;;
    *) return 1 ;;
  esac
}

wait_group() {
  local failures=0
  local pid
  for pid in "$@"; do
    wait "$pid" || failures=1
  done
  return "$failures"
}

"$PYTHON" scripts/prepare_teacher_quality_protocol.py

training_names=(conv3 conv4 alexnet resnet10_standard resnet18_standard)
training_agents=(0 1 2 3 4)
training_pids=()
for index in "${!training_names[@]}"; do
  name="${training_names[$index]}"
  agent_id="${training_agents[$index]}"
  run_pipeline \
    teacher_training \
    "train_${name}_seed0" \
    "$CONFIG_ROOT/train_${name}_seed0.yaml" \
    train_experts \
    "$agent_id" &
  training_pids+=("$!")
done
if ! wait_group "${training_pids[@]}"; then
  echo "teacher training failed; packet jobs were not started" >&2
  exit 1
fi

candidate_pids=()
candidate_failure=0
while IFS= read -r config_path; do
  agent_id="$(agent_for_config "$config_path")" || exit 1
  run_candidate "$config_path" "$agent_id" &
  candidate_pids+=("$!")
  if [[ "${#candidate_pids[@]}" -ge "$MAX_DSDM_PARALLEL" ]]; then
    if ! wait_group "${candidate_pids[@]}"; then
      echo "one or more packet-quality jobs failed" >&2
      candidate_failure=1
    fi
    candidate_pids=()
  fi
done < <(find "$CONFIG_ROOT" -maxdepth 1 -type f -name 'packet_*_seed0_ipc10.yaml' | sort)

if [[ "${#candidate_pids[@]}" -gt 0 ]]; then
  if ! wait_group "${candidate_pids[@]}"; then
    echo "one or more packet-quality jobs failed" >&2
    candidate_failure=1
  fi
fi

"$PYTHON" scripts/summarize_teacher_quality.py \
  >"$LOG_ROOT/quality_summary.log" 2>&1
summary_exit=$?
if [[ "$candidate_failure" -ne 0 || "$summary_exit" -ne 0 ]]; then
  exit 1
fi
