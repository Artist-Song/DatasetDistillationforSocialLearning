#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
LOG_ROOT="$ROOT/logs/pat_class_split_seed0"
STATUS_FILE="$LOG_ROOT/launcher_status.tsv"

mkdir -p "$LOG_ROOT"
cd "$ROOT"
export OMP_NUM_THREADS=4

run_one() {
  local experiment_id="$1"
  local config_path="$2"
  local log_path="$LOG_ROOT/${experiment_id}.log"
  local started_at
  local finished_at
  local exit_code

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\tstarted\t%s\t%s\t-\n' "$experiment_id" "$started_at" "$config_path" >> "$STATUS_FILE"
  "$PYTHON" -u run_social_pipeline.py \
    --config "$config_path" \
    --stage all \
    --packet-method dsdm \
    --init-mode expert \
    --resume \
    > >(tee -a "$log_path") 2>&1
  exit_code=$?
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\tfinished\t%s\t%s\t%s\n' \
    "$experiment_id" "$finished_at" "$config_path" "$exit_code" >> "$STATUS_FILE"
  return "$exit_code"
}

printf 'experiment_id\tstate\ttimestamp_utc\tconfig\texit_code\n' > "$STATUS_FILE"

run_one \
  pat5agent_seed0_ipc10 \
  configs/pat_class_split/main_cifar100_pat5agent_seed0_ipc10.yaml &
pid_pat5=$!

run_one \
  pat10agent_seed0_ipc10 \
  configs/pat_class_split/main_cifar100_pat10agent_seed0_ipc10.yaml &
pid_pat10=$!

printf '%s\n' "$pid_pat5" > "$LOG_ROOT/pat5.pid"
printf '%s\n' "$pid_pat10" > "$LOG_ROOT/pat10.pid"

wait "$pid_pat5"
exit_pat5=$?
wait "$pid_pat10"
exit_pat10=$?

if [[ "$exit_pat5" -ne 0 || "$exit_pat10" -ne 0 ]]; then
  exit 1
fi
