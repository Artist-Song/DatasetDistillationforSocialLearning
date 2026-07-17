#!/usr/bin/env bash
# Run the corrected DeSA-CIL and complete MASC baselines on one GPU.
set -euo pipefail

ROOT="/root/autodl-tmp/DatasetDistillationforSocialLearning"
PY="/root/miniconda3/envs/sp/bin/python"
LOG_DIR="$ROOT/external_baselines/logs/masc_desa_complete"
WAIT_PID="${WAIT_PID:-}"
PARENT_TO_RESUME="${PARENT_TO_RESUME:-}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*"
}

resume_parent() {
  if [[ -n "$PARENT_TO_RESUME" ]] && kill -0 "$PARENT_TO_RESUME" 2>/dev/null; then
    kill -CONT "$PARENT_TO_RESUME" || true
    log "resumed parent queue pid=$PARENT_TO_RESUME"
  fi
}
trap resume_parent EXIT

wait_for_process() {
  local pid="$1"
  while true; do
    local state
    state="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
      break
    fi
    log "waiting for current GPU job pid=$pid state=$state"
    sleep 60
  done
}

run_logged() {
  local name="$1"
  shift
  log "start $name"
  "$@" > >(tee -a "$LOG_DIR/${name}.log") 2>&1
  log "done $name"
}

if [[ -n "$WAIT_PID" ]]; then
  wait_for_process "$WAIT_PID"
fi

run_desa_branch() {
  run_logged "desa_cil_seed0_ipc10" \
    "$PY" baseline_adapters/run_desa_cil.py \
    --config configs/main_cifar100_one_resnet_seed0_ipc10.yaml \
    --ipc 10 \
    --resume

  run_logged "desa_cil_seed0_ipc50" \
    "$PY" baseline_adapters/run_desa_cil.py \
    --config configs/main_cifar100_one_resnet_seed0_ipc50.yaml \
    --ipc 50 \
    --resume
}

run_masc_branch() {
  # Full data is deliberately first. The adapter aborts before IPC runs if the
  # full-data reproduction fails its expert/global/new-knowledge sanity checks.
  run_logged "masc_complete_seed0" \
    "$PY" baseline_adapters/run_masc_complete.py \
    --seed 0 \
    --budget full 10 50
}

log "launching DeSA-CIL and MASC branches in parallel"
set +e
run_desa_branch &
desa_pid=$!
run_masc_branch &
masc_pid=$!
wait "$desa_pid"
desa_status=$?
wait "$masc_pid"
masc_status=$?
set -e

if [[ "$desa_status" -ne 0 || "$masc_status" -ne 0 ]]; then
  log "baseline failure: DeSA-CIL status=$desa_status MASC status=$masc_status"
  exit 1
fi

log "all corrected seed0 baselines completed"
