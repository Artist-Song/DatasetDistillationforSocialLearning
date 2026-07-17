#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
PY="${PYTHON:-/root/miniconda3/envs/sp/bin/python}"
LOG_DIR="$BASE_DIR/logs/heuristic_hard_one_resnet"

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*"
}

run_cmd() {
  local tag="$1"
  shift
  log "start ${tag}"
  "$@" >> "$LOG_DIR/${tag}.log" 2>&1
  log "done  ${tag}"
}

run_group() {
  local seed="$1"
  local ipc="$2"
  local cfg="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}_heuristic_hard.yaml"

  run_cmd "seed${seed}_ipc${ipc}_build" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage build_selection_packets --packet-method heuristic
  run_cmd "seed${seed}_ipc${ipc}_comm" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage build_communication --packet-method heuristic
  run_cmd "seed${seed}_ipc${ipc}_validate" \
    "$PY" validate_packets.py --config "$cfg" --packet-method heuristic
  run_cmd "seed${seed}_ipc${ipc}_recv" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage train_receivers --packet-method heuristic --init-mode expert
}

run_cmd "prepare" "$PY" scripts/prepare_heuristic_hard_one_resnet.py
for ipc in 10 50; do
  for seed in 0 1 2; do
    run_group "$seed" "$ipc"
  done
done
log "all heuristic hard-label runs finished"
