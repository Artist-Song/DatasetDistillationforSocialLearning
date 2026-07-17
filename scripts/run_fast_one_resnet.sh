#!/usr/bin/env bash
# FAST 官方 pixels/per-class/minmax 适配：只传选中的真实图和 hard labels。
set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$BASE_DIR"

PY="/root/miniconda3/envs/sp/bin/python"
LOG_DIR="logs/fast_one_resnet"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*"
}

run_cmd() {
  local tag="$1"
  shift
  log "start ${tag}"
  "$@" >> "${LOG_DIR}/${tag}.log" 2>&1
  log "done  ${tag}"
}

run_fast() {
  local seed="$1"
  local ipc="$2"
  local cfg="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}_fast.yaml"

  run_cmd "seed${seed}_ipc${ipc}_build_fast" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage build_selection_packets --packet-method fast
  run_cmd "seed${seed}_ipc${ipc}_comm_fast" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage build_communication --packet-method fast
  run_cmd "seed${seed}_ipc${ipc}_validate_fast" \
    "$PY" validate_packets.py --config "$cfg" --packet-method fast
  run_cmd "seed${seed}_ipc${ipc}_recv_fast" \
    "$PY" run_social_pipeline.py --config "$cfg" --stage train_receivers --packet-method fast --init-mode expert
}

log "===== FAST one-ResNet experiments started ====="
run_cmd "prepare_fast" "$PY" scripts/prepare_fast_one_resnet.py

for ipc in 10 50; do
  for seed in 0 1 2; do
    run_fast "$seed" "$ipc"
  done
done

log "===== FAST one-ResNet experiments finished ====="
