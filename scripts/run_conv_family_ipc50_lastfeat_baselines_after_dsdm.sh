#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/DatasetDistillationforSocialLearning

PY=/root/miniconda3/envs/sp/bin/python
CFG=configs/main_cifar100_conv_family_ipc50_lastfeat_nologit.yaml
LOG_DIR=logs/conv_family_ipc50_lastfeat
LOG="$LOG_DIR/baselines_after_dsdm_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

{
  echo "[start] $(date '+%F %T %Z')"
  echo "[config] $CFG"
  echo "[wait] waiting for last-feature DSDM queue to finish"
  while pgrep -f "scripts/run_conv_family_ipc50_lastfeat_queue.sh" >/dev/null; do
    date '+[wait] %F %T %Z DSDM queue still running'
    sleep 600
  done

  latest_dsdm_log=$(ls -1t "$LOG_DIR"/run_*.log 2>/dev/null | head -1 || true)
  echo "[check] latest_dsdm_log=$latest_dsdm_log"
  if [ -z "$latest_dsdm_log" ] || ! grep -q '^\[done\]' "$latest_dsdm_log"; then
    echo "[error] DSDM queue did not finish cleanly; skip baselines"
    exit 1
  fi

  for method in full_real heuristic; do
    echo "[stage] build_selection_packets $method"
    "$PY" run_social_pipeline.py --config "$CFG" --stage build_selection_packets --packet-method "$method"
    echo "[stage] build_communication $method"
    "$PY" run_social_pipeline.py --config "$CFG" --stage build_communication --packet-method "$method"
    echo "[stage] validate_packets $method"
    "$PY" validate_packets.py --config "$CFG" --packet-method "$method"
    echo "[stage] train_receivers $method"
    "$PY" run_social_pipeline.py --config "$CFG" --stage train_receivers --packet-method "$method" --init-mode expert
  done

  echo "[stage] update registry"
  "$PY" scripts/update_experiment_registry.py
  echo "[done] $(date '+%F %T %Z')"
} 2>&1 | tee "$LOG"
