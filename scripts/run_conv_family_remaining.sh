#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DatasetDistillationforSocialLearning
PY=/root/miniconda3/envs/sp/bin/python
CFG=configs/main_cifar100_conv_family_ipc10.yaml
RUN_DIR=outputs/cifar100_4agent_25cls_conv_family_ipc10
LOG_DIR=logs/conv_family_ipc10
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
run_step() {
  local name="$1"
  shift
  echo "[$(ts)] START $name"
  "$@"
  echo "[$(ts)] DONE $name"
}
wait_for_file() {
  local path="$1"
  local label="$2"
  echo "[$(ts)] WAIT $label -> $path"
  local waited=0
  until [ -f "$path" ]; do
    sleep 120
    waited=$((waited + 120))
    echo "[$(ts)] STILL WAITING $label (${waited}s)"
  done
  echo "[$(ts)] FOUND $label"
}

# convnet3w1 is currently running in the active session. Continue after it records metrics.
wait_for_file "$RUN_DIR/centralized_full/convnet3w1_conv_family_dsdm_strict/metrics.json" "centralized convnet3w1"

if [ ! -f "$RUN_DIR/centralized_full/convnet4w1_conv_family_dsdm_strict/metrics.json" ]; then
  run_step "centralized convnet4w1" "$PY" run_centralized_full.py --config "$CFG" --model convnet4w1
fi
if [ ! -f "$RUN_DIR/centralized_full/convnet4w15_conv_family_dsdm_strict/metrics.json" ]; then
  run_step "centralized convnet4w15" "$PY" run_centralized_full.py --config "$CFG" --model convnet4w15
fi

# Expert checkpoints are required by expert-init receiver training.
run_step "train_experts" "$PY" run_social_pipeline.py --config "$CFG" --stage train_experts --packet-method dsdm --resume

# Image-only baselines use a no-logit config; only DSDM_LOGIT enables logits.
NOLOGIT_CFG=/tmp/main_cifar100_conv_family_ipc10_nologit.yaml
"$PY" - "$CFG" "$NOLOGIT_CFG" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("logits", {})["enabled"] = False
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

# Full Real Social Transfer
run_step "full_real build_selection_packets" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage build_selection_packets --packet-method full_real
run_step "full_real build_communication" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage build_communication --packet-method full_real
run_step "full_real validate_packets" "$PY" validate_packets.py --config "$NOLOGIT_CFG" --packet-method full_real
run_step "full_real train_receivers" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage train_receivers --packet-method full_real --init-mode expert

# Heuristic IPC=10 baseline
run_step "heuristic build_selection_packets" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage build_selection_packets --packet-method heuristic
run_step "heuristic build_communication" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage build_communication --packet-method heuristic
run_step "heuristic validate_packets" "$PY" validate_packets.py --config "$NOLOGIT_CFG" --packet-method heuristic
run_step "heuristic train_receivers" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage train_receivers --packet-method heuristic --init-mode expert

# DSDM image-only.
run_step "dsdm train_experts" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage train_experts --packet-method dsdm --resume
run_step "dsdm distill_packets" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage distill_packets --packet-method dsdm
run_step "dsdm build_communication" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage build_communication --packet-method dsdm
run_step "dsdm validate_packets" "$PY" validate_packets.py --config "$NOLOGIT_CFG" --packet-method dsdm
run_step "dsdm train_receivers" "$PY" run_social_pipeline.py --config "$NOLOGIT_CFG" --stage train_receivers --packet-method dsdm --init-mode expert

# DSDM + logits
run_step "dsdm_logits attach_logits" "$PY" run_social_pipeline.py --config "$CFG" --stage attach_logits --packet-method dsdm
run_step "dsdm_logits build_communication" "$PY" run_social_pipeline.py --config "$CFG" --stage build_communication --packet-method dsdm
run_step "dsdm_logits validate_packets" "$PY" validate_packets.py --config "$CFG" --packet-method dsdm
run_step "dsdm_logits train_receivers" "$PY" run_social_pipeline.py --config "$CFG" --stage train_receivers --packet-method dsdm --init-mode expert

echo "[$(ts)] ALL DONE"
