#!/usr/bin/env bash
# hetero6arch IPC=50：复用 all-Conv 通信包，异构 receiver 训练。

set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$BASE_DIR"

PY="/root/miniconda3/envs/sp/bin/python"
CONV_PKT="outputs/cifar100_4agent_25cls_conv_family_ipc50"
LOG_DIR="logs/hetero6arch_ipc50_allconv"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

setup_seed_dir() {
  local seed="$1"
  local suffix="$2"
  local outdir="outputs/cifar100_4agent_25cls_hetero6arch_ipc50_${suffix}"
  local expert_src="outputs/cifar100_4agent_25cls_hetero6arch_ipc10_seed${seed}"

  mkdir -p "$outdir/agents/agent_"{0,1,2,3}/{checkpoints,packets}
  for agent in 0 1 2 3; do
    cp -f "$CONV_PKT/agents/agent_${agent}/packets/dsdm_packet.pt" \
      "$outdir/agents/agent_${agent}/packets/dsdm_packet.pt"
    cp -f "$CONV_PKT/agents/agent_${agent}/packets/heuristic_packet.pt" \
      "$outdir/agents/agent_${agent}/packets/heuristic_packet.pt"
    ln -sfn "$BASE_DIR/$expert_src/agents/agent_${agent}/checkpoints/expert_model.pt" \
      "$outdir/agents/agent_${agent}/checkpoints/expert_model.pt"
    if [[ "$suffix" == *"_nologit" ]]; then
      "$PY" - "$outdir/agents/agent_${agent}/packets/dsdm_packet.pt" \
        "$outdir/agents/agent_${agent}/packets/heuristic_packet.pt" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    packet = torch.load(path, map_location="cpu", weights_only=False)
    for key in list(packet.keys()):
        if key.startswith("sender_logit") or key == "sender_logits":
            packet.pop(key, None)
    packet["has_sender_logits"] = False
    torch.save(packet, path)
PY
    fi
  done
  log "prepared $outdir from all-Conv IPC=50 packets and seed${seed} experts"
}

run_pipeline() {
  local cfg="$1"
  local stage="$2"
  local method="$3"
  local extra="${4:-}"
  local tag
  local cmd
  tag="$(basename "${cfg%.yaml}")_${stage}_${method}"
  log "start $tag"
  cmd=("$PY" run_social_pipeline.py --config "$cfg" --stage "$stage")
  if [[ -n "$method" ]]; then
    cmd+=(--packet-method "$method")
  fi
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    local extra_args=($extra)
    cmd+=("${extra_args[@]}")
  fi
  "${cmd[@]}" >> "$LOG_DIR/${tag}.log" 2>&1
  log "done  $tag"
}

prepare_seed() {
  local seed="$1"
  setup_seed_dir "$seed" "seed${seed}"
  setup_seed_dir "$seed" "seed${seed}_nologit"

  local cfg="configs/main_cifar100_hetero6arch_ipc50_seed${seed}.yaml"
  local cfg_nl="configs/main_cifar100_hetero6arch_ipc50_seed${seed}_nologit.yaml"

  run_pipeline "$cfg" attach_logits dsdm
  run_pipeline "$cfg" build_communication dsdm
  run_pipeline "$cfg" build_communication heuristic
  run_pipeline "$cfg_nl" build_communication dsdm
  run_pipeline "$cfg_nl" build_communication heuristic
}

run_seed() {
  local seed="$1"
  local cfg="configs/main_cifar100_hetero6arch_ipc50_seed${seed}.yaml"
  local cfg_nl="configs/main_cifar100_hetero6arch_ipc50_seed${seed}_nologit.yaml"

  prepare_seed "$seed"
  run_pipeline "$cfg_nl" train_receivers heuristic "--init-mode expert"
  run_pipeline "$cfg_nl" train_receivers dsdm "--init-mode expert"
  run_pipeline "$cfg" train_receivers dsdm "--init-mode expert"
}

log "===== hetero6arch IPC=50 all-Conv packet run started ====="
log "conv packet source: $CONV_PKT"

"$PY" run_social_pipeline.py \
  --config configs/main_cifar100_hetero6arch_ipc50_seed0.yaml \
  --stage build_communication \
  --packet-method dsdm \
  --dry-run >> "$LOG_DIR/dry_run_seed0.log" 2>&1
log "dry-run seed0 OK"

gpu0_job() {
  run_seed 0
  run_seed 2
}

gpu1_job() {
  run_seed 1
}

gpu0_job >> "$LOG_DIR/gpu0_master.log" 2>&1 &
PID0=$!
gpu1_job >> "$LOG_DIR/gpu1_master.log" 2>&1 &
PID1=$!

echo "$PID0" > "$LOG_DIR/gpu0.pid"
echo "$PID1" > "$LOG_DIR/gpu1.pid"

log "launched GPU0 job pid=$PID0, GPU1 job pid=$PID1"
wait "$PID0"
log "GPU0 job finished"
wait "$PID1"
log "GPU1 job finished"
log "===== hetero6arch IPC=50 all-Conv packet run finished ====="
