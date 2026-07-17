#!/usr/bin/env bash
# one-ResNet 主实验队列：每个 seed 恰好一个 ResNet，其余为 Conv3/Conv4/AlexNet。
set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$BASE_DIR"

PY="/root/miniconda3/envs/sp/bin/python"
LOG_DIR="logs/one_resnet_main"
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

pipeline() {
  local tag="$1"
  local cfg="$2"
  local stage="$3"
  local method="$4"
  shift 4
  local cmd=("$PY" run_social_pipeline.py --config "$cfg" --stage "$stage")
  if [[ -n "$method" ]]; then
    cmd+=(--packet-method "$method")
  fi
  cmd+=("$@")
  run_cmd "$tag" "${cmd[@]}"
}

prepare_main() {
  local seed="$1"
  local ipc="$2"
  run_cmd "prepare_seed${seed}_ipc${ipc}" \
    "$PY" scripts/prepare_one_resnet_main.py --generate-configs --prepare --full-real --seeds "$seed" --ipcs "$ipc"
}

run_social_methods() {
  local seed="$1"
  local ipc="$2"
  local cfg="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}.yaml"
  local cfg_fr="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}_fullreal.yaml"

  pipeline "seed${seed}_ipc${ipc}_attach_dsdm" "$cfg" attach_logits dsdm
  pipeline "seed${seed}_ipc${ipc}_comm_dsdm" "$cfg" build_communication dsdm
  run_cmd "seed${seed}_ipc${ipc}_validate_dsdm" "$PY" validate_packets.py --config "$cfg" --packet-method dsdm
  pipeline "seed${seed}_ipc${ipc}_recv_dsdm" "$cfg" train_receivers dsdm --init-mode expert

  pipeline "seed${seed}_ipc${ipc}_build_heuristic" "$cfg" build_selection_packets heuristic
  pipeline "seed${seed}_ipc${ipc}_attach_heuristic" "$cfg" attach_logits heuristic
  pipeline "seed${seed}_ipc${ipc}_comm_heuristic" "$cfg" build_communication heuristic
  run_cmd "seed${seed}_ipc${ipc}_validate_heuristic" "$PY" validate_packets.py --config "$cfg" --packet-method heuristic
  pipeline "seed${seed}_ipc${ipc}_recv_heuristic" "$cfg" train_receivers heuristic --init-mode expert

  run_cmd "prepare_seed${seed}_ipc${ipc}_fullreal_refresh" \
    "$PY" scripts/prepare_one_resnet_main.py --prepare --full-real --seeds "$seed" --ipcs "$ipc"
  pipeline "seed${seed}_ipc${ipc}_build_fullreal" "$cfg_fr" build_selection_packets full_real
  pipeline "seed${seed}_ipc${ipc}_comm_fullreal" "$cfg_fr" build_communication full_real
  run_cmd "seed${seed}_ipc${ipc}_validate_fullreal" "$PY" validate_packets.py --config "$cfg_fr" --packet-method full_real
  pipeline "seed${seed}_ipc${ipc}_recv_fullreal" "$cfg_fr" train_receivers full_real --init-mode expert
}

distill_agents() {
  local seed="$1"
  local ipc="$2"
  shift 2
  local cfg="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}.yaml"
  for agent in "$@"; do
    pipeline "seed${seed}_ipc${ipc}_distill_agent${agent}" "$cfg" distill_packets dsdm --only-agent "$agent" --resume
  done
}

train_agents() {
  local seed="$1"
  local ipc="$2"
  shift 2
  local cfg="configs/main_cifar100_one_resnet_seed${seed}_ipc${ipc}.yaml"
  for agent in "$@"; do
    pipeline "seed${seed}_ipc${ipc}_train_agent${agent}" "$cfg" train_experts dsdm --only-agent "$agent" --resume
  done
}

run_external_baselines() {
  log "external baselines queued after project runs"
  for seed in 1 2; do
    run_cmd "masc_seed${seed}_ipc10_ipc50" \
      "$PY" baseline_adapters/run_masc_official_ipc.py --ipc 10 50 --seed "$seed"
  done

  # DeSA official-minimal runner keeps the original CIFAR100 4-agent setup.
  for seed in 1 2; do
    for ipc in 10 50; do
      run_cmd "desa_official_seed${seed}_ipc${ipc}" \
        "$PY" external_baselines/repos/DESA/iterative_desa.py \
        --dataset cifar100_4agent \
        --save_path "external_baselines/outputs/desa_official_seed${seed}" \
        --ipc "$ipc" \
        --seed "$seed" \
        --batch 128 \
        --kd_batch 128 \
        --iters 100 \
        --inv_iters 1000 \
        --kd_iters 100 \
        --lr_net 0.01 \
        --lr_kd 0.01 \
        --lr_img 0.05 \
        --pretrain True \
        --generate_image True \
        --kd True \
        --kd_from_scratch True \
        --model_hetero True
    done
  done
}

log "===== one-ResNet main queue started ====="
run_cmd "py_compile_one_resnet" "$PY" -m py_compile scripts/prepare_one_resnet_main.py run_social_pipeline.py

# seed0/ipc10 已有等价结果，但这里复制到统一 run_name 后只训练 receiver/FullReal。
prepare_main 0 10
run_social_methods 0 10

# seed2 复用最多，只需重蒸馏 ResNet18(classes25-49) 和 Conv3(classes50-74)。
prepare_main 2 10
distill_agents 2 10 1 2
run_social_methods 2 10

# seed1 需要先补 Conv3(classes75-99) guide，再蒸馏四个 sender。
prepare_main 1 10
train_agents 1 10 3
distill_agents 1 10 0 1 2 3
run_social_methods 1 10

# IPC=50 按同样 seed 顺序排队；seed1 的 Conv3 guide 会从 IPC=10 复用。
prepare_main 0 50
distill_agents 0 50 2
run_social_methods 0 50

prepare_main 2 50
distill_agents 2 50 1 2
run_social_methods 2 50

prepare_main 1 50
distill_agents 1 50 0 1 2 3
run_social_methods 1 50

run_external_baselines

log "===== one-ResNet main queue finished ====="
