#!/bin/bash
# hetero6arch 全实验主控脚本
# GPU0: seed0(重跑) → seed2
# GPU1: seed1(expert训练 + 实验)
# 目标：16:40 ~ 20:00 跑完全部3个seed × 3个方法

set -e
BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$BASE_DIR"
mkdir -p logs/hetero6arch

PYTHON="conda run -n sp python"
CONV_PKT="outputs/cifar100_4agent_25cls_conv_family_ipc10"

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
setup_seed_dir() {
    local NAME="$1"
    local OUTDIR="outputs/cifar100_4agent_25cls_hetero6arch_ipc10_${NAME}"
    mkdir -p "$OUTDIR/agents/agent_"{0,1,2,3}/{checkpoints,packets}
    for a in 0 1 2 3; do
        for pkt in dsdm_packet.pt heuristic_packet.pt; do
            local LINK="$OUTDIR/agents/agent_${a}/packets/${pkt}"
            [ -L "$LINK" ] || ln -sf "$BASE_DIR/$CONV_PKT/agents/agent_${a}/packets/${pkt}" "$LINK"
        done
    done
    echo "[setup] $OUTDIR OK"
}

link_experts_from() {
    local SRC_NAME="$1"   # e.g. seed1
    local DST_NAME="$2"   # e.g. seed1_nologit
    local SRC="outputs/cifar100_4agent_25cls_hetero6arch_ipc10_${SRC_NAME}"
    local DST="outputs/cifar100_4agent_25cls_hetero6arch_ipc10_${DST_NAME}"
    for a in 0 1 2 3; do
        local LINK="$DST/agents/agent_${a}/checkpoints/expert_model.pt"
        [ -L "$LINK" ] || ln -sf "$BASE_DIR/$SRC/agents/agent_${a}/checkpoints/expert_model.pt" "$LINK"
    done
}

run_pipeline() {
    local GPU="$1"; local CFG="$2"; local STAGE="$3"; local METHOD="$4"
    local EXTRA="${5:-}"
    local LOG="logs/hetero6arch/$(basename ${CFG%.yaml})_${STAGE}_${METHOD}.log"
    echo "[GPU$GPU] $STAGE $METHOD $(basename $CFG) ..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON run_social_pipeline.py \
        --config "$CFG" --stage "$STAGE" \
        ${METHOD:+--packet-method "$METHOD"} \
        ${EXTRA} >> "$LOG" 2>&1
}

# ──────────────────────────────────────────────
# GPU0 流程：seed0重跑 → seed2
# ──────────────────────────────────────────────
gpu0_job() {
    echo "====== GPU0: seed0 重跑 ======"
    local S0="configs/main_cifar100_hetero6arch_ipc10_seed0.yaml"
    local S0NL="configs/main_cifar100_hetero6arch_ipc10_seed0_nologit.yaml"

    run_pipeline 0 "$S0NL" train_receivers heuristic "--init-mode expert"
    echo "GPU0: seed0 Heuristic done $(date +%H:%M:%S)"

    run_pipeline 0 "$S0NL" train_receivers dsdm "--init-mode expert"
    echo "GPU0: seed0 DSDM done $(date +%H:%M:%S)"

    run_pipeline 0 "$S0" train_receivers dsdm "--init-mode expert"
    echo "GPU0: seed0 DSDM+Logits done $(date +%H:%M:%S)"

    echo "====== GPU0: seed2 expert训练 ======"
    local S2="configs/main_cifar100_hetero6arch_ipc10_seed2.yaml"
    local S2NL="configs/main_cifar100_hetero6arch_ipc10_seed2_nologit.yaml"
    setup_seed_dir "seed2"
    setup_seed_dir "seed2_nologit"

    run_pipeline 0 "$S2" train_experts ""
    echo "GPU0: seed2 experts done $(date +%H:%M:%S)"

    link_experts_from "seed2" "seed2_nologit"
    run_pipeline 0 "$S2" attach_logits dsdm
    run_pipeline 0 "$S2" build_communication heuristic
    run_pipeline 0 "$S2" build_communication dsdm
    run_pipeline 0 "$S2NL" build_communication heuristic
    run_pipeline 0 "$S2NL" build_communication dsdm
    echo "GPU0: seed2 communication ready $(date +%H:%M:%S)"

    run_pipeline 0 "$S2NL" train_receivers heuristic "--init-mode expert"
    echo "GPU0: seed2 Heuristic done $(date +%H:%M:%S)"

    run_pipeline 0 "$S2NL" train_receivers dsdm "--init-mode expert"
    echo "GPU0: seed2 DSDM done $(date +%H:%M:%S)"

    run_pipeline 0 "$S2" train_receivers dsdm "--init-mode expert"
    echo "GPU0: seed2 DSDM+Logits done $(date +%H:%M:%S)"

    echo "====== GPU0 ALL DONE ======"
}

# ──────────────────────────────────────────────
# GPU1 流程：seed1 expert训练 → 实验
# ──────────────────────────────────────────────
gpu1_job() {
    echo "====== GPU1: seed1 expert训练 ======"
    local S1="configs/main_cifar100_hetero6arch_ipc10_seed1.yaml"
    local S1NL="configs/main_cifar100_hetero6arch_ipc10_seed1_nologit.yaml"
    setup_seed_dir "seed1"
    setup_seed_dir "seed1_nologit"

    run_pipeline 1 "$S1" train_experts ""
    echo "GPU1: seed1 experts done $(date +%H:%M:%S)"

    link_experts_from "seed1" "seed1_nologit"
    run_pipeline 1 "$S1" attach_logits dsdm
    run_pipeline 1 "$S1" build_communication heuristic
    run_pipeline 1 "$S1" build_communication dsdm
    run_pipeline 1 "$S1NL" build_communication heuristic
    run_pipeline 1 "$S1NL" build_communication dsdm
    echo "GPU1: seed1 communication ready $(date +%H:%M:%S)"

    run_pipeline 1 "$S1NL" train_receivers heuristic "--init-mode expert"
    echo "GPU1: seed1 Heuristic done $(date +%H:%M:%S)"

    run_pipeline 1 "$S1NL" train_receivers dsdm "--init-mode expert"
    echo "GPU1: seed1 DSDM done $(date +%H:%M:%S)"

    run_pipeline 1 "$S1" train_receivers dsdm "--init-mode expert"
    echo "GPU1: seed1 DSDM+Logits done $(date +%H:%M:%S)"

    echo "====== GPU1 ALL DONE ======"
}

# ──────────────────────────────────────────────
# 并行启动
# ──────────────────────────────────────────────
echo "===== 开始时间: $(date +%H:%M:%S) ====="
gpu0_job &
PID0=$!
gpu1_job &
PID1=$!

wait $PID0 && echo "GPU0 job finished OK" || echo "GPU0 job FAILED"
wait $PID1 && echo "GPU1 job finished OK" || echo "GPU1 job FAILED"

echo "===== 全部完成: $(date +%H:%M:%S) ====="
