#!/bin/bash
# 跨架构 packet 评估全流程启动脚本
# GPU 0: class splits 0-24 和 50-74
# GPU 1: class splits 25-49 和 75-99

set -e
BASE="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$BASE"
mkdir -p logs/cross_arch_eval

PY="/root/miniconda3/envs/sp/bin/python"
SCRIPT="scripts/run_cross_arch_packet_eval.py"
OUTCSV="outputs/experiment_registry/cross_arch_packet_eval.csv"

# 6 种 receiver 架构
ARCHS="convnet3w1 convnet4w15 vgg alexnet resnet10 resnet18"

# ───────────────────────────────────────────────
# 辅助函数：运行一个 class split 的评估
# ───────────────────────────────────────────────
run_split() {
    local GPU="$1"
    local SPLIT_NAME="$2"     # e.g. "cls0024"
    local CLASS_IDS="$3"      # e.g. "0 1 2 ... 24"
    local SELF_ARCH="$4"      # e.g. "AlexNet-self"
    local SELF_PACKET="$5"    # hetero4arch agent packet path
    local SHARED_PACKET="$6"  # conv_family agent packet path
    local HEURISTIC_PACKET="$7"

    echo "[GPU$GPU] 开始 $SPLIT_NAME ..."
    CUDA_VISIBLE_DEVICES=$GPU $PY $SCRIPT \
      --packet-paths \
        "$SELF_PACKET" \
        "$SHARED_PACKET" \
        "$HEURISTIC_PACKET" \
      --packet-labels \
        "${SELF_ARCH}-self(${SPLIT_NAME})" \
        "ConvNet3-shared(${SPLIT_NAME})" \
        "Heuristic(${SPLIT_NAME})" \
      --class-ids $CLASS_IDS \
      --archs $ARCHS \
      --epochs 300 \
      --out-csv "$OUTCSV" \
      >> "logs/cross_arch_eval/${SPLIT_NAME}.log" 2>&1
    echo "[GPU$GPU] 完成 $SPLIT_NAME  $(date +%H:%M:%S)"
}

# ───────────────────────────────────────────────
# GPU 0: classes 0-24 (AlexNet self) + 50-74 (VGG self)
# ───────────────────────────────────────────────
gpu0_job() {
    run_split 0 "cls0024" \
      "$(seq -s' ' 0 24)" \
      "AlexNet" \
      "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_3/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_0/packets/heuristic_packet.pt"

    run_split 0 "cls5074" \
      "$(seq -s' ' 50 74)" \
      "VGG11" \
      "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_2/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_2/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_2/packets/heuristic_packet.pt"
}

# ───────────────────────────────────────────────
# GPU 1: classes 25-49 (ConvNet-3 self) + 75-99 (ConvNet-4 self)
# ───────────────────────────────────────────────
gpu1_job() {
    run_split 1 "cls2549" \
      "$(seq -s' ' 25 49)" \
      "ConvNet3" \
      "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_1/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_1/packets/heuristic_packet.pt"

    run_split 1 "cls7599" \
      "$(seq -s' ' 75 99)" \
      "ConvNet4" \
      "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_1/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_3/packets/dsdm_packet.pt" \
      "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_3/packets/heuristic_packet.pt"
}

# ───────────────────────────────────────────────
# 并行启动
# ───────────────────────────────────────────────
echo "===== 跨架构评估开始: $(date +%H:%M:%S) ====="
gpu0_job &
PID0=$!
gpu1_job &
PID1=$!
wait $PID0 && echo "GPU0 完成" || echo "GPU0 失败"
wait $PID1 && echo "GPU1 完成" || echo "GPU1 失败"
echo "===== 全部完成: $(date +%H:%M:%S) ====="
