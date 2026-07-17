#!/bin/bash
# 今晚全实验主控脚本（00:04 → 10:00）
# GPU0: Full Real + 消融实验（FR_KL/feat/self_kd）+ 跨架构评估
# GPU1: ResNet-10蒸馏 → ResNet-18蒸馏 → 跨架构评估

set -e
cd /root/autodl-tmp/DatasetDistillationforSocialLearning
mkdir -p logs/overnight

PY="/root/miniconda3/envs/sp/bin/python"
CROSS_EVAL="scripts/run_cross_arch_packet_eval.py"
CROSS_OUT="outputs/experiment_registry/cross_arch_packet_eval.csv"
ARCHS="convnet3w1 convnet4w15 vgg alexnet resnet10 resnet18"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ─── GPU0 流程 ────────────────────────────────────────────────────────────────
gpu0_job() {
    log "GPU0: 启动 Full Real social transfer（3 seeds）"
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=0 $PY run_social_pipeline.py \
            --config "configs/main_cifar100_hetero6arch_ipc10_seed${seed}_fullreal.yaml" \
            --stage train_receivers --packet-method full_real --init-mode expert \
            >> logs/overnight/fullreal_seed${seed}.log 2>&1
        log "GPU0: Full Real seed${seed} done"
    done

    log "GPU0: 启动消融实验（4个loss变体）"
    for variant in fr_kl fr_kl_feat fr_kl_selfkd fr_kl_feat_selfkd; do
        # attach_logits（消融组用seed0的expert，logits需重新生成）
        CUDA_VISIBLE_DEVICES=0 $PY run_social_pipeline.py \
            --config "configs/main_cifar100_hetero6arch_ipc10_seed0_${variant}.yaml" \
            --stage attach_logits --packet-method dsdm \
            >> logs/overnight/ablation_${variant}.log 2>&1
        # train_receivers（DSDM+Logits）
        CUDA_VISIBLE_DEVICES=0 $PY run_social_pipeline.py \
            --config "configs/main_cifar100_hetero6arch_ipc10_seed0_${variant}.yaml" \
            --stage train_receivers --packet-method dsdm --init-mode expert \
            >> logs/overnight/ablation_${variant}.log 2>&1
        log "GPU0: 消融 ${variant} done"
    done

    log "GPU0: 跨架构评估（ConvNet-3 self + Conv3-shared, cls 25-49）"
    CUDA_VISIBLE_DEVICES=0 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_1/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_1/packets/heuristic_packet.pt" \
        --packet-labels "ConvNet3-self(25-49)" "ConvNet3-shared(25-49)" "Heuristic(25-49)" \
        --class-ids $(seq -s' ' 25 49) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 0 \
        >> logs/overnight/cross_arch_cls2549.log 2>&1
    log "GPU0: cross-arch cls25-49 done"

    log "GPU0: 跨架构评估（VGG self + Conv3-shared, cls 50-74）"
    CUDA_VISIBLE_DEVICES=0 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_2/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_2/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_2/packets/heuristic_packet.pt" \
        --packet-labels "VGG11-self(50-74)" "ConvNet3-shared(50-74)" "Heuristic(50-74)" \
        --class-ids $(seq -s' ' 50 74) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 0 \
        >> logs/overnight/cross_arch_cls5074.log 2>&1
    log "GPU0: cross-arch cls50-74 done"

    log "GPU0: 跨架构评估（AlexNet self + Conv3-shared, cls 0-24）"
    CUDA_VISIBLE_DEVICES=0 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_3/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_0/packets/heuristic_packet.pt" \
        --packet-labels "AlexNet-self(0-24)" "ConvNet3-shared(0-24)" "Heuristic(0-24)" \
        --class-ids $(seq -s' ' 0 24) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 0 \
        >> logs/overnight/cross_arch_cls0024.log 2>&1
    log "GPU0 ALL DONE"
}

# ─── GPU1 流程 ────────────────────────────────────────────────────────────────
gpu1_job() {
    log "GPU1: 启动 ResNet-10 自蒸馏（cls 50-74, IPC=10, f_idx=5）"
    CUDA_VISIBLE_DEVICES=1 $PY run_social_pipeline.py \
        --config "configs/cross_arch_distill_resnet10_cls5074.yaml" \
        --stage distill_packets \
        >> logs/overnight/resnet10_distill.log 2>&1
    log "GPU1: ResNet-10 蒸馏完成"

    log "GPU1: 启动 ResNet-18 自蒸馏（cls 25-49, IPC=10, f_idx=5）"
    CUDA_VISIBLE_DEVICES=1 $PY run_social_pipeline.py \
        --config "configs/cross_arch_distill_resnet18_cls2549.yaml" \
        --stage distill_packets \
        >> logs/overnight/resnet18_distill.log 2>&1
    log "GPU1: ResNet-18 蒸馏完成"

    # ResNet cross-arch eval（等蒸馏完后）
    log "GPU1: 跨架构评估（ResNet-10 self, cls 50-74）"
    CUDA_VISIBLE_DEVICES=1 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cross_arch_distill_resnet10_cls5074_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
        --packet-labels "ResNet10-self(50-74)" \
        --class-ids $(seq -s' ' 50 74) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 1 \
        >> logs/overnight/cross_arch_resnet10.log 2>&1
    log "GPU1: ResNet-10 cross-arch done"

    log "GPU1: 跨架构评估（ResNet-18 self, cls 25-49）"
    CUDA_VISIBLE_DEVICES=1 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cross_arch_distill_resnet18_cls2549_ipc10/agents/agent_0/packets/dsdm_packet.pt" \
        --packet-labels "ResNet18-self(25-49)" \
        --class-ids $(seq -s' ' 25 49) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 1 \
        >> logs/overnight/cross_arch_resnet18.log 2>&1

    log "GPU1: 跨架构评估（ConvNet4 self, cls 75-99）"
    CUDA_VISIBLE_DEVICES=1 $PY $CROSS_EVAL \
        --packet-paths \
            "outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents/agent_1/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_3/packets/dsdm_packet.pt" \
            "outputs/cifar100_4agent_25cls_conv_family_ipc10/agents/agent_3/packets/heuristic_packet.pt" \
        --packet-labels "ConvNet4-self(75-99)" "ConvNet3-shared(75-99)" "Heuristic(75-99)" \
        --class-ids $(seq -s' ' 75 99) \
        --archs $ARCHS --epochs 300 --out-csv $CROSS_OUT --gpu 1 \
        >> logs/overnight/cross_arch_cls7599.log 2>&1
    log "GPU1 ALL DONE"
}

# ─── 并行启动 ─────────────────────────────────────────────────────────────────
log "===== 夜间实验启动: $(date) ====="
gpu0_job &
PID0=$!
gpu1_job &
PID1=$!
wait $PID0 && log "GPU0 finished OK" || log "GPU0 FAILED"
wait $PID1 && log "GPU1 finished OK" || log "GPU1 FAILED"
log "===== 全部完成: $(date) ====="
