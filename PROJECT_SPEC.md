# PROJECT_SPEC.md

## 问题定义
本项目研究多 agent 社会化学习中的低成本知识交互问题。不同于直接交换模型参数，我们将跨 agent 交互对象定义为 social packet。每个 agent 先在本地 expert 数据上训练 anchor model，再通过 DSDM 将本地 expert 数据蒸馏为少量 distilled images，并结合 anchor model 生成 soft targets，构成 social packet。随后，各 agent 通过 social pool 交换 packet，并在接收端进行 packet-based social learning。

## 任务流程
Step 1. agent 学习自己的 expert 类
Step 2. agent 使用 DSDM 蒸馏 expert 类，并生成 soft targets
Step 3. social packet 传入整体通信池
Step 4. agent 从通信池抓取 packet 并学习
Step 5. 输出 expert/general/average/communication/heterogeneity 指标

## 数据划分
### direct
- CIFAR10-5-2
- CIFAR100-4-25

### social
每个 agent 具有：
- private expert classes
- bridge classes

## model pool
- conv
- resnet18
- vit_tiny

## social packet
SocialPacket(
    sender_id,
    class_ids,
    images,
    hard_labels,
    soft_targets,
    meta
)

## baseline 套件
- local_only
- raw_share
- masc_style_baseline
- packet_x_only
- packet_x_q

## 默认训练策略
- sender 先离线蒸馏 packet
- packet 缓存到磁盘
- receiver 训练时读取 packet
- social policy 使用 all-to-all
- receiver loss = local loss + packet loss + retain loss

## receiver loss
L = L_local + lambda_packet * L_packet + lambda_retain * L_retain

其中：
- L_local: 本地 expert 数据上的监督损失
- L_packet: packet 上的 CE + KD
- L_retain: 当前模型与本地 anchor 在 expert 数据上的保持约束