# EXPERIMENT_LOG.md

本文件只记录关键实验、协议变化和结论。原始 stdout/stderr 保存在 `logs/` 或
`external_baselines/logs/`，完整逐 run 指标由 `experiments/registry.yaml` 索引。

## 日志格式

```text
日期 / experiment_id / role / 目的
配置或相对基准修改
状态与核心结果
结论、caveat、原始产物路径
```

## 2026-06-25 | centralized-upper-bounds | diagnostic

- 修正旧训练 recipe 后，ConvNet-3-IN 达到 65.24，ResNet-10 达到 72.08，
  ResNetAP-10 达到 73.47。
- 结论：旧 ResNet 低上界主要来自训练 recipe，不是残差架构本身无能力。
- 产物：`outputs/cifar100_4agent_25cls_upper_bound/centralized_full/`。

## 2026-07-01 | conv-family-and-allconv | historical_evidence

- Conv-family IPC=10/50 和 all-ConvNet IPC=50 用于建立 DSDM 可行性、通信预算趋势和
  receiver 超参基础。
- 这些是重要中间证据，不能标记 invalid；但不再作为当前 one-ResNet 主表。
- 相关汇总保存在旧 experiment registry 和对应 run metrics。

## 2026-07-05 | hetero4arch | diagnostic

- 完成 ConvNet/VGG/AlexNet 跨架构适配和 feature-index 核对。
- 观察到 VGG/AlexNet stability 对 packet 来源和 receiver loss 高度敏感。
- 该阶段推动 no-VGG one-ResNet 主设定形成。

## 2026-07-07 | hetero6arch-convnet-packet | historical_evidence

- 将 ConvNet-family DSDM packet 复用到异构 receiver，验证 input-space packet 可跨架构消费。
- ResNet receiver 的 self-guided/ConvNet-guided 差异随 IPC 改变，不能声称统一 guide 恒优。
- 历史 ConvNet-4 packet 存在旧 feature-index 版本，正式消融需带 provenance。

## 2026-07-10--2026-07-17 | one-resnet-main | main

- 主设定固定为 ConvNet-3、ConvNet-4、AlexNet 和恰好一个 compact ResNet。
- 主方法固定为 sender task-backbone-guided DSDM + expert-class logits。
- IPC=10 三 seed 完成：Ours global 32.78、new 29.11、expert 43.81、forgetting 23.24。
- IPC=50 最终采用 seed0/2 两种子 paired 结果；seed1 后续经用户决定停止并排除。

## 2026-07-16 | fast-hard-label | baseline

- FAST 使用官方 pixels/per-class/minmax，每类严格 IPC 张真实图，无 logits。
- 三 seed 下 FAST 在 global/new 上低于随机 Heuristic；该现象保留，不按预期篡改。
- Receiver RNG 尚未显式按 config seed 初始化，后续严格 paired comparison 需要统一重跑。

## 2026-07-17 | masc-desa-complete | baseline

- 最新 MASC 使用 complete CC + RA + HFE reproduction；旧 MASC official-flow 数值被 supersede。
- DeSA 使用 owner-aware class-incremental adaptation；旧 comparable_fixed 数值保留历史但不进主表。
- MASC Full validation 已通过，说明低 IPC 结果不是缺少完整流程导致。

## 2026-07-17 | documentation-consolidation | operations

- `AGENTS.md` 重构为指示矩阵；`PROJECT_SPEC.md` 成为唯一研究事实来源。
- 建立人工 registry，将主结果、baseline、ablation、diagnostic、historical evidence 分离。
- 旧指示和历史 spec 已先做 SHA-256 一致的归档副本。
- 删除 checkpoint/packet 前仅生成候选清单，等待用户确认。

## 2026-07-17 | one-resnet-seed1-stop | decision

- 用户决定停止 seed1 IPC=50 队列，并以 seed0/2 作为 IPC=50 最终两种子结果。
- 终止整个进程组后 GPU 降为 0% 利用率；后续 receiver、MASC 和 DeSA 未启动。
- seed1 agent0 完成、agent1 约 17% 时停止；其 partial artifact 保留但不进入结果聚合。
- 为保持 paired comparison，IPC=50 的 Ours、Heuristic 和 FAST 均只聚合 seed0/2。
