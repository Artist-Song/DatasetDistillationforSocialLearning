# PROJECT_SPEC.md

> 本文件保留历史研究设计。当前唯一有效的实验口径以 `AGENTS.md` 为准；
> hetero6arch、VGG 和统一 ConvNet packet encoder 不再是当前主实验。

## 项目名称

```text
DatasetDistillationforSocialLearning
```

## 课题名称

中文：

```text
面向异构社会化学习的高知识密度蒸馏通信包构建方法
```

英文：

```text
Distilled Knowledge Packets for Communication-Efficient Heterogeneous Socialized Learning
```

---

## 研究背景

边缘智能设备（摄像头、传感器、手机、工业终端等）在实际部署中存在两个天然约束：

```text
1. 硬件异构：不同设备算力差异大，运行的模型架构必然不同（ConvNet / VGG / ResNet / AlexNet）
2. 通信受限：边缘设备通过无线网络通信，传输大量原始数据的代价过高
```

在此背景下，社会化学习关注多个 agent 之间通过互相传递知识来扩展各自的能力。每个 agent 初始只掌握本地场景对应的 expert classes，社会化学习后需要同时满足两个目标：

```text
目标 1（可塑性）：新类准确率不低 —— 充分吸收其他 agent 的外部知识
目标 2（稳定性）：旧类准确率不降 —— 保持自身 expert knowledge，避免遗忘
```

本项目关注的核心问题：

```text
在模型异构和通信受限同时存在的社会化学习场景中，agent 之间应该传递什么形式的知识？
```

传统参数、梯度和中间特征通信依赖具体网络结构，异构 agent 之间难以直接共享。相比之下，输入空间图像样本天然可以被不同结构模型读取和学习，更适合作为异构 agent 之间的共同通信载体。

直接传输真实样本在低通信预算下存在信息覆盖不足的问题。本项目将数据集蒸馏（DSDM）引入社会化学习，将每个 agent 的局部专家知识压缩为少量 synthetic images，形成 distilled knowledge packet，兼顾异构兼容性和高知识密度通信。

---

## 核心研究问题

```text
Q1：如何构建一种异构模型均可学习的通信对象？
Q2：如何在低通信预算下提高单位通信量中的知识密度？
Q3：如何让 receiver 在吸收外部新类知识的同时，不遗忘自身 expert 类？
```

对应思路：

```text
A1：使用输入空间 image packet，而不是参数、梯度或特征。
A2：使用共享轻量指导模型（ConvNet-3）统一生成 DSDM synthetic packet，
    解耦 sender 推理架构与蒸馏质量，确保任意异构 receiver 均可吸收。
A3：使用 expert initialization、FR loss（含 feature-level 约束）、expert 类加权 L_cls
    以及自身数据 replay（sched_self）共同缓解遗忘，同时用 KD loss 增强新类吸收。
```

---

## 方法概述

方法由四个阶段组成。

### 1. Local Expert Training

每个 agent 只使用自己的 expert classes 训练本地专家模型。所有模型输出维度固定为 100，不重映射标签。

### 2. Knowledge Packet Construction

支持四种 packet 类型：

```text
DSDM        ：数据集蒸馏生成 synthetic image packet（目标方法）
Heuristic   ：随机真实样本 subset packet（baseline）
Importance  ：低置信度真实样本 subset packet（baseline）
Full Real   ：sender 全部真实训练数据（oracle upper-bound，不是低通信方法）
```

### 3. Optional Sender Logits Attachment

在 image packet 基础上附加 sender expert-class logits。只保留 sender 自己 expert classes（25 个）上的输出，不保留 full 100-class logits。用于补充 soft decision information，增强 receiver 对外部类别的吸收能力。

### 4. Receiver Social Training

receiver 读取 self packet + external packets，联合训练：

```text
L = L_cls + lambda_fr * L_FR + lambda_kd * L_KD

L_cls ：hard-label classification loss（self + external packet）
L_FR  ：保持 receiver expert classes 上旧模型响应（expert init 时启用）
L_KD  ：学习 external sender expert-class logits（有 logits 时启用）
```

---

## 论文主实验设定（当前阶段）

### 数据集与划分

```text
Dataset  : CIFAR-100
Agents   : 4
Split    : class-disjoint，每 agent 25 个 expert classes，全局标签不重映射

agent_0: classes 0–24
agent_1: classes 25–49
agent_2: classes 50–74
agent_3: classes 75–99
```

### Agent 架构（hetero6arch 扩展异构模型池）

本实验采用扩展异构模型池，4 个 agent 从以下 6 种架构中随机选取（允许重复）：

| 架构 | width/variant | 参数量 | 架构族 |
|---|---|---|---|
| ConvNet-3 | w1.0（DSDM 默认） | ~400K | ConvNet |
| ConvNet-4 | w1.5 | ~2.4M | ConvNet |
| VGG11-CIFAR | — | ~9.8M | VGG |
| AlexNetCIFAR | — | ~7.0M | AlexNet |
| ResNet-10-CIFAR-w0.5 | depth=10 / base width 32 | 1.25M | ResNet |
| ResNet-18-CIFAR-w0.5 | depth=18 / base width 32 | 2.82M | ResNet |

该历史模型池横跨 ConvNet / VGG / AlexNet / ResNet 四个架构族。旧配置 ID
`resnet10` / `resnet18` 对应上述 compact 版本，不是 11M canonical ResNet。新增
`resnet_cifar_standard` 的 ResNet-18-CIFAR-w1.0 为 11,220,132 参数，只用于后续诊断。

#### 历史实验参考：Hetero4arch（ConvNet + VGG + AlexNet，保留参考）

早期采用固定 4 arch 设定：ConvNet-3 / ConvNet-4 / VGG11 / AlexNet。代码适配已完成，初步实验显示 AlexNet receiver 效果较好，VGG receiver 遗忘较严重。对应 config：

```text
configs/main_cifar100_hetero4arch_ipc10.yaml
configs/main_cifar100_hetero4arch_ipc50.yaml
```

### 通信预算

```text
主实验：IPC = 10
扩展实验：IPC = 50

IPC=10 时每个 sender raw packet = 25 × 10 = 250 张
IPC=50 时每个 sender raw packet = 25 × 50 = 1250 张
（DSDM factor=2，decode 后 effective images 翻4倍，但通信量统计以 raw 计）
```

### 对比方法

```text
1. Expert Only（Before Social）    ：社会化前基线
2. Full Real Social Transfer        ：通信充分上限（oracle）
3. Heuristic Packet, IPC=10        ：随机真实样本核心集（250张/sender）
4. DSDM Packet, IPC=10             ：共享指导模型生成合成包（250张/sender）
5. DSDM Packet + Logits, IPC=10    ：合成包 + sender expert-class soft labels
```

### 核心研究问题（实验层）

```text
1. 在低通信预算（IPC=10）下，DSDM 是否相比 Heuristic 展现更高知识密度（acc_new 更高）？
2. Sender logits 是否在 image packet 基础上进一步提升 new-class absorption？
3. 不同架构 receiver（ConvNet / VGG / AlexNet / ResNet）对 DSDM packet 的吸收能力是否不同？
4. 共享指导模型生成的 packet 是否对所有异构 receiver 均有效？
5. FR loss 改进（feature-level + expert 加权）能否有效缓解 forgetting，使 acc_expert 保持稳定？
```

---

## DSDM 配置约定

### Feature Index 规则

DSDM 蒸馏时使用 sender guide model 的最后一个非 logits 语义特征层（last_feature）。各架构的 `f_idx` 固定如下：

| 架构 | f_idx | 特征 shape |
|---|---|---|
| ConvNet-3 | 2 | block3 output |
| ConvNet-4 | 3 | block4 output |
| VGG11-CIFAR | 10 | [B, 512] |
| AlexNetCIFAR | 7 | [B, 512] |

`f_idx` 由 `agent_data._refresh_model_metadata` 按 net_type 自动设置，每次修改 per-agent 模型配置后必须通过 dry-run 打印验证，确保 `f_idx` 与 `idx_from/idx_to` 同步更新。

### DSDM 公共参数

```text
factor       : 2
decode_type  : single
init         : mix（factor>1 时自动启用）
match        : semantic
metric       : mse
aug_type     : color_crop_cutout
pretrained_epochs : 20（teacher bank 固定使用 e20 版本）
```

### 各架构 DSDM 最优 Recipe（IPC=10，formal sweep）

| 架构 | best_lr_img | best_iter | best_eval_acc | full_real_upper |
|---|---|---|---|---|
| ConvNet-3-w1.0 | 0.1 | 10000 | — | — |
| ConvNet-4-w1.5 | 0.1 | 10000 | — | — |
| VGG11-CIFAR | 0.02 | 10000 | 43.34 | 72.42 |
| AlexNetCIFAR | 0.005 | 5000 | 46.00 | 74.43 |

VGG/AlexNet 的 teacher bank 路径：

```text
AlexNet:
/root/autodl-tmp/outputs/dsdm_arch_recipe_cifar100_ipc10/pretrained_factor2_probe/e20/alexnet

VGG:
/root/autodl-tmp/outputs/dsdm_arch_recipe_cifar100_ipc10/pretrained_factor2_probe_parallel/e20/vgg
```

每个 teacher bank 必须有 10 个 `cifar100_model_*.pth`，必须是 factor=2 / e20 版本。

### 注意事项

- AlexNet/VGG 内部写死 BatchNorm，`norm_type` 字段仅用于记录，不动态切换。
- 不要使用 torchvision ImageNet 版本的 AlexNet/VGG，本项目使用 CIFAR-100 32×32 定制版。
- `lr_img=0.1` 对 VGG/AlexNet 均明显退化，禁止用于这两个架构。
- 不要用 DSDM teacher 权重初始化 agent，teacher bank 仅用于蒸馏阶段。

---

## 历史实验记录（已完成，仅供参考）

### 跨架构异构诊断（ConvNet / ResNet / ResNetAP，IPC=50）

旧设定使用 ConvNet-3 / ResNet-10 / ResNetAP-10 异构。主要发现：

- DSDM_LOGIT 相比 image-only 有提升；HEURISTIC_LOGIT 在 IPC=50 下竞争力强。
- agent 2 存在严重 expert forgetting，暴露 receiver loss 需要调整。
- ResNet/ResNetAP sender 中 DSDM 使用的 `idx_from=2` 对应 layer2 而非最后特征层，是异常主因之一。

该设定已不作为主实验，相关 config 保留在 `configs/main_cifar100_logit.yaml`。

### Centralized Full Data Upper Bound

| 架构 | recipe | accuracy |
|---|---|---|
| ConvNet-3-IN | strict DSDM recipe | 65.24 |
| ResNet-10-CIFAR-w0.5 | CIFAR recipe | 72.08 |
| ResNetAP-10-BN | CIFAR recipe | 73.47 |

结果文件：`outputs/cifar100_4agent_25cls_upper_bound/centralized_full/`

### ConvNet Family 异构实验（conv-3/conv-4 only，IPC=10/50）

过渡阶段实验，用于降低跨架构噪声并调优 receiver 超参数。已完成 DSDM 蒸馏与 receiver 调参，参考结果：

```text
IPC=10 best receiver: ep060 / lambda_fr=0.20 / lambda_kd=0.50 -> mean_global=33.905
IPC=50 best receiver: ep250 / lambda_fr=0.05 / lambda_kd=0.50 -> mean_global=46.1825
```

相关结果目录：
```text
outputs/cifar100_4agent_25cls_conv_family_ipc10/
outputs/cifar100_4agent_25cls_conv_family_ipc50/
```

---

## 代码修改原则

1. 尽量复用现有 pipeline，不大规模重构 DSDM 源码。
2. 新验证实验优先通过新增 packet method、config 和小模块完成。
3. 所有新增函数和类必须有简短中文注释。
4. 每次修改后至少运行 `py_compile` 和 `dry-run`。
5. 修改 DSDM feature-index 逻辑时，必须同时验证 `f_idx` 与 `idx_from/idx_to`，并用 dry-run 打印每个 agent 的最终解析值。
6. 不要覆盖已有可用结果；新实验必须使用新的 `run_name` 和输出目录。
7. 修改前告知用户修改逻辑与方案，确认后执行。
8. 不要强制 reset、rebase、clean，除非用户明确要求。

## 推荐 smoke test

```bash
python -m py_compile \
  run_social_pipeline.py \
  agent_data.py \
  agent_trainer.py \
  config_adapter.py \
  output_manager.py \
  packet_consumer.py \
  packet_logits.py \
  selection_methods.py \
  social_trainer.py \
  social_output_manager.py \
  validate_packets.py \
  run_centralized_full.py
```

---

## 当前状态与下一步

**历史状态（2026-07-07，非当前主设定）**

- ConvNet family IPC=10 主实验已完成，核心结论成立（DSDM 新类吸收近乎翻倍）
- Hetero4arch（ConvNet+VGG+AlexNet）代码适配与初步实验已完成
- All-ConvNet DSDM 蒸馏包已生成，可复用作为共享指导模型 packet
- ResNet-10 / ResNet-18 模型已有代码支持，需训练 expert 模型
- 共享指导模型方案（方案 B）已确定为主实验方向

**下一步（hetero6arch 主实验推进）**

```text
1. 训练 ResNet-10 / ResNet-18 的 4 个 class split expert 模型
2. 写新 config：hetero6arch_ipc10.yaml（6 架构模型池 + 随机分配 + 复用 conv_family packets）
3. Dry-run 验证新 config 流程正常
4. 跑 Heuristic baseline（3 个随机种子）
5. 跑 DSDM 主实验（3 个随机种子）
6. 跑 DSDM+Logits（3 个随机种子）
7. 实现并验证 loss 改进（提高 sched_self、expert 类加权 L_cls、feature-level FR）
8. 整理结果，准备 AAAI 2027 投稿材料
```

**Receiver Loss 改进待实现**

```text
改进1（最优先）：提高 sched_self 比例至 40–50%（改 config 即可）
改进2：Expert 类在 L_cls 中加权（权重 1.5–2.0）
改进3（可选）：Feature-level FR loss（需加 get_feature 接口）
```
