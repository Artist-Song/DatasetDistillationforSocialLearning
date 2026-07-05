# PROJECT_SPEC.md

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

社会化学习关注多个智能体之间通过交互实现能力增长。每个 agent 初始只掌握一部分 expert classes，社会化学习后需要获得其他 agent 的互补知识，同时尽量保持自身原有 expert knowledge。

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
Q3：如何让 receiver 在吸收外部知识的同时保持自身 expert knowledge？
```

对应思路：

```text
A1：使用输入空间 image packet，而不是参数、梯度或特征。
A2：使用数据集蒸馏生成 compact synthetic packet。
A3：使用 expert initialization、FR loss 和可选 sender logits 缓解遗忘并增强知识吸收。
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

### Agent 架构（4arch 异构）

本实验采用跨 family 异构设定，agent pool 包含 ConvNet、VGG 和 AlexNet 三种架构系列：

| agent | 架构 | width/variant | 参数量 |
|---|---|---|---|
| 0 | ConvNet-3 | w1.0（DSDM 默认） | ~400K |
| 1 | ConvNet-4 | w1.5 | ~2.4M |
| 2 | VGG11-CIFAR | — | ~9.8M |
| 3 | AlexNetCIFAR | — | ~7.0M |

四个 agent 模型结构、参数形状、特征维度均不同，不能直接进行参数平均、梯度共享或特征对齐，符合异构社会化学习的实验前提。

Config 文件：

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
3. Heuristic Packet, IPC=10/50     ：随机真实样本核心集
4. DSDM Packet, IPC=10/50          ：蒸馏 synthetic packet
5. DSDM Packet + Logits, IPC=10/50 ：蒸馏 packet + sender expert logits
```

### 核心研究问题（实验层）

```text
1. 在低通信预算（IPC=10）下，DSDM 是否相比 Heuristic 展现更高知识密度？
2. Sender logits 是否在 image packet 基础上进一步提升 new-class absorption？
3. 不同架构 receiver（ConvNet / VGG / AlexNet）对 DSDM packet 的吸收能力是否不同？
4. 是否存在明显 expert forgetting，需要调整 FR loss 或 receiver 训练策略？
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
| ResNet-10-BN | CIFAR recipe | 72.08 |
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

**当前状态（2026-07-05）**

- AlexNet/VGG 已完成代码适配（`DSDM/train.py`、`config_adapter.py`、`agent_data.py`）
- 论文主实验 config 已建立（`hetero4arch_ipc10.yaml`、`hetero4arch_ipc50.yaml`）
- per-agent f_idx、modeltag、lr_img dry-run 验证通过

**下一步**

```text
1. 确认 VGG/AlexNet teacher bank 路径可访问（各 10 个 pth 文件）
2. 运行 train_experts 阶段，验证 AlexNet/VGG expert 训练正常收敛
3. 运行 distill_packets 阶段（IPC=10），验证 DSDM f_idx 特征匹配正确
4. 完成 IPC=10 主实验全组对比（Heuristic / DSDM / DSDM+Logits）
5. 复用经验启动 IPC=50 主实验
6. 根据实验结果调整 VGG/AlexNet IPC=50 的 lr_img（当前沿用 IPC=10 最优值）
```
