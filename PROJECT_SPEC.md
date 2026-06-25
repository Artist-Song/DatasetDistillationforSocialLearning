# PROJECT_SPEC.md

## 项目名称

```text
DatasetDistillationforSocialLearning
```

## 课题名称

中文暂定名：

```text
面向异构社会化学习的高知识密度蒸馏通信包构建方法
```

英文暂定名：

```text
Distilled Knowledge Packets for Communication-Efficient Heterogeneous Socialized Learning
```

## 研究背景

社会化学习关注多个智能体之间通过交互实现能力增长。每个 agent 初始只掌握一部分 expert classes，在社会化学习后需要获得其他 agent 的互补知识，同时尽可能保持自身原有 expert knowledge。

当前项目关注更具体的问题：

```text
在模型异构和通信受限同时存在的社会化学习场景中，agent 之间应该传递什么形式的知识？
```

传统参数、梯度和中间特征通信依赖具体网络结构。当 sender 和 receiver 采用不同模型结构时，模型空间知识难以直接共享。相比之下，输入空间图像样本天然可以被不同结构模型读取和学习，因此更适合作为异构 agent 之间的共同通信对象。

但是，直接传输真实样本在低通信预算下存在信息覆盖不足的问题。为此，本项目将数据集蒸馏引入社会化学习，将每个 agent 的局部专家知识压缩为少量 synthetic images，形成 distilled knowledge packet，从而兼顾异构兼容性和高知识密度通信。

## 核心研究问题

本项目围绕三个核心问题展开：

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

## 方法概述

当前方法由四部分组成。

### 1. Local Expert Training

每个 agent 只使用自己的 expert classes 训练本地专家模型。

CIFAR-100 主设定：

```text
agent 0: classes 0-24
agent 1: classes 25-49
agent 2: classes 50-74
agent 3: classes 75-99
```

每个模型输出维度仍然是 100。

### 2. Knowledge Packet Construction

当前支持四种 packet：

```text
DSDM：蒸馏 synthetic image packet
Heuristic：随机真实样本 subset packet
Importance：低置信度真实样本 subset packet
Full Real：全部真实样本 packet，用作 upper-bound baseline
```

DSDM 是目标方法。Full Real 不是低通信方法，只用于诊断上限。

### 3. Optional Sender Logits Attachment

在 image packet 基础上，可以附加 sender expert-class logits。

logits 只保留 sender 自己 expert classes 上的输出，不保留 full 100-class logits。

该模块用于补充 soft decision information，增强 receiver 对外部类别的吸收能力。

### 4. Receiver Social Training

receiver 读取所有 packet，包括 self packet 和 external packets。

训练目标：

```text
L = L_cls + lambda_fr * L_FR + lambda_kd * L_KD
```

其中：

```text
L_cls：hard-label classification loss
L_FR：保持 receiver expert classes 上旧模型响应
L_KD：学习 external sender expert-class logits
```

expert initialization 下启用 FR；scratch initialization 下不启用 FR。

## 当前主要实验设定

### 数据集

```text
CIFAR-100
```

### 划分方式

```text
4 agents，每个 agent 25 classes，class-disjoint split
```

### 模型设置

已有跨架构异构诊断设置：

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet_ap",
}
```

该设置已经用于 CIFAR-100 IPC=50 初步诊断。当前发现跨架构异构会引入较强架构迁移噪声，因此下一阶段主实验优先切换为 ConvNet family 内部异构。

下一阶段 ConvNet family 异构设置：

```text
agent_0: ConvNet-3-w0.5
agent_1: ConvNet-3-w1.0
agent_2: ConvNet-4-w1.0
agent_3: ConvNet-4-w1.5
```

该设置保留模型容量、参数形状和中间特征维度差异，因此仍然不能直接进行参数平均、梯度共享或特征对齐；但所有 agent 处于同一 ConvNet family，有助于降低跨架构 DSDM transfer 的额外噪声。

### 通信预算

当前已有主要实验：

```text
IPC = 50
```

CIFAR-100 IPC=50 时：

```text
每个 sender raw packet images = 25 × 50 = 1250
每个 receiver external raw images = 3 × 1250 = 3750
```

## 当前已有结果总结

当前 CIFAR-100 IPC=50 实验显示：

```text
1. DSDM_LOGIT 相比 DSDM image-only 在平均 global accuracy 和 new accuracy 上有提升。
2. HEURISTIC_LOGIT 在 IPC=50 下平均性能高于 DSDM_LOGIT。
3. IMPORTANCE_LOGIT 表现明显较差。
4. DSDM_LOGIT 在 ResNet / ResNet_AP receiver 上的 new-class absorption 有一定优势。
5. agent 2 在 DSDM_LOGIT 下出现严重 expert forgetting。
6. 用户已经完成 all-ConvNet 同构实验。
```

### Upper-bound 诊断结果

截至 2026-06-25，已完成 Full Real Social Transfer 和 Centralized Full Data Upper Bound。

旧 centralized recipe 结果：

```text
ConvNet-3-IN:     50.89
ResNet-10:        28.69
ResNetAP-10:      33.28
```

该结果用于诊断，不能作为最终模型能力上限。ResNet / ResNetAP 过低主要由训练 recipe 不适配导致。

修正 recipe 后的 centralized upper-bound：

```text
ConvNet-3-IN, strict DSDM recipe: 65.24
ResNet-10-BN, CIFAR recipe:       72.08
ResNetAP-10-BN, CIFAR recipe:     73.47
```

对应结果文件：

```text
outputs/cifar100_4agent_25cls_upper_bound/centralized_full/centralized_results_conv3in_dsdm_strict.csv
outputs/cifar100_4agent_25cls_upper_bound/centralized_full/centralized_results_resnet10_bn_aug_ms.csv
```

Full Real Social Transfer 当前结果：

```text
receiver 0 ConvNet:   global after 49.94, new after 49.31
receiver 1 ConvNet:   global after 49.85, new after 50.39
receiver 2 ResNet:    global after 29.18, new after 26.68
receiver 3 ResNetAP:  global after 37.98, new after 35.97
```

注意：Full Real Social Transfer 仍使用旧 receiver training recipe，因此 ResNet receiver 结果偏低不能解释为模型能力上限。下一阶段 social receiver 训练需要同步 recipe。

## 下一阶段主实验：ConvNet Family 异构 IPC=10

日期：2026-06-25

### 研究动机

跨架构异构设置 ConvNet / ResNet / ResNetAP 虽然符合异构学习设定，但会额外引入架构迁移、训练 recipe 和 receiver 优化差异。为了更清晰地验证 DSDM packet 的单位通信知识密度，下一阶段先采用 ConvNet family 内部异构。

该设置有三个优点：

```text
1. 保留异构性：不同 depth / width 导致参数形状、特征维度和模型容量不同。
2. 降低噪声：所有 agent 使用同一 ConvNet family，减少跨架构迁移带来的不可控因素。
3. 对齐 DSDM：ConvNet-3/4 + InstanceNorm 更接近 DSDM 常用 evaluator / backbone 口径。
```

### 实验设定

```text
Dataset: CIFAR-100
Agents: 4
Class split: 每个 agent 25 个类别，class-disjoint
Labels: 保留 CIFAR-100 全局标签，不重映射
Communication budget: IPC=10
```

agent 结构：

```text
agent_0: ConvNet-3-w0.5   弱模型
agent_1: ConvNet-3-w1.0   标准模型
agent_2: ConvNet-4-w1.0   更深模型
agent_3: ConvNet-4-w1.5   强模型
```

类别划分：

```text
agent_0: classes 0-24
agent_1: classes 25-49
agent_2: classes 50-74
agent_3: classes 75-99
```

### 对比方法

```text
Expert Only / Before Social
Full Real Social Transfer
Heuristic Real Packet, IPC=10
DSDM Packet, IPC=10
DSDM Packet + Logits, IPC=10
```

实验只改变通信包方法，评价指标和 receiver training 口径保持一致。

### 关键问题

```text
1. 在 IPC=10 下，DSDM 是否相比随机真实样本 Heuristic 展现更高知识密度？
2. DSDM_LOGIT 是否在 image packet 基础上进一步提升 new-class absorption？
3. 不同容量 ConvNet receiver 对 DSDM packet 的吸收能力是否不同？
4. Full Real Social Transfer 与 centralized upper-bound 之间的差距是否主要来自 receiver training recipe？
5. 是否仍存在明显 expert forgetting，需要调整 FR loss 或 receiver 训练策略？
```

### 执行前要求

```text
1. 先新增 ConvNet family 异构 config，不覆盖现有跨架构诊断 config。
2. per-agent 显式配置 depth / width / norm_type。
3. expert / receiver 训练尽量采用与 DSDM 兼容的 ConvNet recipe。
4. 每次修改后运行 py_compile 和 dry-run。
5. 正式运行前先确认实验计划和输出目录。
```
