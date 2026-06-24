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

主实验为异构设置：

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet_ap",
}
```

用户已经完成同构实验，用于验证异构是否影响 DSDM packet transferability。

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

当前较稳妥的阶段性结论是：

```text
输入空间 packet 可以跨异构模型使用，sender logits 能在平均意义上增强外部知识吸收；
但 IPC=50 下真实样本随机子集已经很强，DSDM 的高知识密度优势需要在更低 IPC 下继续验证；
同时，agent 2 的 forgetting 表明 receiver loss 和训练策略仍需调整。
```

当前不能过度声称：

```text
DSDM 在 IPC=50 下全面优于核心集方法。
```

## 当前最优先验证实验

用户当前只需要继续完成两个实验：

```text
1. Full Real Social Transfer
2. Centralized Full Data Upper Bound
```

其他实验，包括 factor ablation、agent 2 loss sweep、低 IPC 曲线，暂时放后。

---

## Experiment 1：Full Real Social Transfer

目的：

```text
验证社会化训练流程在通信充分条件下的上限。
```

设计：

```text
每个 sender 传输自己全部真实训练数据。
每个 sender：25 classes × 500 images = 12500 images。
每个 receiver external data：3 × 12500 = 37500 images。
receiver 从自身 expert model 初始化。
训练流程保持与 social training 一致。
```

需要输出：

```text
global accuracy
expert accuracy
new accuracy
forgetting
external_comm_images
```

预期解释：

```text
如果 FULL_REAL 明显高于 DSDM / Heuristic：
    当前瓶颈主要是 packet 信息不足。

如果 FULL_REAL 也不高：
    当前任务设定、模型容量或 receiver training 本身上限较低。

如果 FULL_REAL 下 agent 2 不遗忘：
    agent 2 遗忘主要与 DSDM packet 或训练分布有关。

如果 FULL_REAL 下 agent 2 仍遗忘：
    agent 2 问题主要来自 loss / lr / receiver 训练策略。
```

---

## Experiment 2：Centralized Full Data Upper Bound

目的：

```text
确认不同模型在 CIFAR-100 全量数据上的基础上限。
```

设计：

```text
ConvNet on full CIFAR-100
ResNet on full CIFAR-100
ResNet_AP on full CIFAR-100
```

该实验不属于 social learning，只作为 upper bound。

需要输出：

```text
full test accuracy
training epochs
model type
```

解释：

```text
如果 centralized full data 上限本身不高，则 social result 不应期待过高。
如果 centralized full data 很高，则当前 packet / social training 仍有较大改进空间。
```

## 需要新增或确认的工程功能

### 1. full_real packet method

新增或确认正式 packet method：

```text
full_real
```

功能：

```text
每个 agent 将自身全部 expert-class 真实训练数据保存为 packet。
```

涉及文件：

```text
run_social_pipeline.py
output_manager.py
packet_consumer.py
selection_methods.py
validate_packets.py
```

### 2. centralized full data training script

新增或确认脚本：

```text
run_centralized_full.py
```

功能：

```text
在完整 CIFAR-100 训练集上训练指定模型。
```

支持参数：

```text
--config
--model convnet/resnet/resnet_ap
--epochs
--lr
--batch-size
```

输出：

```text
outputs/{run_name}/centralized_full/{model}/metrics.json
outputs/{run_name}/centralized_full/{model}/checkpoint.pt
outputs/{run_name}/centralized_full/centralized_results.csv
```

## 实验结果汇总格式

Full Real Social Transfer 保留逐 agent 结果，并额外计算 4-agent 平均值。

### 表 1：逐 agent 结果

| Method | Receiver | Model | Global | Expert | New | Forgetting | External Images |
|---|---:|---|---:|---:|---:|---:|---:|

### 表 2：平均结果

| Method | Init | Logits | Global Avg | Expert Avg | New Avg | Forgetting Avg | External Images |
|---|---|---|---:|---:|---:|---:|---:|

Centralized Full Data 只需要 global accuracy：

| Model | Data | Epochs | LR | Batch Size | Global Acc |
|---|---|---:|---:|---:|---:|

## 阶段性汇报口径

当前阶段可以这样向老师汇报：

```text
当前 CIFAR-100 IPC=50 实验已经证明 input-space packet 可以跨 ConvNet、ResNet 和 ResNet_AP receiver 使用，sender logits 也能在平均意义上增强 DSDM packet 的外部知识吸收能力。但 IPC=50 下随机真实样本核心集已经具有较强竞争力，说明 DSDM 的高知识密度优势需要在更低通信预算下进一步验证。

同时，agent 2 出现严重 expert forgetting，说明当前 receiver loss 和训练策略还不够稳定。因此，当前优先补充 Full Real Social Transfer 和 Centralized Full Data Upper Bound 两个验证实验，以明确当前性能瓶颈到底来自任务上限、模型上限、packet 信息不足，还是 social training 流程本身。
```

## 当前最小任务清单

优先级从高到低：

```text
1. 实现或确认 FULL_REAL social transfer。
2. 实现或确认 centralized full CIFAR-100 upper bound。
3. 更新并上传 AGENTS.md。
4. 更新并上传 PROJECT_SPEC.md。
5. 运行 py_compile 和 dry-run。
6. 提交并 push 到 GitHub。
```

## 当前不建议立即做的事情

```text
1. 不建议立刻设计复杂 attention fusion。
2. 不建议立刻引入多轮通信。
3. 不建议立刻改 DSDM 主算法。
4. 不建议只根据 IPC=50 结果判断方法失败。
5. 不建议在 Full Real 和 Centralized 结果出来前大规模调 agent 2 loss。
```
