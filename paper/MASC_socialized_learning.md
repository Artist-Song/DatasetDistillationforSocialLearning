# MASC 精读笔记

文件：

```text
MASC - Socialized Learning Through Multi-Agent Collaboration.pdf
```

题目：

```text
Socialized Learning: Making Each Other Better Through Multi-Agent Collaboration
```

作者：

```text
Xinjie Yao, Yu Wang, Pengfei Zhu, Wanyu Lin, Jialu Li, Weihao Li, Qinghua Hu
```

## 一句话贡献

这篇文章提出 Socialized Learning 范式，并用 MASC 框架让多个 agent 在协作中学习彼此的新类别，同时保持自身 expert classes 的优势。

## 解决的问题

MASC 关注的问题不是普通 continual learning，也不是标准 federated learning，而是：

```text
多个 agent 初始各自掌握不同 expert classes；
经过社会化交互后，每个 agent 都应学会更多 general classes；
同时每个 agent 在自身 expert classes 上仍应保持方向性优势。
```

它强调两个核心性质：

1. Versatility：agent 能学习更多 general classes。
2. Directionality：agent 在自身 expert classes 上仍保持优势，而不是变成无偏的全局平均模型。

## 方法核心

MASC 由两个主要模块组成：

| 模块 | 作用 | 直观理解 |
|---|---|---|
| Collective Collaboration | 多个 teacher 共同指导一个 student/generalist | 让 student 吸收所有 agent 的类别知识 |
| Reciprocal Altruism | generalist 反过来帮助每个 teacher 成长 | 让每个 teacher 学新类，同时保留 expert 优势 |

训练逻辑可以理解为三步：

1. 每个 teacher agent 先掌握自己的 expert classes。
2. 多个 teacher 通过 CE、KD、energy alignment 指导 student，student 成为 generalist。
3. 每个 teacher 借助 generalist 的 backbone 和自身 expert classifier 形成 grown teacher。

推理阶段，MASC 使用 HFE classifier 在 general classifier 和 expert classifier 之间选择，从而实现 divide-and-conquer。

## 损失设计

MASC 的 student loss 由三部分组成：

| 损失 | 作用 |
|---|---|
| CE loss | 使用 ground-truth 学习类别监督 |
| KD loss | 从多个 teacher 学 soft knowledge |
| Alignment loss | 用 energy anchor 对齐 teacher/student 能量空间 |

grown teacher loss 也由 CE、KD、alignment 组成。区别是 grown teacher 阶段固定 student backbone，只训练对应 teacher classifier。

## 关键实验设置

MASC 的主实验包括：

| 数据集 | agent 数量 | 每个 agent expert classes |
|---|---:|---:|
| CIFAR-10 | 5 | 2 |
| CIFAR-100 | 4 | 25 |

这和我们当前 CIFAR-100 主设定高度相关：

```text
CIFAR-100
4 agents
每个 agent 25 classes
class-disjoint split
```

## 关键实验结果

CIFAR-100 主表中，MASC 的平均结果为：

| 阶段 | Expert | General | Average |
|---|---:|---:|---:|
| before growth | 66.28 | 0.00 | 33.14 |
| after growth | 65.40 | 58.64 | 62.02 |

逐 agent CIFAR-100 after growth：

| Agent | Expert | General | Average |
|---|---:|---:|---:|
| Agent-1 | 65.40 | 58.64 | 62.02 |
| Agent-2 | 65.36 | 55.68 | 60.52 |
| Agent-3 | 61.64 | 58.59 | 60.11 |
| Agent-4 | 65.92 | 59.57 | 62.75 |

消融结果：

| Setting | CIFAR-100 Expert | CIFAR-100 General | CIFAR-100 Average |
|---|---:|---:|---:|
| CC only | 62.16 | 61.55 | 61.85 |
| RA only | 67.92 | 54.36 | 61.14 |
| CC + RA | 65.40 | 58.64 | 62.02 |

解释：

1. CC 更偏向 non-directional general learning，能学新类，但不保证 expert 优势。
2. RA 更偏向保持 expert 优势，但 general absorption 较弱。
3. CC + RA 试图在 general learning 和 expert retention 之间取得平衡。

## 和我们课题的关系

我们可以沿用 MASC 的问题目标：

```text
每个 agent 初始掌握 expert classes；
social training 后应吸收其他 agent 的 new/general classes；
同时保持自身 expert knowledge。
```

我们当前的指标也应与 MASC 口径对齐：

| MASC 口径 | 我们当前口径 |
|---|---|
| Expert | Expert |
| General | New |
| Average | Balanced Avg = (Expert + New) / 2 |
| Directionality | Expert - New |

这解释了为什么我们不能只报 Global accuracy。CIFAR-100 中每个 receiver 的 new classes 有 75 类、expert classes 有 25 类，Global 会更偏向 new classes，而 MASC 的 Average 更像 expert/new 平衡平均。

## 我们和 MASC 的关键区别

这点写论文时必须讲清楚：

| MASC | 我们的方法 |
|---|---|
| 重点是完整 socialized collaboration 框架 | 重点是异构 agent 间的通信载体 |
| 使用 collective collaboration + reciprocal altruism | 使用 input-space knowledge packet |
| 依赖 student/generalist 与 HFE classifier | 当前不引入 HFE classifier |
| 主要在模型/分类器层面组织交互 | 通过图像 packet 传递 sender knowledge |
| 不强调 dataset distillation packet 的通信密度 | 用 DSDM 提高单位通信量知识密度 |

因此当前论文不能写成：

```text
我们复现或改进了完整 MASC。
```

更准确的写法是：

```text
我们沿用 socialized learning 的目标和评价思想，但把研究焦点放在异构模型可共享的 input-space communication carrier 上。
```

## 可直接服务论文的位置

### Introduction

可以用 MASC 引出：

1. agent 不只是要学新类，还要保持自身 expert knowledge。
2. socialized learning 比单 agent continual learning 更强调多 agent 交互。
3. 现有 socialized learning 仍需要进一步考虑异构模型之间如何通信。

### Problem Formulation

可以对齐：

1. 4-agent CIFAR-100 25-class split。
2. Expert / New / Balanced Avg 指标。
3. Directionality = Expert - New 的解释。

### Related Work

放在：

```text
Socialized and Collaborative Learning
```

写作重点：

```text
MASC establishes the socialized-learning objective, while our work studies what should be communicated among heterogeneous agents.
```

## 对当前实验的启发

1. 我们当前 Full Real Social Transfer 很重要，因为它能检查当前 receiver social training 是否接近 MASC 式完整知识吸收的上限。
2. agent 2 forgetting 不能只看 Global，要看 Expert、New、Balanced Avg 和 Directionality。
3. 如果 Full Real 后 Expert 仍低，说明问题更像 receiver training/loss，而不是 DSDM packet 本身。
4. 当前不应马上引入 HFE classifier，因为我们的第一层贡献不是复现 MASC，而是验证 input-space packet 的异构兼容性和知识密度。

## 论文中的谨慎表述

建议写：

```text
We adopt the socialized-learning objective of learning complementary classes from other agents while retaining each agent's original expertise.
```

建议写：

```text
Unlike MASC, which designs a full collaboration-and-inference framework, our work focuses on the communication carrier under model heterogeneity.
```

不要写：

```text
Our method is a complete MASC implementation.
```

不要写：

```text
Our current receiver training is directly comparable to MASC's HFE-based inference.
```

## 待复查

1. MASC supplementary 中是否有更详细的 CIFAR100-4-25 split 和模型结构。
2. MASC 是否所有 agent 使用同构模型；如果是，则可强调我们额外考虑 model heterogeneity。
3. MASC 的 Average 是否完全等同于我们的 Balanced Avg；当前从表格看是 Expert/General 的算术平均。
