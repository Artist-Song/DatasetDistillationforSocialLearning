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

### 新要求：
日期：20260627
在ConvNet Family 异构 IPC=10的基础上，调整训练模式
## 任务定位

新增一个通信阶段变体，暂命名为：

Conv-Family Packet Generalist Communication

该任务只调整 communication / receiver training 阶段，不修改前期 expert 训练和 DSDM packet 蒸馏逻辑。

核心目标是把 socialized learning 中的 generalist / feedback 思想迁移到 input-space packet 框架中：

expert agents
-> distilled packets + sender expert-class logits
-> packet generalist
-> generalist logits 反哺 heterogeneous receivers

这里的 generalist 不是服务器，不聚合模型参数，不向 receiver 传 backbone，不使用 full real data。它只是一个由所有 communicated packets 训练出来的 packet-mediated social generalist，用来给 packets 生成额外 logits annotation。

## 总体流程

新增流程如下：

1. 沿用现有 conv-family expert checkpoints、DSDM packets、sender expert-class logits。
2. 新增阶段 train_packet_generalist：
   用所有 sender packets 训练一个 conv-family generalist。
3. 新增阶段 attach_generalist_logits：
   用 packet generalist 给所有 packet images 生成 generalist logits。
4. receiver 训练阶段新增 packet_generalist communication mode：
   第一版沿用现有 receiver 训练口径，即使用 manifest 中的 self packet + external packets 做 packet CE，
   不额外引入 self real CE，以保证与已完成的 DSDM_LOGIT baseline 可比。
   在此基础上增加 external sender KD + external generalist KD + FR。

不要破坏现有 direct packet / logit pipeline。packet generalist 应作为可选增强模式存在。

## Generalist 架构要求

1. generalist 使用 conv-family 中当前性能最强的模型架构。
2. 不要 hard-code 架构名，应通过 config 字段指定，例如：

generalist:
  enabled: true
  model: <best_conv_family_model>
  output_dim: 100

3. 如果当前代码无法自动判断 best conv-family model，则先使用 config 明确指定默认值。
4. generalist 输出维度固定为 100。
5. generalist 训练数据只能来自已有 packet images、packet labels、sender logits。
6. generalist 不允许读取 full CIFAR-100 real training set。
7. 第一版新建独立 run 目录，但通过 config 中的 reuse.source_run_name 复用已完成 conv-family IPC=10 run 的 DSDM packets，
   避免重复 expert training 和 DSDM distillation。

## Generalist 训练损失

packet generalist 的训练目标：

L_G = CE(G(x_packet), y_packet)
    + lambda_skd * KD(G(x_packet)[S_sender], z_sender[S_sender])

其中：

S_sender = sender 的 25 个 expert classes
z_sender[S_sender] = 已有 sender expert-class logits

sender KD 只在 sender expert classes 上计算，不使用 sender 的 full 100-class logits。

第一版超参数尽量复用现有 logits baseline：

temperature: 沿用当前 KD temperature
lambda_skd: 沿用当前 sender logits KD 权重，或从 config 显式读取

不要在第一版引入 entropy weighting、margin weighting、动态温度等额外机制。

## Generalist Logits 保存规则

新增 attach_generalist_logits 阶段后，每个 packet sample 应额外保存：

generalist_logits: shape [100]
generalist_model: 使用的 generalist 架构/ckpt 标识

保留原有字段：

packet image
global label
sender id
sender expert classes
sender expert-class logits

validate_packets.py 应能检查 generalist logits 是否存在、shape 是否为 100、样本数量是否与 packet 对齐。

## Receiver 训练损失

packet_generalist mode 下 receiver loss 为：

第一版沿用现有 packet-only receiver 训练口径：

L_R = L_CE_packet
    + lambda_kd  * L_KD_mix_external
    + lambda_fr  * L_FR

其中：

L_KD_mix_external = (1 - beta) * L_KD_sender_external
                  + beta       * L_KD_generalist_external

计算边界：

L_KD_sender:
  只在 external sender packet 上计算；
  只在 sender expert classes S_sender 上计算。

L_KD_generalist:
  只在 external sender packet 上计算；
  第一版使用 full 100-class logits 计算。

L_FR:
  沿用当前 communication / receiver training 中已有的 FR / retention 设计和 config。
  不要删除、重命名或绕过 FR。

说明：
  当前代码中的 L_CE_packet 包含 self packet 和 external packets。
  本任务第一版不加入 self real CE，后续如需验证 expert retention，可单独新增 self-real retention 变体。

第一版固定或复用：

beta = 0.5
temperature = 当前 KD temperature
lambda_kd = 当前 logits baseline 设置
lambda_fr = 当前 FR 设置

目标是先验证 packet generalist logits 是否带来收益，不要让超参数数量膨胀。

## 实验边界

本任务不是：

1. 重构 DSDM。
2. 复现完整 MASC。
3. 引入 server-based FL。
4. 用 full real data 训练 generalist。
5. 向 receiver 传递 generalist 参数、backbone、feature 或梯度。

本任务是：

在现有 packet communication 上增加一个 socialized feedback 版本：
packet images 仍是主要通信载体；
sender logits 提供局部 expert soft knowledge；
generalist logits 提供跨 sender 的 global class relation。

## 代码实现要求

1. 尽量复用现有 run_social_pipeline.py。
2. 新增 stage 建议为：

train_packet_generalist
attach_generalist_logits

3. receiver 训练新增可选 mode / config，例如：

communication:
  mode: packet_generalist
  use_sender_logits: true
  use_generalist_logits: true
  kd_mix_beta: 0.5

reuse:
  source_run_name: cifar100_4agent_25cls_conv_family_ipc10
  reuse_packets: true

4. 不破坏现有 packet methods：

dsdm
heuristic
importance
full_real

第一版优先支持 dsdm + logits + conv-family。

## 验证要求

实现后至少运行：

python -m py_compile run_social_pipeline.py social_trainer.py packet_logits.py validate_packets.py

并提供 dry-run，不自动启动大规模训练：

python run_social_pipeline.py --config <new_config> --stage train_packet_generalist --dry-run
python run_social_pipeline.py --config <new_config> --stage attach_generalist_logits --dry-run
python run_social_pipeline.py --config <new_config> --stage train_receivers --dry-run

如果需要新增配置文件，建议命名为：

configs/main_cifar100_conv_family_packet_generalist.yaml

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

## 2026-06-30 联合 receiver 超参数验证任务

本轮任务目标：

```text
在不删除、不覆盖既有可用结果的前提下，继续验证 IPC=10 与 IPC=50 的 receiver/social training 超参数。
```

当前可复用 packet：

```text
IPC=10 ConvNet family:
outputs/cifar100_4agent_25cls_conv_family_ipc10

IPC=50 all-ConvNet:
outputs/cifar100_4agent_25cls_ipc50_allconvnet
```

本轮不重新运行 DSDM 蒸馏，只复用已有 packet 和 expert checkpoint，执行通信验证阶段：

```text
1. build_communication
2. validate_packets
3. train_receivers
```

当前最佳点附近继续验证：

```text
IPC=10:
以 ep100 / lambda_fr=0.2 / lambda_kd=0.5 为中心，
重点检查相邻 epoch 与 KD 权重。

IPC=50:
以 ep500 / lambda_fr=0.05 / lambda_kd=0.5 为中心，
重点检查 receiver epoch 与 FR 权重。
```

所有新增实验必须使用新的 run_name，输出到新的 `outputs/` 子目录；已有结果只读复用，不允许删除或覆盖。

## 2026-06-30 19:35 follow-up receiver 调参

当前已完成第一批 refinement 后的最佳候选：

```text
IPC=10:
ep060 / lambda_fr=0.20 / lambda_kd=0.50
mean_global=33.905

IPC=50:
ep250 / lambda_fr=0.05 / lambda_kd=0.50
mean_global=46.1825
```

继续开启第二批 receiver-only follow-up 调参，不重新蒸馏 packet，只复用已有 packet 和 expert checkpoint。

新增队列脚本：

```text
scripts/run_receiver_followup_tuning.py
```

新增配置目录：

```text
configs/receiver_followup_tuning/
```

本批验证组合：

```text
IPC=10:
ep050 / fr=0.20 / kd=0.50
ep060 / fr=0.15 / kd=0.50
ep060 / fr=0.25 / kd=0.50
ep060 / fr=0.20 / kd=0.60

IPC=50:
ep225 / fr=0.05 / kd=0.50
ep275 / fr=0.05 / kd=0.50
ep250 / fr=0.04 / kd=0.50
ep250 / fr=0.06 / kd=0.50
```

运行方式采用 `max-jobs=1`，避免和正在运行的 Conv-family IPC=50 蒸馏实验过度抢资源。
