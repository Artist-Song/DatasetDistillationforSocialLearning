# AGENTS.md

## 项目定位

本项目名为 `DatasetDistillationforSocialLearning`。

当前研究方向是：面向异构社会化学习的高知识密度蒸馏通信包构建方法。

核心目标是在多智能体社会化学习场景中，让不同 agent 之间通过输入空间 knowledge packet 进行知识传递，而不是依赖模型参数、梯度或中间特征通信。由于不同 agent 可能采用不同模型结构，模型空间知识难以直接对齐；因此，本项目以图像形式的 packet 作为异构模型之间的共同通信载体，并进一步利用数据集蒸馏提高单位通信量中的知识密度。

当前课题暂定名称：

```text
Distilled Knowledge Packets for Communication-Efficient Heterogeneous Socialized Learning
```

中文名称：

```text
面向异构社会化学习的高知识密度蒸馏通信包构建方法
```

## 当前研究阶段

当前项目已经从 CIFAR-10 初步验证阶段进入 CIFAR-100 诊断验证阶段。

截至 2026-06-25，已经完成两类 upper-bound 诊断：

```text
1. Full Real Social Transfer
2. Centralized Full Data Upper Bound
```

诊断结论是：旧 ResNet / ResNet_AP 上界过低主要来自训练 recipe 不适配，而不是模型本身能力不足。修正训练 recipe 后，ResNet-10 / ResNetAP-10 在完整 CIFAR-100 上可以达到 72%+。ConvNet-3 按严格 DSDM 训练 recipe 可达到 65%+。

当前下一阶段研究重点转向：

```text
1. 在更干净的 ConvNet family 异构设定下验证 DSDM packet 的社会化学习效果。
2. 在 IPC=10 等更低通信预算下比较 DSDM / DSDM_LOGIT / Heuristic / Full Real。
3. 分析 DSDM 的高知识密度优势是否能在低通信预算下超过随机真实样本核心集。
4. 继续观察 receiver forgetting，并判断是否需要调整 receiver loss。
```

## 当前核心实验设定

### 数据集

```text
CIFAR-100
```

### agent 数量

```text
4 个 agent
```

### 类别划分

采用 class-disjoint split，每个 agent 拥有 25 个 expert classes。

```python
agent_class_split = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}
```

所有标签必须保留 CIFAR-100 全局标签，不允许重映射。

### 模型异构设定

当前已有跨架构异构诊断模型：

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet_ap",
}
```

下一阶段主实验优先采用 ConvNet family 内部异构，以降低跨架构迁移噪声，同时保留参数、梯度和中间特征不可直接共享的问题：

```python
agent_model_split = {
    0: "convnet",  # ConvNet-3-w0.5
    1: "convnet",  # ConvNet-3-w1.0
    2: "convnet",  # ConvNet-4-w1.0
    3: "convnet",  # ConvNet-4-w1.5
}
```

ConvNet family 异构需要通过 per-agent model config 显式指定 depth / width / norm_type。

所有模型输出维度固定为：

```text
100
```

即使某个 agent 只训练 25 个 expert classes，也不能把分类头改成 25 维。

## 当前应支持的 packet method

当前代码应支持：

```text
dsdm
heuristic
importance
full_real
```

其中：

```text
dsdm       : 数据集蒸馏生成 synthetic image packet
heuristic  : 每类随机选择真实样本构建 packet
importance : 根据 guide model 置信度选择真实样本构建 packet
full_real  : 直接传输 agent 全部真实训练数据，用作 oracle / upper-bound baseline
```

`full_real` 不是低通信方法，也不是本文目标方法，只用于验证通信充分条件下的社会化训练上限。

## 当前 pipeline 阶段

主入口为：

```text
run_social_pipeline.py
```

支持阶段：

```text
train_experts
distill_packets
build_selection_packets
attach_logits
build_communication
train_receivers
all
```

## Full Real Social Transfer

目的：

```text
验证在通信完全充足时，当前 social training 流程的性能上限。
```

CIFAR-100 当前设定下：

```text
每个 sender full_real packet = 25 类 × 500 张 = 12500 张
每个 receiver external images = 3 × 12500 = 37500 张
receiver 训练时读取 self + external = 50000 张完整 CIFAR-100 训练图
```

推荐运行：

```bash
python run_social_pipeline.py \
  --config configs/main_cifar100_logit.yaml \
  --stage build_selection_packets \
  --packet-method full_real

python run_social_pipeline.py \
  --config configs/main_cifar100_logit.yaml \
  --stage build_communication \
  --packet-method full_real

python validate_packets.py \
  --config configs/main_cifar100_logit.yaml \
  --packet-method full_real

python run_social_pipeline.py \
  --config configs/main_cifar100_logit.yaml \
  --stage train_receivers \
  --packet-method full_real \
  --init-mode expert
```

第一版不强制跑 full_real logits。Full Real 的核心作用是测数据充分条件下的 social training 上限。

## Centralized Full Data Upper Bound

目的：

```text
验证 ConvNet / ResNet / ResNet_AP 在完整 CIFAR-100 数据上的基础能力上限。
```

该实验不属于 social learning，不走 packet，不走 manifest。

推荐脚本：

```text
run_centralized_full.py
```

推荐运行采用 config 中的模型专属 recipe，不再手动传旧 lr：

```bash
python run_centralized_full.py \
  --config configs/main_cifar100_upper_bound.yaml \
  --model convnet

python run_centralized_full.py \
  --config configs/main_cifar100_upper_bound.yaml \
  --model resnet

python run_centralized_full.py \
  --config configs/main_cifar100_upper_bound.yaml \
  --model resnet_ap
```

当前已记录的新 centralized upper-bound 结果：

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

旧的 50.89 / 28.69 / 33.28 结果保留在 `centralized_results.csv`，仅作为训练 recipe 诊断对照。

## 下一阶段：ConvNet Family 异构社会化实验

目的：当前 ConvNet / ResNet / ResNetAP 跨架构异构包含较强的架构迁移噪声。下一阶段先采用 ConvNet family 内部 depth-width 容量异构，保留模型参数形状、特征维度和容量差异，同时降低跨架构蒸馏迁移干扰。

日期：2026-06-25

推荐 agent 结构：

```text
agent_0: ConvNet-3-w0.5   弱模型
agent_1: ConvNet-3-w1.0   标准模型
agent_2: ConvNet-4-w1.0   更深模型
agent_3: ConvNet-4-w1.5   强模型
```

推荐对比方法：

```text
Expert Only / Before Social
Full Real Social Transfer
Heuristic Real Packet, IPC=10
DSDM Packet, IPC=10
DSDM Packet + Logits, IPC=10
```

本组实验只改变 packet method，不改变评价指标口径。重点验证 DSDM 在更低通信预算 IPC=10 下是否相对随机真实样本核心集体现更高单位通信知识密度。


## DSDM factor 规则

DSDM 当前默认使用：

```text
factor = 2
```

含义是：

```text
1 张 raw synthetic image
→ decode
→ 2 × 2 = 4 张 effective training images
```

通信量统计时只统计 raw synthetic image 数量，不统计 decode 后的 effective image 数量。

例如 CIFAR-100 IPC=50：

```text
每个 sender raw packet images = 25 × 50 = 1250
每个 receiver external raw images = 3 × 1250 = 3750
```

对于 DSDM factor=2：

```text
每个 receiver external effective training images = 3750 × 4 = 15000
```

但是通信量仍记为：

```text
external_comm_images = 3750
```

## DSDM feature index 规则

DSDM 的 `f_idx` / `idx_from` 语义应理解为：

```text
选择 sender guide model 的深层语义特征做 synthetic image matching。
```

当前研究口径下，默认应取“最后一个非 logits 的语义特征层”，而不是固定使用某个硬编码层号。不同模型的 feature index 不一致，必须按模型结构显式确认：

```text
ConvNet-3:
  idx 0 = block1 output
  idx 1 = block2 output
  idx 2 = block3 output = 最后一层特征

ConvNet-4:
  idx 0 = block1 output
  idx 1 = block2 output
  idx 2 = block3 output
  idx 3 = block4 output = 最后一层特征

ResNet / ResNet_AP:
  idx 0 = layer0
  idx 1 = layer1
  idx 2 = layer2
  idx 3 = layer3
  idx 4 = layer4 = 最后 residual feature map
  idx 5 = avgpool 后 penultimate vector
  idx 6 = logits
```

因此推荐规则是：

```text
ConvNet-3      -> idx_from = 2
ConvNet-4      -> idx_from = 3
ResNet-10      -> idx_from = 4
ResNetAP-10    -> idx_from = 4
```

如果实验明确要匹配分类器前向量，可单独使用 ResNet / ResNetAP 的 `idx_from = 5`，但不能和默认 image-packet 主实验混为一组。

注意：DSDM 实际蒸馏使用的是 `args.idx_from / args.idx_to`，不是单独显示的 `args.f_idx`。如果在 per-agent model config 中根据 depth / family 刷新了 `f_idx`，必须同步刷新 `idx_from / idx_to`，否则日志或 dry-run 里看到的 `f_idx` 正确也不代表实际匹配层正确。

截至 2026-07-03 已确认：

```text
1. 旧跨架构 hetero CIFAR-100 IPC=50 中，ResNet / ResNetAP sender 使用 idx_from=2，
   这实际对应 ResNet layer2，不是最后语义层，可能是 ResNet packet 异常的重要原因之一。

2. 当前正在运行的 Conv-family IPC=50 DSDM 蒸馏中：
   agent_0 ConvNet-3-w0.5 使用 idx_from=2，逻辑正确；
   agent_1 ConvNet-3-w1.0 使用 idx_from=2，逻辑正确；
   agent_2 ConvNet-4-w1.0 实际仍使用 idx_from=2，属于旧 feature-index baseline；
   agent_3 ConvNet-4-w1.5 实际仍使用 idx_from=2，属于旧 feature-index baseline。

3. 当前运行中的 Conv-family IPC=50 结果先保留为旧逻辑 baseline；
   后续优先只给 agent_2 / agent_3 补跑 last-feature 版本即可，
   即 ConvNet-4 sender 使用 idx_from=3。
```

## logits 规则

logits 是增强模块，不是第一层研究动机。

第一层研究动机是：

```text
异构兼容性 + 高知识密度通信
```

logits 的作用是：

```text
在 distilled image packet 基础上补充 sender expert-class soft decision information。
```

当前 logits 保存规则：

```text
只保存 sender 自己 expert classes 上的 logits
不保存 full 100-class logits
不保存模型参数、梯度或中间特征
```

## 当前已有实验现象

当前 CIFAR-100 IPC=50 实验显示：

```text
1. DSDM_LOGIT 相比 DSDM image-only 平均 global accuracy 和 new accuracy 有提升。
2. HEURISTIC_LOGIT 在 IPC=50 下平均性能高于 DSDM_LOGIT。
3. IMPORTANCE_LOGIT 明显较弱。
4. DSDM_LOGIT 在 ResNet / ResNet_AP receiver 上的 new-class absorption 有一定优势。
5. agent 2 存在严重 expert forgetting。
6. 旧 ResNet / ResNetAP upper-bound 过低由 recipe 导致；修正后 ResNet-10 / ResNetAP-10 上界恢复到 72%+。
7. ConvNet-3-IN 按严格 DSDM recipe 上界为 65.24，高于旧训练 recipe 的 50.89。
8. 2026-07-03 发现 DSDM feature-index 诊断问题：
   ResNet / ResNetAP 不应默认沿用 idx_from=2；
   ConvNet-4 也应使用 idx_from=3 才是最后特征层。
9. Conv-all DSDM_LOGIT packet 复用到 hetero receivers 后，
   agent2 ResNet 的 expert acc 从原 hetero packet 的约 0.8% 恢复到 33.68%，
   支持“原 hetero packet 图像分布/feature-index/蒸馏端问题”是异常主因之一。
```

当前不能直接声称：

```text
DSDM 在 IPC=50 下全面优于真实样本核心集。
```

更稳妥的结论是：

```text
IPC=50 对 CIFAR-100 来说通信预算相对较高，真实样本随机子集已经具有较强竞争力；DSDM 的高知识密度优势需要在 IPC=10、IPC=5、IPC=1 等更低通信预算下进一步验证。
```

## 代码修改原则

1. 尽量复用现有 pipeline。
2. 不大规模重构 DSDM 源码。
3. 新验证实验优先通过新增 packet method、config 和小模块完成。
4. 不要把 full_real 和 centralized_full 的逻辑混到 DSDM 主算法内部。
5. 所有新增函数和类必须有简短中文注释。
6. 每次修改后至少运行 py_compile 和 dry-run。
7. 可以按用户本次要求提交并 push 指导文件和实验代码。
8. 不要强制 reset、rebase、clean，除非用户明确要求。
9.修改前要告诉用户修改的逻辑与方案
10. 修改 DSDM feature-index 逻辑时，必须同时验证 `f_idx` 与 `idx_from / idx_to`，
    并用 dry-run 或小脚本打印每个 agent 的最终解析值；
    仅修改配置里的 `f_idx` 不足以保证实际蒸馏层改变。

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

dry-run：

```bash
python run_social_pipeline.py \
  --config configs/main_cifar100_logit.yaml \
  --stage build_selection_packets \
  --packet-method full_real \
  --dry-run

python run_social_pipeline.py \
  --config configs/main_cifar100_logit.yaml \
  --stage build_communication \
  --packet-method full_real \
  --dry-run
```

## 当前汇报口径

当前对老师汇报时应强调：

```text
1. CIFAR-100 4-agent 25-class class-disjoint split 本身较难。
2. IPC=50 下 Heuristic 较强，说明该通信预算下真实样本子集已经具备较强竞争力。
3. DSDM 的核心价值应在更低通信预算下验证。
4. agent 2 的 forgetting 暴露出 receiver loss 需要调整。
5. upper-bound 诊断已完成，当前重点转向 ConvNet family 异构 IPC=10 主实验。
6. 汇报 centralized upper-bound 时区分旧 recipe 诊断结果和新 recipe 上界结果。
```
