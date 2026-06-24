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

当前阶段的重点是通过验证实验确认当前性能瓶颈来源：

```text
1. 当前 CIFAR-100 class-disjoint 任务本身的性能上限是多少？
2. DSDM packet 性能较弱是因为 packet 本身质量不足，还是因为异构 receiver 难以吸收？
3. factor=2 的 compact synthetic image 是否带来明显性能损失？
4. agent 2 的严重 forgetting 是 loss 设置问题，还是 packet / 训练流程问题？
5. 在更低通信预算下，DSDM 是否体现出高知识密度优势？
```

目前用户已经完成同构实验。当前最优先只做两个 upper-bound 验证：

```text
1. Full Real Social Transfer
2. Centralized Full Data Upper Bound
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

当前主实验采用异构模型：

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet_ap",
}
```

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

推荐运行：

```bash
python run_centralized_full.py \
  --config configs/main_cifar100_logit.yaml \
  --model convnet \
  --epochs 500 \
  --lr 0.001 \
  --batch-size 128

python run_centralized_full.py \
  --config configs/main_cifar100_logit.yaml \
  --model resnet \
  --epochs 500 \
  --lr 0.001 \
  --batch-size 128

python run_centralized_full.py \
  --config configs/main_cifar100_logit.yaml \
  --model resnet_ap \
  --epochs 500 \
  --lr 0.001 \
  --batch-size 128
```

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
5. 当前优先做 Full Real Social Transfer 和 Centralized Full Data Upper Bound，确认任务上限和训练流程上限。
```
