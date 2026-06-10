# PROJECT_SPEC.md

## 1. 项目定义

本项目 v2.0 实现一个 **DSDM-based Agent-to-Agent Socialized Learning** 框架，用于研究 heterogeneous multi-agent setting 下的低通信知识传播问题。

核心设定：

```text
CIFAR-10
5 agents
每个 agent 初始只掌握 2 个 expert classes
每个 agent 蒸馏自己的 expert classes
agents 之间只传 distilled image packets
receiver agent 使用 packets 学习其他 agents 的类别
```

本项目 v2.0 不使用 generalist，不传参数，不传梯度，不做联邦聚合。

---

## 2. v2.0 与 v1 的关系

v1 中曾经探索过：

```text
generalist-guided learning
social-head-only training
soft target packet
partial-known split
```

这些不再作为 v2.0 主线。v2.0 重构时应优先清理这些旧逻辑，避免后续代码冗余。

推荐处理：

```text
legacy/v1_generalist/
legacy/v1_social_head/
legacy/v1_soft_target/
```

旧代码如仍有参考价值，可以移动到 legacy；没有价值的可以删除。

v2.0 主线应只保留：

```text
direct 5-agent split
agent-wise heterogeneous models
strict DSDM packet building
raw IPC baseline
agent-to-agent social learning
evaluation and comparison
```

---

## 3. 数据与任务设置

### 3.1 Dataset

```yaml
dataset:
  name: cifar10
  num_classes: 10
  image_size: [32, 32]
```

### 3.2 Agent split

```yaml
split:
  mode: direct
  num_agents: 5
  classes_per_agent: 2
```

固定划分：

```python
{
    0: {"expert_classes": [0, 1], "new_classes": [2, 3, 4, 5, 6, 7, 8, 9]},
    1: {"expert_classes": [2, 3], "new_classes": [0, 1, 4, 5, 6, 7, 8, 9]},
    2: {"expert_classes": [4, 5], "new_classes": [0, 1, 2, 3, 6, 7, 8, 9]},
    3: {"expert_classes": [6, 7], "new_classes": [0, 1, 2, 3, 4, 5, 8, 9]},
    4: {"expert_classes": [8, 9], "new_classes": [0, 1, 2, 3, 4, 5, 6, 7]},
}
```

说明：

- `expert_classes`: agent 第一阶段真实掌握的类别。
- `new_classes`: agent 从其他 agents packets 中学习的类别。
- `overall_classes`: `[0,1,2,3,4,5,6,7,8,9]`。

---

## 4. 模型设置

### 4.1 模型库

v2.0 使用三类模型：

```text
conv
resnet
resnet_ap
```

第一版工程细节沿用当前默认实现，不强制修改 ResNet 深度。

### 4.2 异构 agent 分配

```yaml
agent_models:
  0: conv
  1: resnet
  2: resnet_ap
  3: conv
  4: resnet
```

要求：

- 每个 agent 是完整分类模型。
- 每个 agent 输出 10-way logits。
- 不同 agent 可使用不同 backbone。

### 4.3 Head

v2.0 默认使用 **Shallow MLP Head**：

```text
Linear(in_dim, 512)
ReLU
Dropout(0.1)
Linear(512, num_classes)
```

配置：

```yaml
model:
  head_type: shallow_mlp
  head_hidden_dim: 512
  head_dropout: 0.1
```

建议实现：

```text
src/models/heads.py
```

至少支持：

```text
linear
shallow_mlp
deep_mlp
```

v2.0 默认必须为 `shallow_mlp`。

---

## 5. v2.0 工程阶段

## Stage 1: Expert Agent Training

目标：每个 agent 只用自己的 2 个 expert classes 训练完整模型。

输入：

```text
CIFAR-10 train dataset
agent expert_classes
agent model type
```

训练：

```text
expert real data -> backbone + shallow_mlp_head
```

输出：

```text
outputs/{experiment}/checkpoints/agents/agent_{id}_expert.pt
```

checkpoint 字段：

```python
{
    "agent_id": int,
    "expert_classes": List[int],
    "new_classes": List[int],
    "model_name": str,
    "model_state_dict": state_dict,
    "cfg": dict,
    "stage": "expert_train",
}
```

建议入口：

```bash
python -m src.main.run_train_agents_v2 --config CONFIG --agent-ids all
```

---

## Stage 2: DSDM Guide Checkpoint Pretraining

目标：为每个 sender agent 训练多个 strict DSDM guide checkpoints。

每个 guide checkpoint 只在 sender 的 2 个 expert classes 上训练。

默认：

```yaml
dsdm:
  guide_model_number: 10
  guide_pretrain_epochs: 20
```

输出：

```text
outputs/{experiment}/checkpoints/dsdm_guides/agent_{id}/guide_{k}.pt
```

建议入口：

```bash
python -m src.main.run_pretrain_dsdm_guides --config CONFIG --agent-ids all
```

---

## Stage 3: Packet Building

支持两种 packet source：

```text
raw
strict_dsdm
```

### raw packet

每个 sender class 随机采 IPC 张真实图像。

### strict DSDM packet

每个 sender agent 使用 guide checkpoint pool 蒸馏自己的 expert classes。

strict DSDM 关键点：

```text
multiple guide checkpoints
random guide selection per distillation iteration
prototype matching
covariance / semantic distribution matching
historical prototype smoothing
```

packet 格式：

```python
{
    "sender_id": int,
    "class_ids": List[int],
    "images": Tensor,
    "hard_labels": Tensor,
    "meta": {
        "packet_source": "raw" or "strict_dsdm",
        "ipc": int,
        "bytes_images": int,
        "bytes_labels": int,
        "bytes_total": int,
    },
}
```

v2.0 禁止保存：

```text
soft_targets
teacher_logits
teacher_probs
```

输出：

```text
outputs/{experiment}/packets/{packet_source}/agent_{id}_packet.pt
```

建议入口：

```bash
python -m src.main.run_build_packets_v2 --config CONFIG --packet-source raw
python -m src.main.run_build_packets_v2 --config CONFIG --packet-source strict_dsdm
```

---

## Stage 4: Agent-to-Agent Social Learning

目标：receiver agent 使用自己的 expert real data + 其他 agents packets 学习全 10 类。

例如 agent_0：

```text
real data: classes [0, 1]
packet data: classes [2, 3, 4, 5, 6, 7, 8, 9]
```

训练数据必须 class-balanced。

### Phase A: Head Warm-up

```text
freeze backbone
train shallow_mlp_head
loss = CE(real expert + packet images)
```

### Phase B: Last-block Adaptation

```text
freeze early backbone
unfreeze last block + shallow_mlp_head
loss = CE + lambda_anchor * ||theta_trainable - theta_old||^2
```

last block 定义：

```text
conv: 最后一个 conv block
resnet: layer4
resnet_ap: layer4
```

输出：

```text
outputs/{experiment}/checkpoints/socialized/{packet_source}/agent_{id}_socialized.pt
```

建议入口：

```bash
python -m src.main.run_socialize_agents_v2 --config CONFIG --packet-source strict_dsdm --agent-ids all
```

---

## 6. Baselines

v2.0 必须支持：

1. `local_only`
   - agent 只训练自己的 2 类，不接收 packet。

2. `raw_ipc`
   - 其他 agents 每类发送 IPC 张真实图。

3. `strict_dsdm_ipc`
   - 其他 agents 每类发送 IPC 张 strict DSDM 蒸馏图。

公平性要求：

```text
raw_ipc 和 strict_dsdm_ipc 使用完全相同的 receiver social learning 训练策略。
```

---

## 7. Evaluation

### 7.1 Agent metrics

每个 agent 输出：

```python
{
    "agent_id": int,
    "model_name": str,
    "expert_classes": List[int],
    "new_classes": List[int],
    "expert_accuracy": float,
    "new_accuracy": float,
    "overall_accuracy": float,
}
```

### 7.2 Summary metrics

```python
{
    "average_expert_accuracy": float,
    "average_new_accuracy": float,
    "average_overall_accuracy": float,
}
```

建议入口：

```bash
python -m src.main.run_eval_v2 --config CONFIG --checkpoint-stage expert
python -m src.main.run_eval_v2 --config CONFIG --checkpoint-stage socialized --packet-source strict_dsdm
```

---

## 8. Compare

比较：

```text
local_only
raw_ipc
strict_dsdm_ipc
```

输出：

```python
{
    "method": str,
    "average_expert_accuracy": float,
    "average_new_accuracy": float,
    "average_overall_accuracy": float,
    "total_comm_bytes": int,
    "compression_ratio_vs_full_raw": float,
}
```

建议入口：

```bash
python -m src.main.run_compare_v2 --config CONFIG
```

---

## 9. Communication Metrics

必须统计：

```text
bytes_per_packet
total_comm_bytes
total_transmitted_images
compression_ratio_vs_full_raw
accuracy_per_mb
```

packet byte 估计：

```python
bytes_images = packet.images.nelement() * packet.images.element_size()
bytes_labels = packet.hard_labels.nelement() * packet.hard_labels.element_size()
bytes_total = bytes_images + bytes_labels
```

v2.0 暂不统计 soft target bytes，因为不传 soft target。

---

## 10. Recommended v2.0 Config Skeleton

```yaml
seed: 1
device: cuda:0

experiment:
  name: cifar10_5agent_v2

output:
  root: outputs/v2

dataset:
  name: cifar10
  root: ./datasets
  num_classes: 10
  image_size: [32, 32]

split:
  mode: direct
  num_agents: 5
  classes_per_agent: 2

agent_models:
  0: conv
  1: resnet
  2: resnet_ap
  3: conv
  4: resnet

model:
  head_type: shallow_mlp
  head_hidden_dim: 512
  head_dropout: 0.1
  norm_type: instance

train:
  batch_size: 128
  epochs: 100
  lr: 0.01

packet:
  ipc: 10
  source: strict_dsdm

dsdm:
  strict: true
  guide_model_number: 10
  guide_pretrain_epochs: 20
  distill_steps: 10000
  cov_weight: 50.0
  h_p_weight: 0.2
  smooth_factor: 0.99

social:
  phase_a_epochs: 20
  phase_b_epochs: 20
  lr_head: 0.01
  lr_last_block: 0.001
  samples_per_class: 8
  steps_per_epoch: 100
  lambda_anchor: 0.01
```

---

## 11. 当前优先任务顺序

1. v2.0 cleanup：迁移或删除 v1 旧主线。
2. 配置系统：新增 `configs/v2/`。
3. 模型系统：新增 configurable heads + agent-wise model builder。
4. expert agent training。
5. strict DSDM guide pretraining。
6. packet building: raw + strict_dsdm。
7. socialize agents: head warm-up + last-block adaptation。
8. eval v2。
9. compare v2。
