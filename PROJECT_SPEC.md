# PROJECT_SPEC.md

## 1. 项目定义

本项目实现一个 **generalist-guided distilled social packet learning** 框架，用于研究 heterogeneous multi-agent social learning 中的低通信知识补全问题。

每个 specialist agent 初始只掌握部分类别。我们先构建一个掌握全类知识的 generalist teacher agent，再由该 teacher 生成低通信量 distilled social packets，分发给各 specialist agent 的 missing classes。specialist 在第二阶段冻结 backbone 和 local head，仅训练 social classifier head，从而补齐缺失类别。

工程原则：**在现有代码框架上新增接口，不大改、不删除、不破坏旧实验。**

---

## 2. 与现有工程的关系

当前仓库已有：

- CIFAR dataset 工具；
- direct split；
- local pretrain；
- raw / dsdm packet build；
- social train；
- packet-only train；
- eval / compare / summarize；
- conv / resnet / resnet_ap model pool；
- DSDM-style simple distiller。

新工程不要推翻这些模块，而是在其上新增：

- partial known/missing split；
- generalist teacher training；
- generalist class-wise packet build；
- backbone + local_head + social_head 接口；
- social_head_only_balanced training；
- known/missing/general evaluation；
- communication and packet diagnostics。

旧 direct split 与旧 packet-only 逻辑必须继续保留。

---

## 3. 数据划分

### 3.1 CIFAR10 主线

默认配置：

```yaml
dataset:
  name: cifar10
  num_classes: 10

split:
  mode: partial_known
  strategy: fixed_cifar10_5x6
  num_specialists: 5
  known_classes_per_agent: 6
```

固定划分：

```python
{
    0: {
        "known_classes": [0, 1, 2, 3, 4, 5],
        "missing_classes": [6, 7, 8, 9],
        "all_classes": list(range(10)),
    },
    1: {
        "known_classes": [2, 3, 4, 5, 6, 7],
        "missing_classes": [0, 1, 8, 9],
        "all_classes": list(range(10)),
    },
    2: {
        "known_classes": [4, 5, 6, 7, 8, 9],
        "missing_classes": [0, 1, 2, 3],
        "all_classes": list(range(10)),
    },
    3: {
        "known_classes": [0, 1, 6, 7, 8, 9],
        "missing_classes": [2, 3, 4, 5],
        "all_classes": list(range(10)),
    },
    4: {
        "known_classes": [0, 1, 2, 3, 8, 9],
        "missing_classes": [4, 5, 6, 7],
        "all_classes": list(range(10)),
    },
}
```

### 3.2 CIFAR100 后续

后续支持：

```text
known 80 / missing 20
known 60 / missing 40
```

第一版不用实现 CIFAR100 partial split，但接口设计必须可扩展。

---

## 4. 工程阶段

## Stage 1A: Specialist Local Pretraining

目标：每个 specialist 使用自己的 known classes 训练 local anchor。

输入：

- train dataset；
- split plan；
- agent known_classes。

训练：

```text
known real data -> backbone + local_head
```

输出：

```text
outputs/{root}/checkpoints/specialist_local/{run_name}/agent_{id}_local.pt
```

checkpoint 字段：

```python
{
    "agent_id": int,
    "known_classes": List[int],
    "missing_classes": List[int],
    "model_state_dict": ...,
    "cfg": dict,
    "stage": "specialist_local",
}
```

建议入口：

```bash
python -m src.main.run_train_specialists --config CONFIG --agent-ids all
```

兼容方案：

- 可以复用/扩展旧 `run_local_pretrain.py`；
- 但不要改变旧 direct split 默认行为。

---

## Stage 1B: Generalist Teacher Training

目标：构建一个掌握所有类别的 generalist teacher。

输入：

- full train dataset。

训练：

```text
all real data -> teacher model
```

输出：

```text
outputs/{root}/checkpoints/generalist/{run_name}/generalist_teacher.pt
```

checkpoint 字段：

```python
{
    "agent_id": "generalist",
    "known_classes": all_classes,
    "model_state_dict": ...,
    "cfg": dict,
    "stage": "generalist",
}
```

建议入口：

```bash
python -m src.main.run_train_generalist --config CONFIG
```

通信统计：

```text
stage1_comm_bytes = bytes of raw images used to construct generalist
```

Stage 1 通信成本单独记录，不作为 Stage 2 低通信贡献的主指标。

---

## Stage 2A: Generalist Packet Building

目标：generalist teacher 生成 class-wise packets。

支持 packet source：

```text
global_raw_packet
global_dsdm_packet
```

第一版优先实现 `global_raw_packet`。

输出路径：

```text
outputs/{root}/packets/generalist/{packet_source}/{run_name}/class_{c}_packet.pt
```

packet 格式：

```python
SocialPacket(
    sender_id="generalist",
    sender_type="generalist",
    class_ids=[c],
    images=Tensor[ipc, C, H, W],
    hard_labels=Tensor[ipc],
    soft_targets=Tensor[ipc, num_classes],
    meta={
        "packet_source": "global_raw_packet" or "global_dsdm_packet",
        "ipc": int,
        "class_id": int,
        "teacher_model": str,
        "bytes_images": int,
        "bytes_soft_targets": int,
        "bytes_total": int,
    },
)
```

如果当前 `SocialPacket` dataclass 没有 `sender_type`，请向后兼容添加默认值，不要破坏旧 packet。

建议入口：

```bash
python -m src.main.run_build_generalist_packets --config CONFIG
```

---

## Stage 2B: Social Head Training

目标：specialist 使用 missing-class packets 训练 social_head。

输入：

- specialist local checkpoint；
- split plan；
- receiver known_classes / missing_classes；
- missing class packets。

模型规则：

```text
freeze backbone
freeze local_head
init social_head from local_head
train social_head only
```

训练数据：

```text
known classes: local real data
missing classes: generalist packets
```

batch 构造：

```text
每步：
  known 每类采 samples_per_class 张 real images
  missing 每类采 samples_per_class 张 packet images
```

packet 样本可 replacement sampling。

loss：

```text
L = lambda_packet_ce * L_packet_ce
  + lambda_packet_kd * L_packet_kd
  + lambda_known_ce * L_known_ce
  + lambda_retain * L_retain
```

默认：

```yaml
social_head:
  train_mode: social_head_only_balanced
  epochs: 30
  lr: 0.005
  samples_per_class: 8
  steps_per_epoch: 100
  lambda_packet_ce: 1.0
  lambda_packet_kd: 0.1
  lambda_known_ce: 0.2
  lambda_retain: 0.05
  temperature: 2.0
```

输出：

```text
outputs/{root}/checkpoints/social_head/{run_name}/agent_{id}_social_head.pt
```

建议入口：

```bash
python -m src.main.run_train_social_head --config CONFIG --agent-ids all
```

---

## 5. 模型接口要求

现有模型尽量少改，通过 wrapper 或兼容方法新增接口。

建议在 `AgentModel` 或新 wrapper 中支持：

```python
forward_features(x)
forward_head(features, head="local")
forward(x, head="local")
init_social_head_from_local()
freeze_backbone()
freeze_local_head()
train_social_head_only()
```

最低可接受方案：

- 第一版只支持 `conv`；
- `resnet` 支持可第二步实现；
- 如果底层模型难拆，可以先通过 wrapper 暴露 feature extractor 和 classifier。

不要一次性重构全部 model pool。

---

## 6. Evaluation

新增或扩展 eval，支持：

```text
checkpoint-stage:
  local
  generalist
  social_head
```

对每个 specialist 输出：

```python
{
    "agent_id": int,
    "known_classes": [...],
    "missing_classes": [...],
    "known_accuracy": float,
    "missing_accuracy": float,
    "general_accuracy": float,
}
```

summary：

```python
{
    "average_known_accuracy": float,
    "average_missing_accuracy": float,
    "average_general_accuracy": float,
}
```

建议入口：

```bash
python -m src.main.run_eval_specialists --config CONFIG --checkpoint-stage local
python -m src.main.run_eval_specialists --config CONFIG --checkpoint-stage social_head
```

---

## 7. Compare

比较：

```text
local -> social_head
```

输出：

```python
{
    "delta_average_known_accuracy": ...,
    "delta_average_missing_accuracy": ...,
    "delta_average_general_accuracy": ...,
    "stage1_comm_bytes": ...,
    "stage2_comm_bytes": ...,
    "accuracy_per_mb_stage2": ...,
}
```

建议入口：

```bash
python -m src.main.run_compare_generalist --config CONFIG
```

---

## 8. Communication Metrics

必须统计：

```text
bytes_per_packet
stage1_comm_bytes
stage2_comm_bytes
total_comm_bytes
accuracy_per_mb_stage2
accuracy_per_mb_total
```

第一版可用 tensor 实际内存估计：

```python
num_bytes = tensor.nelement() * tensor.element_size()
```

对 packet：

```python
bytes_images = packet.images.nelement() * packet.images.element_size()
bytes_soft_targets = packet.soft_targets.nelement() * packet.soft_targets.element_size()
bytes_labels = packet.hard_labels.nelement() * packet.hard_labels.element_size()
bytes_total = bytes_images + bytes_soft_targets + bytes_labels
```

Stage 2 receiver 通信量：

```text
sum(bytes_total of packets for receiver missing_classes)
```

---

## 9. Packet Diagnostics

后续新增：

```bash
python -m src.main.run_packet_diagnostics --config CONFIG --agent-ids all
```

输出：

```text
teacher_packet_accuracy
receiver_anchor_packet_accuracy
receiver_post_packet_accuracy
packet_readability_gap
```

用途：

- 判断 packet 是否只对 teacher 可读；
- 判断异构 receiver 是否能读取 teacher-generated packet；
- 诊断 DSDM packet 的跨模型可迁移性。

---

## 10. Baselines

第一版：

1. `local_only`
2. `global_raw_packet`
3. `global_dsdm_packet`

第二版：

4. `global_raw_missing_share`
5. `full_finetune_receiver`
6. `peer_dsdm_packet`

---

## 11. 推荐配置文件

新增：

```text
configs/exp/2606-cifar10_partial6_global_raw_packet.yaml
```

内容模板：

```yaml
seed: 42
device: cuda

dataset:
  name: cifar10
  root: ./data
  image_size: [32, 32]
  num_classes: 10

split:
  mode: partial_known
  strategy: fixed_cifar10_5x6
  num_specialists: 5
  known_classes_per_agent: 6

models:
  generalist:
    name: conv
    norm_type: instance
  specialists:
    default:
      name: conv
      norm_type: instance

train:
  batch_size: 128
  lr: 0.01
  local_epochs: 50
  generalist_epochs: 50

packet:
  source: global_raw_packet
  ipc: 10
  temperature: 2.0

social_head:
  train_mode: social_head_only_balanced
  epochs: 30
  lr: 0.005
  samples_per_class: 8
  steps_per_epoch: 100
  lambda_packet_ce: 1.0
  lambda_packet_kd: 0.1
  lambda_known_ce: 0.2
  lambda_retain: 0.05
  freeze_backbone: true
  freeze_local_head: true
  init_social_head: copy_local_head

output:
  root: ./outputs/2606-cifar10_partial6_global_raw_packet
```

之后新增：

```text
configs/exp/2606-cifar10_partial6_global_dsdm_packet.yaml
```

将：

```yaml
packet:
  source: global_dsdm_packet
```

并加入 DSDM 参数。

---

## 12. 最小验收流程

第一版 raw packet 验收命令：

```bash
python -m src.main.run_train_generalist \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml

python -m src.main.run_train_specialists \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml \
  --agent-ids all

python -m src.main.run_build_generalist_packets \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml

python -m src.main.run_train_social_head \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml \
  --agent-ids all

python -m src.main.run_eval_specialists \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml \
  --checkpoint-stage local

python -m src.main.run_eval_specialists \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml \
  --checkpoint-stage social_head

python -m src.main.run_compare_generalist \
  --config configs/exp/2606-cifar10_partial6_global_raw_packet.yaml
```

验收：

- local known_accuracy 高；
- local missing_accuracy 低；
- social missing_accuracy 明显提升；
- social general_accuracy 提升；
- known_accuracy 不严重下降；
- JSON report 正常保存；
- compare report 正常保存。

---

## 13. 不允许做的事

- 不允许删除旧实验入口。
- 不允许把旧 direct split 默认行为改成 partial split。
- 不允许把旧 packet-only 逻辑改坏。
- 不允许一次性重写全部模型池。
- 不允许第一版就强行实现所有 baseline。
- 不允许把 DSDM、异构、通信诊断全部混在第一步实现。

---

## 14. 第一阶段 Codex 任务拆分

### Task 1: split

实现 partial known split，并加最小测试打印。

### Task 2: generalist train

新增 generalist training 入口。

### Task 3: specialist train

新增或兼容 specialist local training。

### Task 4: raw packet

实现 class-wise global raw packet build。

### Task 5: social head model

在不大改模型池的情况下支持 local/social head。

### Task 6: social head train

实现 class-balanced social head training。

### Task 7: eval compare

实现 known/missing/general eval 和 compare。

### Task 8: dsdm packet

接入已有 simple_distiller。

### Task 9: heterogeneity

支持 teacher / specialist 不同 model name。

### Task 10: diagnostics

实现 packet diagnostics。
