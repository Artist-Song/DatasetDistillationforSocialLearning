# AGENTS.md

## 0. Codex 最高优先级原则

本仓库当前进入 **v2.0 大版本重构**。

v2.0 主线已经从旧版 `generalist-guided / social-head-only / soft-target packet` 方案，切换为：

```text
DSDM-based Agent-to-Agent Socialized Learning
```

请严格遵守：

1. **不要继续沿用 generalist 设计**。
   - 不训练 generalist teacher。
   - 不构建 generalist packets。
   - 不做 generalist-guided social head training。

2. **不要继续沿用 soft target 主线**。
   - v2.0 第一版 packet 只包含 `images + hard_labels + meta`。
   - 不保存 `soft_targets`、`teacher_logits`、`teacher_probs`。
   - 后续如果需要 soft target，只能作为 v2.1 扩展，不进入当前主线。

3. **不要继续沿用 social-head-only 方案**。
   - v2.0 不采用 `freeze backbone + freeze local head + train social head only`。
   - v2.0 social learning 使用 `head warm-up + last-block adaptation`。

4. **先清理旧代码，再搭建 v2.0 主线**。
   - 旧 generalist / social head / partial-known / soft-target 相关代码应移动到 `legacy/` 或明确标记 deprecated。
   - 不要在旧逻辑上继续堆新逻辑。

5. **每次修改必须保持工程可运行**。
   - 每个任务尽量小步提交。
   - 修改后至少保证 import 不报错。
   - 配置、入口脚本、输出路径命名要一致。

一句话：

> 当前任务不是兼容 v1，而是建立清晰的 v2.0 主线；旧代码可迁移到 legacy，但不要继续污染主线。

---

## 1. v2.0 项目目标

实现一个基于 PyTorch 的低通信多 agent 社会化学习框架：

```text
CIFAR-10
5 agents
每个 agent 初始只掌握 2 个 expert classes
每个 agent 使用 DSDM strict distillation 蒸馏自己的 expert classes
agent-to-agent 互传 distilled packets
receiver agent 使用自己的 expert real data + 其他 agents packets 进行增量学习
```

研究问题：

- DSDM distilled packets 能否替代 raw image sharing？
- 每类只传 IPC 张蒸馏图时，agent 能否学习其他 agents 的类别？
- receiver agent 如何在学习新类时保持自身 expert classes？
- 异构模型之间只传图像 packet、不传参数，是否仍能完成知识传播？

---

## 2. 不再作为 v2.0 主线的内容

以下内容不要继续扩展，若仍有参考价值，请移动到 `legacy/`：

```text
run_train_generalist.py
run_build_generalist_packets.py
run_train_social_head.py
SocialHeadAgent / social_head_model.py
generalist comparison / eval scripts
partial-known split main configs
soft-target packet main dependency
packet-only generalist workflow
```

注意：

- 可以保留到 `legacy/v1_generalist/` 或 `legacy/v1_social_head/`。
- 移动后如有 import 断裂，需要修复主线 import。
- v2.0 主线目录里不要出现会误导 Codex 的 generalist-first 入口。

---

## 3. v2.0 固定实验设置

### 3.1 Dataset

```yaml
dataset:
  name: cifar10
  num_classes: 10
  image_size: [32, 32]
```

### 3.2 Split

```yaml
split:
  mode: direct
  num_agents: 5
  classes_per_agent: 2
```

固定划分：

```python
agent_0: [0, 1]
agent_1: [2, 3]
agent_2: [4, 5]
agent_3: [6, 7]
agent_4: [8, 9]
```

语义：

- `expert_classes`: agent 自己的 2 个类。
- `new_classes`: 其他 8 个类。
- `overall_classes`: CIFAR-10 全部 10 类。

---

## 4. v2.0 模型库与异构方案

模型库固定为：

```text
conv
resnet
resnet_ap
```

第一版工程细节沿用当前默认实现，不强制修改 resnet depth。

主异构分配：

```yaml
agent_models:
  0: conv
  1: resnet
  2: resnet_ap
  3: conv
  4: resnet
```

要求：

- 每个 agent 都是完整分类模型，不是裸 backbone。
- 每个 agent 输出 CIFAR-10 的 10-way logits。
- 不同 agent 可以使用不同 backbone。
- 通信只传 packet images，不传模型参数、梯度或 prototypes。

---

## 5. v2.0 Head 设计

v2.0 默认分类头使用 **Shallow MLP Head**。

结构：

```python
Linear(in_dim, 512)
ReLU
Dropout(0.1)
Linear(512, num_classes)
```

推荐配置：

```yaml
model:
  head_type: shallow_mlp
  head_hidden_dim: 512
  head_dropout: 0.1
```

要求：

- 将当前较重的 `MLPClassifier` 替换为可配置 head。
- 至少支持：
  - `linear`
  - `shallow_mlp`
  - `deep_mlp`（仅作为 ablation，不作为默认）
- v2.0 默认必须是 `shallow_mlp`。

---

## 6. v2.0 严格 DSDM packet 设计

v2.0 使用 strict DSDM，不再使用单 anchor 的简化 DSDM。

每个 sender agent：

1. 在自己的 2 个 expert classes 上训练多个 DSDM guide checkpoints。
2. 蒸馏时每一步随机选择一个 guide checkpoint。
3. 使用 DSDM 的 prototype matching、covariance / semantic distribution matching、historical prototype smoothing。
4. 为自己的 expert classes 生成 IPC 张 distilled images。

默认建议：

```yaml
dsdm:
  strict: true
  guide_model_number: 10
  guide_pretrain_epochs: 20
  ipc: 10
  distill_steps: 10000
  cov_weight: 50.0
  h_p_weight: 0.2
  smooth_factor: 0.99
```

packet 内容：

```python
{
    "sender_id": int,
    "class_ids": List[int],
    "images": Tensor,
    "hard_labels": Tensor,
    "meta": Dict,
}
```

禁止在 v2.0 主线中保存：

```text
soft_targets
teacher_logits
teacher_probs
```

---

## 7. v2.0 工程阶段

### Stage 1: Expert Agent Training

每个 agent 只用自己的 2 个 expert classes 训练：

```text
expert real data -> agent model
```

模型：

```text
backbone + shallow_mlp_head
```

输出：

```text
outputs/{exp}/checkpoints/agents/agent_{id}_expert.pt
```

checkpoint 字段：

```python
{
    "agent_id": int,
    "expert_classes": List[int],
    "new_classes": List[int],
    "model_name": str,
    "model_state_dict": ...,
    "cfg": dict,
    "stage": "expert_train",
}
```

---

### Stage 2: DSDM Guide Checkpoint Pretraining

每个 agent 训练多个 guide checkpoints：

```text
outputs/{exp}/checkpoints/dsdm_guides/agent_{id}/guide_{k}.pt
```

入口建议：

```bash
python -m src.main.run_pretrain_dsdm_guides --config CONFIG --agent-ids all
```

---

### Stage 3: Packet Building

支持两类 packet：

```text
raw_ipc
strict_dsdm_ipc
```

raw baseline：

```text
每个 sender class 随机采 ipc 张真实图
```

strict DSDM：

```text
每个 sender class 蒸馏 ipc 张 synthetic images
```

输出：

```text
outputs/{exp}/packets/{packet_source}/agent_{id}_packet.pt
```

入口建议：

```bash
python -m src.main.run_build_packets_v2 --config CONFIG --packet-source raw
python -m src.main.run_build_packets_v2 --config CONFIG --packet-source dsdm
```

---

### Stage 4: Agent-to-Agent Social Learning

每个 receiver agent 使用：

```text
自己的 expert real data
+
其他 4 个 agents 的 packets
```

例如 agent_0：

```text
real classes: [0, 1]
packet classes: [2, 3, 4, 5, 6, 7, 8, 9]
```

训练策略分两阶段。

#### Phase A: Head Warm-up

```text
freeze backbone
train shallow_mlp_head
loss = CE
```

#### Phase B: Last-block Adaptation

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

入口建议：

```bash
python -m src.main.run_socialize_agents_v2 --config CONFIG --packet-source dsdm --agent-ids all
```

---

## 8. Baselines

v2.0 至少支持：

1. `local_only`
   - agent 只训练自己的 2 类。

2. `raw_ipc`
   - sender 每类随机采 IPC 张真实图。

3. `strict_dsdm_ipc`
   - sender 每类使用 strict DSDM 蒸馏 IPC 张图片。

要求：

- `raw_ipc` 和 `strict_dsdm_ipc` 使用相同 social training 策略。
- 这样才能公平比较 packet 质量。

---

## 9. Evaluation Metrics

每个 agent 输出：

```python
{
    "agent_id": int,
    "expert_classes": List[int],
    "new_classes": List[int],
    "expert_accuracy": float,
    "new_accuracy": float,
    "overall_accuracy": float,
}
```

summary 输出：

```python
{
    "average_expert_accuracy": float,
    "average_new_accuracy": float,
    "average_overall_accuracy": float,
}
```

通信指标：

```python
{
    "packet_source": str,
    "ipc": int,
    "bytes_per_packet": int,
    "total_transmitted_images": int,
    "total_comm_bytes": int,
    "compression_ratio_vs_full_raw": float,
}
```

---

## 10. Codex 工作风格要求

Codex 每次工作必须：

1. 先阅读 `AGENTS.md` 和 `PROJECT_SPEC.md`。
2. 确认当前任务属于 v2.0，不要恢复 v1 generalist 逻辑。
3. 小步实现。
4. 修改后运行最小 import / lint / smoke test。
5. 输出：
   - 修改了哪些文件；
   - 为什么这么改；
   - 如何运行；
   - 哪些旧逻辑被迁移到 legacy；
   - 下一步建议。
