# AGENTS.md

## 0. 给 Codex 的最高优先级原则

本项目当前目标是在现有工程框架上增量实现新的 **generalist-guided distilled social packet learning** 目标。

非常重要：

- 不要大规模重构现有项目。
- 不要删除已有的 `run_local_pretrain.py`、`run_build_packets.py`、`run_social_train.py`、`run_packet_only_train.py`、`run_eval.py`、`run_compare.py` 等旧入口。
- 不要把已有 direct split / packet-only / mixed social train 逻辑直接改坏。
- 优先通过**新增接口、新增配置、新增脚本、新增可选参数**实现新功能。
- 旧代码能跑的实验必须继续能跑；新实验使用新配置和新入口。
- 如果必须修改已有文件，只做兼容性扩展，不改变默认行为。
- 每次提交只解决一个清晰子目标，避免一次性大改。

一句话：**保留旧框架，新增 generalist-guided pipeline。**

---

## 1. 项目新目标

实现一个基于 PyTorch 的两阶段社会化学习框架：

1. 每个 specialist agent 初始只掌握部分类别，例如 CIFAR10 中 known 6 类、missing 4 类。
2. 第一阶段通过 raw image sharing 构建一个掌握所有类别的 generalist teacher agent。
3. 第二阶段由 generalist teacher 通过数据集蒸馏生成低通信量 distilled social packets。
4. specialist agents 只接收自己 missing classes 对应的 packets。
5. specialist 在第二阶段冻结 backbone 和 local head，仅训练新增 social head。
6. 通过这种 image-level packet 机制，在 teacher 和 receiver backbone 不同的情况下仍然实现异构通信。

研究问题：

- distilled packet 能否教会 specialist 原本不会的 missing classes？
- 相比 raw image sharing，Stage 2 通信量是否显著降低？
- teacher 与 receiver 模型异构时，image-level packet 是否仍然有效？
- 统一 generalist teacher 生成 packet 是否缓解多个 specialist 分别蒸馏带来的 packet inconsistency？

---

## 2. 理论与论文背景对齐

本项目借鉴 MASC 的两阶段思想：

- MASC 先通过 collective collaboration 让 student/generalist 获得全类能力；
- 再通过 reciprocal altruism 让原 expert agents 学习 general classes，同时保持 expert classes。

本工程中的对应关系：

- specialist agents ≈ MASC teacher agents；
- generalist teacher ≈ MASC student/generalist；
- Stage 1 raw image sharing 构建 generalist ≈ collective collaboration 的工程化稳定实现；
- Stage 2 distilled packet 回传 ≈ reciprocal altruism 的低通信实现。

同时，本项目借鉴 CD2/FSCIL 思想：

- specialist 已有 known/base classes；
- missing classes 类似 incremental / novel classes；
- 第二阶段冻结 backbone，只训练 classifier/social head；
- distilled packets 类似 compressed memory / distilled memory。

本项目借鉴 DSDM 思想：

- distilled images 通过 feature prototype matching、covariance/semantic diversity matching 生成；
- 当前代码中的 DSDM-style distiller 可先复用；
- 后续再补充更严格的 cross-architecture distillation。

---

## 3. 当前主线任务设置

### 3.1 数据集优先级

第一阶段只做：

```text
CIFAR10 known 6 / missing 4
```

后续再做：

```text
CIFAR100 known 80 / missing 20
CIFAR100 known 60 / missing 40
```

### 3.2 CIFAR10 默认 split

新增 partial known/missing split，不替换旧 direct split。

默认：

```yaml
split:
  mode: partial_known
  strategy: fixed_cifar10_5x6
  num_specialists: 5
  known_classes_per_agent: 6
```

推荐固定划分：

```text
agent_0 known: [0, 1, 2, 3, 4, 5], missing: [6, 7, 8, 9]
agent_1 known: [2, 3, 4, 5, 6, 7], missing: [0, 1, 8, 9]
agent_2 known: [4, 5, 6, 7, 8, 9], missing: [0, 1, 2, 3]
agent_3 known: [0, 1, 6, 7, 8, 9], missing: [2, 3, 4, 5]
agent_4 known: [0, 1, 2, 3, 8, 9], missing: [4, 5, 6, 7]
```

每个 agent split 必须保存：

```python
{
    "agent_id": int,
    "known_classes": List[int],
    "missing_classes": List[int],
    "all_classes": List[int],
}
```

---

## 4. 两阶段工程流程

### Stage 1A: specialist local pretrain

每个 specialist 只使用自己的 known classes 训练：

```text
known data -> backbone + local_head
```

输出：

```text
outputs/.../checkpoints/specialist_local/{run_name}/agent_{id}_local.pt
```

要求：

- 不要复用 missing classes 数据；
- checkpoint 中保存 known/missing classes；
- 旧 `run_local_pretrain.py` 默认行为不变；
- 可新增 `run_train_specialists.py` 或给旧脚本加兼容 `split.mode=partial_known`。

### Stage 1B: generalist teacher training

generalist teacher 使用所有类别数据训练：

```text
all raw data -> generalist teacher
```

输出：

```text
outputs/.../checkpoints/generalist/{run_name}/generalist_teacher.pt
```

要求：

- 新增入口 `src/main/run_train_generalist.py`；
- 不要影响 specialist local training；
- 统计 Stage 1 raw image sharing 通信量 `stage1_comm_bytes`；
- teacher 可先用 `conv` 跑通，再换 `resnet`。

### Stage 2A: generalist packet building

由 generalist teacher 生成 class-wise packets。

支持两种 packet source：

1. `global_raw_packet`: 每类采 IPC 张真实图像；
2. `global_dsdm_packet`: 每类生成 IPC 张 DSDM distilled images。

第一版优先实现 `global_raw_packet`，跑通全流程后再接 `global_dsdm_packet`。

建议保存：

```text
outputs/.../packets/generalist/{packet_source}/class_{c}_packet.pt
```

### Stage 2B: specialist social head training

receiver 只加载自己的 missing classes packets。

例如 agent_0 只加载：

```text
class_6_packet.pt
class_7_packet.pt
class_8_packet.pt
class_9_packet.pt
```

训练规则：

- 加载 specialist local checkpoint；
- 冻结 backbone；
- 冻结 local_head；
- 新增并初始化 social_head；
- 只训练 social_head；
- 使用 class-balanced batch；
- 保存 social checkpoint。

输出：

```text
outputs/.../checkpoints/social_head/{run_name}/agent_{id}_social_head.pt
```

---

## 5. 模型设计原则

当前项目已有 `AgentModel`、`model_pool`、`ConvNet`、`ResNet` 等结构。

不要推翻它们。优先新增兼容接口：

```python
forward_features(x)
forward_head(features, head="local" | "social")
forward(x, head="local" | "social")
init_social_head_from_local()
freeze_backbone()
freeze_local_head()
train_social_head_only()
```

第一版可以只支持 `conv`，后续再支持 `resnet` / `resnet_ap`。

### 5.1 local head / social head

local pretrain 阶段：

```text
train backbone + local_head
```

social train 阶段：

```text
freeze backbone
freeze local_head
initialize social_head from local_head
train social_head only
```

默认评估 social 后模型时使用：

```text
head="social"
```

默认评估 local 时使用：

```text
head="local"
```

### 5.2 不要第一版就实现复杂 HFE routing

MASC 中存在 HFE classifier selection，但本项目第一版不实现复杂 HFE 路由。

先做：

```text
local stage -> local_head
social stage -> social_head
```

等主线跑通后，再考虑 HFE / classifier selection。

---

## 6. Stage 2 训练 batch 与 loss

### 6.1 Class-balanced batch

不要把全部 known real data 和少量 packet data 直接混合 shuffle。

正确做法：

```text
for each step:
    对每个 known class，从本地真实数据中采 k 张；
    对每个 missing class，从 packet 中采 k 张；
    packet 允许 replacement sampling。
```

例如 CIFAR10 agent_0：

```text
known 6 类，每类 8 张 -> 48 张
missing 4 类，每类 8 张 -> 32 张
batch 共 80 张
```

### 6.2 Loss

默认：

```text
L = lambda_packet_ce * L_packet_ce
  + lambda_packet_kd * L_packet_kd
  + lambda_known_ce * L_known_ce
  + lambda_retain * L_retain
```

含义：

- `L_packet_ce`: social_head 在 missing packet images 上的 hard-label CE；
- `L_packet_kd`: social_head 在 packet images 上模仿 generalist teacher soft targets；
- `L_known_ce`: social_head 在 known real images 上的 hard-label CE；
- `L_retain`: social_head 在 known real images 上模仿 frozen local_head 输出。

推荐默认：

```yaml
lambda_packet_ce: 1.0
lambda_packet_kd: 0.1
lambda_known_ce: 0.2
lambda_retain: 0.05
```

不要默认用 full model fine-tune。full model fine-tune 只作为 ablation。

---

## 7. Baselines

第一版必须优先支持：

1. `local_only`
   - specialist 只训练 known classes，不接收 packet。

2. `global_raw_packet`
   - generalist 从 raw data 中每类采 IPC 张；
   - 先用它跑通全流程。

3. `global_dsdm_packet`
   - generalist teacher 生成 DSDM distilled packets；
   - 主方法。

后续再支持：

4. `global_raw_missing_share`
   - 发送 missing classes 全量 raw data；
   - 高通信上界。

5. `peer_dsdm_packet`
   - specialist 各自蒸馏并互传；
   - 用于比较 generalist teacher 是否缓解 packet inconsistency。

6. `full_finetune_receiver`
   - Stage 2 更新整个 receiver model；
   - ablation。

---

## 8. 评价指标

每个 specialist receiver 必须报告：

```text
known_accuracy
missing_accuracy
general_accuracy
```

并报告相对 local baseline 的 delta：

```text
delta_known_accuracy
delta_missing_accuracy
delta_general_accuracy
```

通信指标：

```text
bytes_per_packet
stage1_comm_bytes
stage2_comm_bytes
total_comm_bytes
accuracy_per_mb_stage2
accuracy_per_mb_total
```

异构指标：

```text
teacher_model
receiver_model
is_heterogeneous
heterogeneity_gap_missing
heterogeneity_gap_general
```

packet 诊断指标：

```text
teacher_packet_accuracy
receiver_anchor_packet_accuracy
receiver_post_packet_accuracy
packet_readability_gap
```

---

## 9. 推荐新增文件

优先新增，不要覆盖旧文件。

### Dataset

```text
src/datasets/splits.py
```

功能：

```python
make_partial_known_splits(...)
save_split_plan(...)
load_split_plan(...)
```

### Main scripts

```text
src/main/run_train_generalist.py
src/main/run_train_specialists.py
src/main/run_build_generalist_packets.py
src/main/run_train_social_head.py
src/main/run_eval_specialists.py
src/main/run_compare_generalist.py
src/main/run_packet_diagnostics.py
```

### Config

```text
configs/exp/2606-cifar10_partial6_global_raw_packet.yaml
configs/exp/2606-cifar10_partial6_global_dsdm_packet.yaml
```

### Script

```text
scripts/run_cifar10_partial6_generalist_packet.sh
```

---

## 10. 推荐实现顺序

严格按顺序做，避免一次性大改：

1. 新增 partial-known split，不影响旧 direct split。
2. 新增 `run_train_generalist.py`。
3. 新增 `run_train_specialists.py` 或兼容扩展旧 local pretrain。
4. 新增 `global_raw_packet` 构建。
5. 新增 social_head 模型接口，第一版只支持 conv。
6. 新增 `run_train_social_head.py`，使用 class-balanced batch。
7. 新增 known/missing/general eval。
8. 新增 compare summary。
9. 接入 DSDM packet。
10. 增加 teacher=resnet, receiver=conv 异构设置。
11. 增加 communication metrics。
12. 增加 packet diagnostics。
13. 最后再实现 peer_dsdm_packet 对照。

---

## 11. 最小验收命令链

第一版先跑 raw packet：

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

验收标准：

- local known_accuracy 明显高于 random；
- local missing_accuracy 低；
- social missing_accuracy 明显高于 local missing_accuracy；
- social general_accuracy 高于 local general_accuracy；
- known_accuracy 不应严重崩塌；
- report 能保存为 JSON；
- compare 能输出 delta metrics。

---

## 12. 编码风格要求

- 所有新增脚本必须有 `argparse --config`。
- 所有 checkpoint/report/packet 路径必须由 config 和 run_name 生成，不要硬编码。
- 所有新增功能必须支持 CPU fallback。
- 所有 tensor 保存使用 `torch.save`。
- 所有 report 使用 JSON，summary 可额外输出 CSV/Markdown。
- 所有新增函数尽量小而清楚，不要把全部逻辑塞进 main。
- 保留中文日志输出可以，但关键字段名用英文，方便 JSON 后处理。

---

## 13. 每次完成后必须说明

每次 Codex 完成一个阶段，需要输出：

```text
改了哪些文件
新增了哪些入口
是否影响旧入口
当前能跑到哪一步
下一条测试命令是什么
还缺什么
```
