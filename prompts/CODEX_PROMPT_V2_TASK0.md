# CODEX PROMPT — v2.0 Task 0

你现在要在本仓库中开始 v2.0 工程搭建。请先阅读并严格遵守：

```text
AGENTS.md
PROJECT_SPEC.md
```

当前 v2.0 主线是：

```text
DSDM-based Agent-to-Agent Socialized Learning
```

不要恢复旧版 generalist / social-head-only / soft-target packet 设计。

---

## 本次任务目标

本次只做 **Task 0: v2.0 cleanup and scaffold**。

不要一次性实现所有训练流程。先把仓库主线整理干净，并搭建 v2.0 最小骨架。

---

## 1. 清理旧版 v1 主线

请检查当前仓库，把不再作为 v2.0 主线的文件移动到 legacy 目录，或明确标记 deprecated。

重点对象包括但不限于：

```text
run_train_generalist.py
run_build_generalist_packets.py
run_train_social_head.py
SocialHeadAgent / social_head_model.py
generalist comparison scripts
partial-known split main configs
soft-target packet main dependency
packet-only generalist workflow
```

建议目录：

```text
legacy/v1_generalist/
legacy/v1_social_head/
legacy/v1_soft_target/
```

要求：

- 不要让 v2.0 主线入口继续 import legacy 代码。
- 如果某些旧文件仍被其他主线模块 import，先不要强行删除，改为迁移后修复 import 或保留 deprecated 注释。
- 清理后保证基础 import 不报错。

---

## 2. 新增 v2.0 配置目录

新增：

```text
configs/v2/cifar10_5agent_dsdm.yaml
configs/v2/cifar10_5agent_raw.yaml
```

配置至少包含：

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
  source: strict_dsdm   # raw 配置中改为 raw

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

## 3. 新增 v2.0 模型 head 骨架

新增或修改：

```text
src/models/heads.py
```

实现至少三个 head：

```text
linear
shallow_mlp
deep_mlp
```

其中 v2.0 默认使用：

```text
shallow_mlp = Linear(in_dim, 512) -> ReLU -> Dropout(0.1) -> Linear(512, num_classes)
```

请提供函数：

```python
def build_head(head_type: str, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
    ...
```

---

## 4. 新增 agent-wise model builder 骨架

新增或修改一个清晰接口，例如：

```python
def build_agent_model(cfg: dict, agent_id: int, device: torch.device):
    ...
```

要求：

- 从 `cfg["agent_models"][agent_id]` 读取 agent model type。
- 支持 `conv`、`resnet`、`resnet_ap`。
- 使用 v2.0 configurable head。
- 不要在本次任务中实现完整训练流程。

---

## 5. 新增 v2.0 入口脚本占位

新增以下脚本文件，只需完成 argparse、配置读取、清晰 TODO、基础 print，不要实现复杂训练：

```text
src/main/run_train_agents_v2.py
src/main/run_pretrain_dsdm_guides.py
src/main/run_build_packets_v2.py
src/main/run_socialize_agents_v2.py
src/main/run_eval_v2.py
src/main/run_compare_v2.py
```

每个脚本至少能：

```bash
python -m src.main.<script_name> --help
```

不报错。

---

## 6. 最小检查

完成后请运行尽可能轻量的检查：

```bash
python -m src.main.run_train_agents_v2 --help
python -m src.main.run_build_packets_v2 --help
python -m src.main.run_socialize_agents_v2 --help
python -m src.main.run_eval_v2 --help
python -m src.main.run_compare_v2 --help
```

如果环境允许，也运行：

```bash
python -m compileall src
```

---

## 7. 输出要求

完成后请在回复中说明：

1. 移动/删除了哪些 v1 文件。
2. 新增了哪些 v2 配置文件。
3. 新增了哪些模型 head / builder 接口。
4. 新增了哪些 v2 入口脚本。
5. 哪些检查已通过。
6. 下一步建议做什么。

本次任务不要实现 strict DSDM 蒸馏细节，也不要实现 social training 细节。那些属于后续 Task 1/2/3。
