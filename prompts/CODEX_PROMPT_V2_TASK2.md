# CODEX PROMPT — v2.0 Task 2

你现在继续 v2.0 工程搭建。请先阅读：

```text
AGENTS.md
PROJECT_SPEC.md
prompts/CODEX_PROMPT_V2_TASK1.md
```

当前状态：

- `run_train_agents_v2.py` 已能训练 expert agents；
- `src/models/heads.py` 已支持 `linear / shallow_mlp / deep_mlp`；
- `build_agent_model(cfg, agent_id, device)` 已可按 agent_id 构建异构模型；
- `run_pretrain_dsdm_guides.py` 仍是占位脚本。

本次任务只做：

```text
Task 2: Strict DSDM Guide Checkpoint Pretraining
```

不要实现 packet building，不要实现 DSDM synthetic image optimization，不要实现 social learning。

---

## 0. 重要概念

v2.0 strict DSDM 要求每个 sender agent 拥有一个 guide checkpoint pool。

每个 sender agent：

```text
只使用自己的 2 个 expert classes
训练 M 个独立 guide models
后续 DSDM distillation 每个 iteration 随机选 guide checkpoint
```

注意：

- guide checkpoint 不是 receiver model；
- guide checkpoint 也不是把 expert agent checkpoint 复制 M 份；
- 默认应从不同随机种子独立训练，形成 guide model pool；
- guide model 架构默认使用 sender agent 自己的 model type，即：

```yaml
agent_models:
  0: conv
  1: resnet
  2: resnet_ap
  3: conv
  4: resnet
```

---

## 1. 实现 run_pretrain_dsdm_guides.py

修改：

```text
src/main/run_pretrain_dsdm_guides.py
```

要求支持命令：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids all
```

同时支持 smoke test：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids 0 \
  --max-guides 1 \
  --max-epochs 1 \
  --max-batches 1 \
  --no-download
```

需要新增 argparse 参数：

```text
--agent-ids
--max-guides
--max-epochs
--max-batches
--num-workers
--no-download
--dataset-root
--smoke-synthetic-samples
--skip-existing
```

参数语义：

- `--max-guides`: smoke test 时限制 guide 数量；正式实验不传。
- `--max-epochs`: smoke test 时限制 epoch；正式实验不传。
- `--max-batches`: smoke test 时限制每个 epoch batch 数；正式实验不传。
- `--skip-existing`: 如果 guide checkpoint 已存在，则跳过，方便断点续跑。

---

## 2. 可复用 Task 1 中的逻辑

优先复用已有函数：

```python
from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_runtime import resolve_device
```

也可以复用或迁移 `run_train_agents_v2.py` 中的：

```python
SyntheticCIFARDataset
get_new_classes
train_one_epoch
```

如果为了避免跨 main 脚本 import，可以把这些通用函数移动到：

```text
src/training/v2_train_utils.py
```

但不要过度重构。

---

## 3. 训练逻辑

对每个 selected agent：

1. 根据 direct split 得到 `expert_classes`。
2. 过滤 CIFAR train dataset，只保留该 agent 的 expert classes。
3. 读取：

```python
guide_model_number = cfg["dsdm"]["guide_model_number"]
guide_pretrain_epochs = cfg["dsdm"]["guide_pretrain_epochs"]
```

4. 对 `guide_id in range(guide_model_number)`：
   - 如果 `--max-guides` 存在，则限制 guide 数量；
   - 设置不同随机种子，例如：

```python
guide_seed = cfg["seed"] + agent_id * 1000 + guide_id
```

   - 构建一个新的 `build_agent_model(cfg, agent_id, device)`；
   - 使用该 agent 的 expert classes 子集训练；
   - 使用 SGD + CE：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg["train"].get("lr", 0.01),
    momentum=0.9,
    weight_decay=5e-4,
)
criterion = torch.nn.CrossEntropyLoss()
```

5. 每个 epoch 输出 loss / acc。
6. 保存 guide checkpoint。

注意：

- label 仍使用 CIFAR 原始标签 `[0..9]`。
- 不要 remap 成 `[0,1]`。
- 模型输出仍是 10-way logits。
- 每个 guide model 都应该是独立初始化训练，不要加载 agent expert checkpoint。

---

## 4. 保存路径与 checkpoint 格式

新增 path helper：

```python
def get_v2_dsdm_guide_dir(cfg: dict, agent_id: int) -> Path:
    return get_v2_experiment_root(cfg) / "checkpoints" / "dsdm_guides" / f"agent_{agent_id}"
```

保存路径：

```text
{output.root}/{experiment.name}/checkpoints/dsdm_guides/agent_{id}/guide_{guide_id}.pt
```

例如：

```text
outputs/v2/cifar10_5agent_v2_dsdm/checkpoints/dsdm_guides/agent_0/guide_0.pt
```

checkpoint 字段：

```python
{
    "agent_id": int,
    "guide_id": int,
    "expert_classes": List[int],
    "new_classes": List[int],
    "model_name": str,
    "model_state_dict": model.state_dict(),
    "cfg": cfg,
    "stage": "dsdm_guide_pretrain",
    "guide_seed": int,
    "train_summary": {
        "epochs": int,
        "final_loss": float,
        "final_acc": float,
    },
}
```

---

## 5. 正式实验命令

smoke test 通过后，可以开始真实 Stage 2 guide pretraining。

正式训练单个 agent：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids 0 \
  --no-download
```

正式训练全部 agents：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids all \
  --no-download
```

断点续跑：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids all \
  --no-download \
  --skip-existing
```

如果你发现 10 guides × 20 epochs 太慢，可以先只跑：

```bash
--max-guides 2 --max-epochs 5
```

但不要修改 config 默认值。

---

## 6. 最小检查

请运行：

```bash
python -m src.main.run_pretrain_dsdm_guides --help
```

再运行 smoke test：

```bash
python -m src.main.run_pretrain_dsdm_guides \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --agent-ids 0 \
  --max-guides 1 \
  --max-epochs 1 \
  --max-batches 1 \
  --no-download
```

确认生成：

```text
outputs/v2/cifar10_5agent_v2_dsdm/checkpoints/dsdm_guides/agent_0/guide_0.pt
```

如果环境允许，运行：

```bash
python -m compileall src
```

---

## 7. 输出要求

完成后请汇报：

1. 修改了哪些文件。
2. `run_pretrain_dsdm_guides.py` 如何运行。
3. guide checkpoint 保存在哪里。
4. smoke test 是否生成了 `guide_0.pt`。
5. 是否通过 `--help` / `compileall`。
6. 下一步建议。

本次任务完成后，下一步进入：

```text
Task 3: strict DSDM packet building
```

Task 3 才开始实现：

```text
load guide checkpoint pool
random guide selection per distillation iteration
prototype matching
covariance matching
historical prototype smoothing
save distilled packet images
```
