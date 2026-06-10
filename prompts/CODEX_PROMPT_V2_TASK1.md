# CODEX PROMPT — v2.0 Task 1

你现在继续 v2.0 工程搭建。请先阅读：

```text
AGENTS.md
PROJECT_SPEC.md
prompts/CODEX_PROMPT_V2_TASK0.md
```

当前已完成 Task 0 的 scaffold：

- v2 configs 已新增；
- `src/models/heads.py` 已新增；
- `build_agent_model(cfg, agent_id, device)` 已加入；
- v2 main scripts 已有占位；
- 但 `run_train_agents_v2.py` 还没有实现真正训练。

本次任务只做 **Task 1: Expert Agent Training**。

不要实现 DSDM guide pretraining，不要实现 packet building，不要实现 social learning。

---

## 目标

实现 v2.0 Stage 1：

```text
每个 agent 只用自己的 2 个 expert classes 训练完整分类模型。
```

固定 direct split：

```text
agent_0: [0, 1]
agent_1: [2, 3]
agent_2: [4, 5]
agent_3: [6, 7]
agent_4: [8, 9]
```

每个 agent 使用 config 中的异构模型：

```yaml
agent_models:
  0: conv
  1: resnet
  2: resnet_ap
  3: conv
  4: resnet
```

每个模型输出仍然是 10-way logits。

---

## 1. 实现 run_train_agents_v2.py

修改：

```text
src/main/run_train_agents_v2.py
```

要求实现：

```bash
python -m src.main.run_train_agents_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --agent-ids all
```

支持：

```bash
--agent-ids all
--agent-ids 0
--agent-ids 0,2,4
--agent-ids 0-3
```

可复用现有：

```python
src.utils.agent_selection.parse_agent_ids
src.datasets.cifar.build_cifar_train_dataset
src.datasets.cifar.make_direct_class_splits
src.datasets.cifar.subset_by_classes
src.models.agent_model.build_agent_model
src.main.run_local_pretrain.resolve_device
src.utils.config.load_yaml
src.utils.seed.set_seed
```

如果某个工具函数不存在或签名不同，请优先复用已有实现，不要重复造太多轮子。

---

## 2. 训练逻辑

对每个 selected agent：

1. 根据 direct split 得到 `expert_classes`。
2. `new_classes = all_classes - expert_classes`。
3. 从 CIFAR train dataset 过滤出 expert_classes 子集。
4. 使用 `build_agent_model(cfg, agent_id, device)` 构建异构模型。
5. 使用 SGD 训练：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg["train"]["lr"],
    momentum=0.9,
    weight_decay=5e-4,
)
criterion = torch.nn.CrossEntropyLoss()
```

6. 每个 epoch 输出 loss / acc。
7. 保存 checkpoint。

注意：

- label 仍然使用 CIFAR 原始标签 `[0..9]`。
- 不要把 label remap 到 `[0,1]`。
- 因为模型输出是 10-way logits。

---

## 3. 保存路径与 checkpoint 格式

保存路径：

```text
{output.root}/{experiment.name}/checkpoints/agents/agent_{id}_expert.pt
```

例如：

```text
outputs/v2/cifar10_5agent_v2_dsdm/checkpoints/agents/agent_0_expert.pt
```

checkpoint 字段：

```python
{
    "agent_id": int,
    "expert_classes": List[int],
    "new_classes": List[int],
    "model_name": str,
    "model_state_dict": model.state_dict(),
    "cfg": cfg,
    "stage": "expert_train",
}
```

请保存必要训练摘要，例如：

```python
"train_summary": {
    "epochs": int,
    "final_loss": float,
    "final_acc": float,
}
```

---

## 4. 增加轻量工具函数

如有需要，可以新增：

```text
src/utils/v2_paths.py
src/utils/v2_splits.py
```

但不要过度抽象。

推荐至少有一个清晰函数生成 experiment root：

```python
def get_v2_experiment_root(cfg):
    return Path(cfg["output"]["root"]) / cfg["experiment"]["name"]
```

---

## 5. 修复 Task 0 遗留问题

Task 0 没有真正迁移 legacy 文件。不要在本次大规模清理，但请至少做到：

1. 不让 v2 入口 import v1 generalist / social head 代码。
2. 如果发现明显的 v1 文件仍在主线被 v2 引用，请断开引用。
3. 不要删除旧文件，除非确认没有任何 import 依赖。

真正的 legacy 大清理可以留到单独 Task。

---

## 6. 最小检查

请运行：

```bash
python -m src.main.run_train_agents_v2 --help
python -m src.main.run_train_agents_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --agent-ids 0
```

如果实际训练太慢，可以临时在本地测试时把 config epoch 改小或加 `--max-epochs 1` 参数。

建议为脚本添加可选参数：

```bash
--max-epochs 1
```

用于 smoke test，不改变 config 文件默认值。

如果环境允许，运行：

```bash
python -m compileall src
```

---

## 7. 输出要求

完成后请汇报：

1. 修改了哪些文件。
2. `run_train_agents_v2.py` 如何运行。
3. checkpoint 保存在哪里。
4. 是否成功训练了 agent_0 smoke test。
5. 是否通过 `--help` / `compileall`。
6. 下一步建议。

本次任务完成后，下一步才进入：

```text
Task 2: strict DSDM guide checkpoint pretraining
```
