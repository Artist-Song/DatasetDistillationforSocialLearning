# CODEX PROMPT — v2.0 Task 4: Make the Full Experiment Runnable

请先阅读：

```text
AGENTS.md
PROJECT_SPEC.md
prompts/CODEX_PROMPT_V2_TASK1.md
prompts/CODEX_PROMPT_V2_TASK2.md
prompts/CODEX_PROMPT_V2_TASK3.md
```

当前状态：

- Stage 1 expert agent training 已可运行；
- Stage 2 DSDM guide pretraining 已可运行；
- Stage 3 raw / strict_dsdm packet building 已可运行；
- `run_socialize_agents_v2.py`、`run_eval_v2.py`、`run_compare_v2.py` 仍是占位。

本次目标：

```text
把工程推进到“可以开始整体实验”的状态。
```

也就是实现：

1. agent-to-agent social learning；
2. expert / socialized checkpoint evaluation；
3. local_only / raw_ipc / strict_dsdm_ipc comparison；
4. 可选的 run-all 脚本或命令文档。

不要修改 strict DSDM packet 的核心蒸馏逻辑，除非是修 bug。

---

## 1. 实现 run_socialize_agents_v2.py

修改：

```text
src/main/run_socialize_agents_v2.py
```

目标：receiver agent 加载自己的 expert checkpoint，并使用：

```text
自己的 expert real data
+
其他 agents 的 packet images
```

进行两阶段训练。

命令：

```bash
python -m src.main.run_socialize_agents_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source raw \
  --agent-ids all \
  --no-download
```

以及：

```bash
python -m src.main.run_socialize_agents_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids all \
  --no-download
```

支持参数：

```text
--agent-ids
--packet-source raw|strict_dsdm
--max-epochs-a
--max-epochs-b
--max-batches
--num-workers
--no-download
--dataset-root
--skip-existing
```

### 1.1 数据构造

对 receiver agent_i：

```text
real data: agent_i expert classes
packet data: all sender agents except i
```

例如：

```text
agent_0 real classes: [0,1]
agent_0 packet classes: [2,3,4,5,6,7,8,9]
```

注意：

- 不加载自己的 packet；
- packet labels 保持 CIFAR 原始 label；
- 模型输出仍是 10-way logits。

### 1.2 Class-balanced batch

不要简单 concat 后 shuffle。

需要实现 class-balanced sampler / dataset batch builder。

每个 step：

```text
每个 class 采 samples_per_class 张
expert classes 从本地真实数据采样
new classes 从 packet data 采样
packet 数据允许 replacement sampling
```

从 config 读取：

```yaml
social:
  samples_per_class: 8
  steps_per_epoch: 100
```

### 1.3 Phase A: head warm-up

```text
freeze backbone feature extractor
train shallow MLP head only
loss = CE
```

实现要求：

- 对 `AgentModel.backbone` 找到 classifier head：`classifier` / `fc` / `head`；
- 先 freeze all parameters；
- 再 unfreeze head parameters；
- optimizer 只包含 requires_grad=True 的参数；
- lr 使用 `social.lr_head`。

### 1.4 Phase B: last-block adaptation

```text
freeze early backbone
unfreeze last block + head
loss = CE + lambda_anchor * ||theta_trainable - theta_old||^2
```

last block 定义：

```text
conv: backbone.layers['conv'][-1] plus its norm if exists
resnet: backbone.layer4
resnet_ap: backbone.layer4
```

实现建议：

- 新增工具函数：

```python
freeze_all(model)
unfreeze_head(model)
unfreeze_last_block(model)
get_trainable_named_params(model)
make_anchor_snapshot(model)
anchor_regularization(model, anchor_snapshot)
```

- `theta_old` 应该是 Phase B 开始前的 trainable parameter snapshot；
- 只对 trainable params 做 anchor regularization；
- `lambda_anchor` 从 `social.lambda_anchor` 读取。

### 1.5 checkpoint 保存

保存路径：

```text
outputs/v2/{experiment}/checkpoints/socialized/{packet_source}/agent_{id}_socialized.pt
```

checkpoint 字段：

```python
{
  "agent_id": int,
  "expert_classes": List[int],
  "new_classes": List[int],
  "model_name": str,
  "model_state_dict": state_dict,
  "packet_source": str,
  "used_sender_ids": List[int],
  "stage": "socialized",
  "train_summary": {...},
  "cfg": cfg,
}
```

---

## 2. 实现 run_eval_v2.py

修改：

```text
src/main/run_eval_v2.py
```

支持：

```bash
python -m src.main.run_eval_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --checkpoint-stage expert \
  --agent-ids all \
  --no-download
```

以及：

```bash
python -m src.main.run_eval_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --checkpoint-stage socialized \
  --packet-source strict_dsdm \
  --agent-ids all \
  --no-download
```

支持参数：

```text
--agent-ids
--checkpoint-stage expert|socialized
--packet-source raw|strict_dsdm
--batch-size
--num-workers
--no-download
--dataset-root
```

### 2.1 指标

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

解释：

```text
expert_accuracy: 只在 receiver expert classes 测试样本上算
new_accuracy: 只在其他 8 类测试样本上算
overall_accuracy: CIFAR-10 全测试集算
```

保存：

```text
outputs/v2/{experiment}/metrics/{checkpoint_stage}_{packet_source_or_none}.json
```

summary：

```python
average_expert_accuracy
average_new_accuracy
average_overall_accuracy
```

---

## 3. 实现 run_compare_v2.py

修改：

```text
src/main/run_compare_v2.py
```

读取 eval 结果并生成 comparison summary。

比较方法：

```text
local_only: expert checkpoints eval
raw_ipc: socialized raw packets eval
strict_dsdm_ipc: socialized strict_dsdm packets eval
```

输出路径：

```text
outputs/v2/{experiment}/metrics/comparison_v2.json
outputs/v2/{experiment}/metrics/comparison_v2.csv
```

comparison 字段：

```python
method
average_expert_accuracy
average_new_accuracy
average_overall_accuracy
total_comm_bytes
total_transmitted_images
ipc
```

通信量统计：

- local_only: 0；
- raw_ipc / strict_dsdm_ipc: sum all packet.meta['bytes_total'] for transmitted packets；
- 注意 agent-to-agent 中每个 sender packet 会被其他 4 个 agents 接收，统计 total communication 时要乘以 receiver count 或在 meta 中明确记录。

推荐同时保存：

```python
unique_packet_bytes
agent_to_agent_total_bytes
```

---

## 4. 新增实验运行文档或 run-all 脚本

新增其一或两者都做：

```text
scripts/run_v2_smoke.sh
scripts/run_v2_full.sh
```

或者：

```text
docs/RUN_V2_EXPERIMENT.md
```

至少写清楚完整实验顺序：

```bash
# 1 expert agents
python -m src.main.run_train_agents_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --agent-ids all --no-download

# 2 guide checkpoints
python -m src.main.run_pretrain_dsdm_guides --config configs/v2/cifar10_5agent_dsdm.yaml --agent-ids all --no-download --skip-existing

# 3 raw packets
python -m src.main.run_build_packets_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --packet-source raw --agent-ids all --no-download --skip-existing

# 4 strict DSDM packets
python -m src.main.run_build_packets_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --packet-source strict_dsdm --agent-ids all --no-download --skip-existing

# 5 social learning raw
python -m src.main.run_socialize_agents_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --packet-source raw --agent-ids all --no-download --skip-existing

# 6 social learning strict DSDM
python -m src.main.run_socialize_agents_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --packet-source strict_dsdm --agent-ids all --no-download --skip-existing

# 7 eval expert
python -m src.main.run_eval_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --checkpoint-stage expert --agent-ids all --no-download

# 8 eval raw socialized
python -m src.main.run_eval_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --checkpoint-stage socialized --packet-source raw --agent-ids all --no-download

# 9 eval strict DSDM socialized
python -m src.main.run_eval_v2 --config configs/v2/cifar10_5agent_dsdm.yaml --checkpoint-stage socialized --packet-source strict_dsdm --agent-ids all --no-download

# 10 compare
python -m src.main.run_compare_v2 --config configs/v2/cifar10_5agent_dsdm.yaml
```

---

## 5. Smoke test commands

必须能跑以下 smoke test：

```bash
python -m src.main.run_socialize_agents_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids 0 \
  --max-epochs-a 1 \
  --max-epochs-b 1 \
  --max-batches 1 \
  --no-download
```

然后：

```bash
python -m src.main.run_eval_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --checkpoint-stage socialized \
  --packet-source strict_dsdm \
  --agent-ids 0 \
  --no-download
```

最后：

```bash
python -m src.main.run_compare_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml
```

如果 eval/compare 因 raw 或 all-agent results 缺失而不能完整比较，要输出清晰 warning，不要直接崩溃。

---

## 6. 输出要求

完成后请汇报：

1. 修改了哪些文件；
2. social learning 是否实现 Phase A + Phase B；
3. eval 是否输出 expert/new/overall；
4. compare 是否输出 local/raw/strict_dsdm summary；
5. smoke test 是否通过；
6. 完整实验命令在哪里；
7. 当前还需要用户先补跑哪些长期训练。

本次完成后，仓库应进入“可以开始整体实验”的状态。
