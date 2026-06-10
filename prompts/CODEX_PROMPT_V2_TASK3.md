# CODEX PROMPT — v2.0 Task 3

你现在继续 v2.0 工程搭建。请先阅读：

```text
AGENTS.md
PROJECT_SPEC.md
prompts/CODEX_PROMPT_V2_TASK1.md
prompts/CODEX_PROMPT_V2_TASK2.md
```

当前状态：

- `run_train_agents_v2.py` 已能训练 expert agents；
- `run_pretrain_dsdm_guides.py` 已能训练 strict DSDM guide checkpoints；
- `src/packet/packet_dataclass.py` 已是 v2 packet dataclass，只包含 `images + hard_labels + meta`，不包含 soft target；
- `run_build_packets_v2.py` 仍是占位脚本；
- legacy 中有旧版 single-anchor DSDM-style distiller，可参考但不要直接恢复为主线。

本次任务只做：

```text
Task 3: v2 Packet Building
```

包括：

1. raw IPC packet building；
2. strict DSDM IPC packet building；
3. 保存 v2 `SocialPacket`；
4. smoke test 可真实生成 packet 文件。

不要实现 social learning，不要实现 eval，不要引入 soft target。

---

## 0. 当前 v2 packet 约束

v2 packet 必须是 image-level hard-label packet。

允许保存：

```python
SocialPacket(
    sender_id=int,
    class_ids=List[int],
    images=torch.Tensor,
    hard_labels=torch.Tensor,
    meta=dict,
)
```

禁止保存：

```text
soft_targets
teacher_logits
teacher_probs
gradients
model parameters
```

---

## 1. 实现 run_build_packets_v2.py

修改：

```text
src/main/run_build_packets_v2.py
```

要求支持命令：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source raw \
  --agent-ids all \
  --no-download
```

以及：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids all \
  --no-download
```

需要新增 argparse 参数：

```text
--agent-ids
--packet-source        choices: raw, strict_dsdm
--max-steps           smoke test distillation steps cap
--max-agents          optional smoke test cap
--max-classes         optional smoke test cap per sender
--num-workers
--no-download
--dataset-root
--smoke-synthetic-samples
--skip-existing
```

其中：

- `--packet-source raw`: 不需要 guide checkpoints。
- `--packet-source strict_dsdm`: 必须读取对应 agent 的 guide checkpoints。
- `--max-steps`: 仅用于 smoke test，不修改 config 默认 `dsdm.distill_steps`。
- `--max-classes`: 例如 agent_0 expert classes `[0,1]` 时，`--max-classes 1` 只蒸馏 class 0，用于快速测试。
- `--skip-existing`: packet 已存在时跳过。

---

## 2. raw IPC packet building

raw packet 逻辑：

```text
对 sender agent 的每个 expert class，随机采 ipc 张真实训练图像。
```

要求：

- 使用 CIFAR 原始 label，不要 remap。
- 输出 images shape 应为：

```text
[len(class_ids) * ipc, 3, 32, 32]
```

- hard_labels shape：

```text
[len(class_ids) * ipc]
```

- 保存为 `SocialPacket`。

---

## 3. strict DSDM packet building

请新增主线 v2 distiller 文件，例如：

```text
src/distill/v2_strict_dsdm.py
```

可以参考 legacy 文件：

```text
legacy/v1_soft_target/src/distill/simple_distiller.py
```

但必须做以下关键修改：

### 3.1 不使用 single anchor model

旧版 distiller 是：

```python
distill_images_with_dsdm(anchor_model, train_dataset, class_ids, packet_cfg, device)
```

v2 strict DSDM 应改成 guide pool：

```python
distill_images_with_strict_dsdm(
    guide_models: List[nn.Module],
    train_dataset,
    class_ids: List[int],
    packet_cfg: dict,
    device: torch.device,
)
```

每个 distillation iteration 随机选择一个 guide model：

```python
guide_model = random.choice(guide_models)
```

然后用这个 guide model 做 feature prototype / covariance / historical prototype matching。

### 3.2 保留 DSDM 核心 losses

必须实现：

```text
prototype matching
covariance / semantic distribution matching
historical prototype smoothing
```

可以复用 legacy 中的逻辑：

```text
DiffAug
ClassDatasetSampler
Synthesizer
matchloss
get_feature_list
dist
```

但不要保留任何 soft-target 逻辑。

### 3.3 guide models 加载

新增辅助函数，建议放在 `run_build_packets_v2.py` 或 `src/distill/v2_strict_dsdm.py`：

```python
def load_guide_models(cfg, agent_id, device, max_guides=None) -> List[nn.Module]:
    ...
```

要求：

- 读取路径：

```text
outputs/v2/{experiment.name}/checkpoints/dsdm_guides/agent_{id}/guide_{k}.pt
```

- 使用 `build_agent_model(cfg, agent_id, device)` 构建模型；
- 加载 `model_state_dict`；
- `model.eval()`；
- `requires_grad_(False)`；
- 如果 guide checkpoint 不存在，报清晰错误，提示先运行 Task 2。

可以支持 `--max-guides` 参数，如果你认为有必要；否则 strict_dsdm 默认加载所有已存在 guide checkpoints。

### 3.4 feature extraction

使用 guide model 的 `get_backbone().get_feature(...)` 或现有兼容接口。

如果模型 feature API 不一致，请实现一个兼容函数：

```python
def get_feature_list(model, images, idx_from, idx_to):
    ...
```

目标是能支持：

```text
conv
resnet
resnet_ap
```

---

## 4. packet 保存路径

新增 path helper：

```python
def get_v2_packet_dir(cfg: dict, packet_source: str) -> Path:
    return get_v2_experiment_root(cfg) / "packets" / packet_source
```

保存路径：

```text
outputs/v2/{experiment.name}/packets/{packet_source}/agent_{id}_packet.pt
```

例如：

```text
outputs/v2/cifar10_5agent_v2_dsdm/packets/raw/agent_0_packet.pt
outputs/v2/cifar10_5agent_v2_dsdm/packets/strict_dsdm/agent_0_packet.pt
```

packet meta 至少包含：

```python
{
    "packet_source": "raw" or "strict_dsdm",
    "sender_id": int,
    "class_ids": List[int],
    "ipc": int,
    "bytes_images": int,
    "bytes_labels": int,
    "bytes_total": int,
}
```

strict DSDM packet 额外包含：

```python
{
    "distill_method": "strict_dsdm_guide_pool",
    "guide_count": int,
    "guide_checkpoint_paths": List[str],
    "distill_steps": int,
    "distill_final_loss": float,
    "distill_final_proto_loss": float,
    "distill_final_sem_loss": float,
    "distill_final_mem_loss": float,
}
```

---

## 5. config 与默认参数

从 config 读取：

```yaml
packet:
  ipc: 10
  source: strict_dsdm

dsdm:
  distill_steps: 10000
  cov_weight: 50.0
  h_p_weight: 0.2
  smooth_factor: 0.99
```

如果需要补充默认参数，可在代码中提供默认值：

```python
lr_img = cfg["dsdm"].get("distill_lr", 5e-3)
mom_img = cfg["dsdm"].get("mom_img", 0.5)
batch_real = cfg["dsdm"].get("batch_real", 256)
batch_syn_max = cfg["dsdm"].get("batch_syn_max", 256)
metric = cfg["dsdm"].get("metric", "l1")
idx_from = cfg["dsdm"].get("idx_from", 0)
idx_to = cfg["dsdm"].get("idx_to", -1)
aug_type = cfg["dsdm"].get("aug_type", "color_crop_cutout")
```

---

## 6. 最小检查与 smoke tests

先检查 help：

```bash
python -m src.main.run_build_packets_v2 --help
```

### 6.1 raw packet smoke test

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source raw \
  --agent-ids 0 \
  --max-classes 1 \
  --no-download
```

预期生成：

```text
outputs/v2/cifar10_5agent_v2_dsdm/packets/raw/agent_0_packet.pt
```

### 6.2 strict DSDM packet smoke test

前提：已经有：

```text
outputs/v2/cifar10_5agent_v2_dsdm/checkpoints/dsdm_guides/agent_0/guide_0.pt
```

运行：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids 0 \
  --max-classes 1 \
  --max-steps 2 \
  --no-download
```

预期生成：

```text
outputs/v2/cifar10_5agent_v2_dsdm/packets/strict_dsdm/agent_0_packet.pt
```

并且 packet 应满足：

```python
packet.sender_id == 0
packet.class_ids == [0]       # 因为 max_classes=1
packet.images.shape[0] == ipc
packet.hard_labels.unique() == tensor([0])
"soft_targets" not in packet.meta
```

如果环境允许，运行：

```bash
python -m compileall src
```

---

## 7. 正式实验命令

raw packets for all agents：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source raw \
  --agent-ids all \
  --no-download
```

strict DSDM packets for one agent：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids 0 \
  --no-download
```

strict DSDM packets for all agents：

```bash
python -m src.main.run_build_packets_v2 \
  --config configs/v2/cifar10_5agent_dsdm.yaml \
  --packet-source strict_dsdm \
  --agent-ids all \
  --no-download \
  --skip-existing
```

如果 10000 steps 太慢，先做小规模真实实验：

```bash
--max-steps 100
```

不要修改 config 默认值。

---

## 8. 输出要求

完成后请汇报：

1. 修改了哪些文件；
2. 新增的 v2 strict DSDM distiller 在哪里；
3. raw packet smoke test 是否通过；
4. strict DSDM packet smoke test 是否通过；
5. packet 保存在哪里；
6. packet 中是否确认没有 soft target；
7. 是否通过 `--help` / `compileall`；
8. 下一步建议。

本次任务完成后，下一步进入：

```text
Task 4: Agent-to-Agent Social Learning
```

Task 4 才实现：

```text
receiver loads own expert real data + other agents packets
Phase A: head warm-up
Phase B: last-block adaptation
expert/new/overall checkpoint saving
```
