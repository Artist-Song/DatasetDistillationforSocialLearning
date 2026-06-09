# AGENTS.md

## Current Objective

Build the first stable closed loop for **generalist-guided social packet learning** on the existing codebase.

Do not redesign the framework. Do not delete old functionality. Do not break the old entry points:

- `src/main/run_local_pretrain.py`
- `src/main/run_build_packets.py`
- `src/main/run_social_train.py`
- `src/main/run_packet_only_train.py`
- `src/main/run_eval.py`
- `src/main/run_compare.py`
- `src/distill/simple_distiller.py`
- `src/models/model_pool.py`
- `src/models/agent_model.py`

New behavior should be added through new scripts, new configs, and compatible wrappers.

## First Stable Version

Only implement and stabilize:

- CIFAR10 partial known split.
- Generalist teacher training on all classes.
- Specialist local training on known classes only.
- `global_raw_packet` class-wise packets from the generalist.
- Specialist social-head-only training on missing class packets.
- Known, missing, and general evaluation.
- Compare report with accuracy deltas and Stage 1 / Stage 2 communication estimates.

Do not add DSDM packets yet. Do not add heterogeneous models yet. Do not add peer packets yet.

## Packet Source Naming

Use:

```yaml
packet:
  source: global_raw_packet
```

Keep `global_raw` only as a backward-compatible alias. Internally, new outputs should use:

```text
outputs/packets/generalist/global_raw_packet/
```

## Fixed CIFAR10 Split

```text
agent_0 known [0,1,2,3,4,5], missing [6,7,8,9]
agent_1 known [2,3,4,5,6,7], missing [0,1,8,9]
agent_2 known [4,5,6,7,8,9], missing [0,1,2,3]
agent_3 known [0,1,6,7,8,9], missing [2,3,4,5]
agent_4 known [0,1,2,3,8,9], missing [4,5,6,7]
```

Specialist checkpoints must save:

```python
{
    "agent_id": int,
    "known_classes": list,
    "missing_classes": list,
    "stage": "specialist_local",
    "model_state_dict": ...,
    "cfg": dict,
}
```

## Social Head Contract

Prefer a wrapper around the current `AgentModel` and backbone. The wrapper should support:

```python
forward(x, head="local")
forward(x, head="social")
init_social_head_from_local()
freeze_backbone()
freeze_local_head()
train_social_head_only()
```

Social-head training must freeze the backbone and local head. Only `social_head` parameters should update.

## Balanced Step Sampling

Do not build a static mixed dataset of all known real samples plus a few packet samples.

Each social-head training step must sample:

- From known real data: `samples_per_class` images per known class.
- From missing packets: `samples_per_class` images per missing class.

Packet sampling may use replacement.

## Social Head Loss

Use separate weights:

```yaml
social_head:
  lambda_packet_ce: 1.0
  lambda_packet_kd: 1.0
  lambda_known_ce: 1.0
  lambda_retain: 1.0
```

Loss terms:

- `lambda_known_ce`: CE on known real images.
- `lambda_packet_ce`: CE on missing packet images.
- `lambda_packet_kd`: KD from packet soft targets on missing packet images.
- `lambda_retain`: retain loss on known real images, using the frozen local head as teacher.

## Configs

Keep smoke and real experiments separate:

- Smoke: `configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml`
- Real: `configs/exp/2606-cifar10_partial6_global_raw_packet_real.yaml`

The smoke config uses 1 epoch. The real config uses:

- generalist epochs: 50
- specialist epochs: 50
- social_head epochs: 30

## Expected Commands

```bash
python -m src.main.run_train_generalist --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml
python -m src.main.run_train_specialists --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml --agent-ids all
python -m src.main.run_build_generalist_packets --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml
python -m src.main.run_train_social_head --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml --agent-ids all
python -m src.main.run_eval_specialists --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml --checkpoint-stage local
python -m src.main.run_eval_specialists --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml --checkpoint-stage social_head
python -m src.main.run_compare_generalist --config configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml
```

## Git Rule

After each completed implementation step, commit locally. Stop before `git push` and tell the user to run `git push` manually.
