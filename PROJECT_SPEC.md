# PROJECT_SPEC.md

## Project Definition

This project implements a conservative first version of **generalist-guided social packet learning**.

The system has specialist agents and one generalist teacher:

1. Each specialist initially knows 6 CIFAR10 classes and misses 4 classes.
2. The generalist teacher trains on all CIFAR10 classes.
3. The generalist builds class-wise `global_raw_packet` packets.
4. Each specialist receives only packets for its missing classes.
5. The specialist freezes its backbone and local head, then trains only a social head.
6. Evaluation reports known, missing, and general accuracy.
7. Compare reports include accuracy deltas and Stage 1 / Stage 2 communication estimates.

The first stable version intentionally does not implement DSDM packets, heterogeneous models, or peer packet exchange.

## Existing Framework Compatibility

The following old entry points and modules must remain usable:

- `src/main/run_local_pretrain.py`
- `src/main/run_build_packets.py`
- `src/main/run_social_train.py`
- `src/main/run_packet_only_train.py`
- `src/main/run_eval.py`
- `src/main/run_compare.py`
- `src/distill/simple_distiller.py`
- `src/models/model_pool.py`
- `src/models/agent_model.py`

New generalist-guided behavior is implemented through new entry points:

- `src/main/run_train_generalist.py`
- `src/main/run_train_specialists.py`
- `src/main/run_build_generalist_packets.py`
- `src/main/run_train_social_head.py`
- `src/main/run_eval_specialists.py`
- `src/main/run_compare_generalist.py`

## CIFAR10 Partial Split

The first supported split is:

```yaml
split:
  mode: partial_known
  name: cifar10_partial6
  num_agents: 5
  known_classes_per_agent: 6
  missing_classes_per_agent: 4
```

Fixed classes:

```text
agent_0 known [0,1,2,3,4,5], missing [6,7,8,9]
agent_1 known [2,3,4,5,6,7], missing [0,1,8,9]
agent_2 known [4,5,6,7,8,9], missing [0,1,2,3]
agent_3 known [0,1,6,7,8,9], missing [2,3,4,5]
agent_4 known [0,1,2,3,8,9], missing [4,5,6,7]
```

## Stage 1A: Specialist Local Training

Entry:

```bash
python -m src.main.run_train_specialists --config CONFIG --agent-ids all
```

Each specialist trains on its own known classes only.

Checkpoint:

```text
outputs/checkpoints/specialists/agent_{id}_specialist.pt
```

Required checkpoint fields:

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

## Stage 1B: Generalist Teacher Training

Entry:

```bash
python -m src.main.run_train_generalist --config CONFIG
```

The generalist trains on all CIFAR10 classes.

Checkpoint:

```text
outputs/checkpoints/generalist/generalist.pt
```

## Stage 2A: Generalist Raw Packet Building

Entry:

```bash
python -m src.main.run_build_generalist_packets --config CONFIG
```

Canonical packet source:

```yaml
packet:
  source: global_raw_packet
```

Backward-compatible alias:

```yaml
packet:
  source: global_raw
```

Both should resolve to the canonical output directory:

```text
outputs/packets/generalist/global_raw_packet/class_{c}_packet.pt
```

Each packet stores raw images, hard labels, generalist soft targets, and metadata.

## Stage 2B: Social Head Training

Entry:

```bash
python -m src.main.run_train_social_head --config CONFIG --agent-ids all
```

The receiver loads its specialist checkpoint, initializes a social head from the local head, freezes the backbone and local head, then updates only the social head.

Each training step is class-balanced:

```text
known real:     samples_per_class per known class
missing packet: samples_per_class per missing class
```

Packet sampling allows replacement.

Social-head loss:

```text
loss =
  lambda_known_ce  * CE(social_head(known_real), known_label)
+ lambda_packet_ce * CE(social_head(packet), packet_label)
+ lambda_packet_kd * KD(social_head(packet), packet_soft_target)
+ lambda_retain    * retain(social_head(known_real), frozen_local_head(known_real))
```

Checkpoint:

```text
outputs/checkpoints/social_head/agent_{id}_social_head.pt
```

## Evaluation

Entry:

```bash
python -m src.main.run_eval_specialists --config CONFIG --checkpoint-stage local
python -m src.main.run_eval_specialists --config CONFIG --checkpoint-stage social_head
```

Required metrics:

```text
known_accuracy
missing_accuracy
general_accuracy
```

## Compare

Entry:

```bash
python -m src.main.run_compare_generalist --config CONFIG
```

Required deltas:

```text
delta_known_accuracy
delta_missing_accuracy
delta_general_accuracy
```

Communication report:

- Stage 1: raw images used to train the generalist.
- Stage 2: missing-class packets consumed by specialists.

## Configs

Smoke test:

```text
configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml
```

Real experiment:

```text
configs/exp/2606-cifar10_partial6_global_raw_packet_real.yaml
```

The legacy filename below is kept as a smoke-compatible config:

```text
configs/exp/2606-cifar10_partial6_global_raw_packet.yaml
```
