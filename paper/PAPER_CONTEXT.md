# Paper Context Sync - 2026-07-01

## Project

Working title:

`Distilled Knowledge Packets for Communication-Efficient Heterogeneous Socialized Learning`

Chinese title:

`面向异构社会化学习的高知识密度蒸馏通信包构建方法`

Core claim under validation:

Input-space distilled knowledge packets can serve as architecture-compatible communication carriers in heterogeneous socialized learning. The main evidence should emphasize low communication budgets, especially IPC=10, where DSDM packets absorb new classes better than randomly selected real-image packets.

## Current Setting

- Dataset: CIFAR-100
- Agents: 4
- Split: class-disjoint, 25 expert classes per agent
- Labels: global CIFAR-100 labels, no remapping
- Main heterogeneous setting: ConvNet family
  - agent 0: ConvNet-3-w0.5
  - agent 1: ConvNet-3-w1.0
  - agent 2: ConvNet-4-w1.0
  - agent 3: ConvNet-4-w1.5

## Best Receiver Hyperparameters

Source: `outputs/experiment_registry/best_hyperparams.csv`

| IPC | Method | Receiver epochs | lambda_fr | lambda_kd | T | Global | New | Expert | Forgetting |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | DSDM_LOGIT | 60 | 0.20 | 0.60 | 2.0 | 34.18 | 31.05 | 43.57 | 23.57 |
| 50 | DSDM_LOGIT | 225 | 0.05 | 0.50 | 2.0 | 46.23 | 45.66 | 47.94 | 17.43 |

Important caveat: the IPC=50 hyperparameter result above is from the all-ConvNet setting where all agents use ConvNet-3-w1.0, not the strict ConvNet depth-width heterogeneous setting.

## IPC=10 ConvNet Family Results

Source: `outputs/experiment_registry/conv_family_ipc10_ipc50_paper_table.csv`

| Method | Role | Global | New | Expert | Forgetting | Note |
|---|---|---:|---:|---:|---:|---|
| HEURISTIC | recipe | 24.08 | 14.37 | 53.18 | 13.96 | random real packet, IPC=10 |
| DSDM | recipe | 30.47 | 26.40 | 42.68 | 24.46 | distilled image packet |
| DSDM_LOGIT | recipe | 32.87 | 29.69 | 42.40 | 24.74 | distilled packet + sender logits |
| DSDM_LOGIT | tuning | 34.18 | 31.05 | 43.57 | 23.57 | current best tuned receiver recipe |
| FULL_REAL | upper bound | 48.60 | 47.78 | 51.05 | 16.09 | full real social transfer |

Interpretation for writing:

- DSDM image packets substantially improve new-class absorption over random real packets at IPC=10.
- Sender logits further improve global and new-class accuracy.
- The method is still below full-real social transfer, so the correct claim is communication efficiency and high knowledge density, not matching the full-data upper bound.
- Forgetting remains a real limitation and should be discussed as receiver-side retention/future work.

## IPC=50 Status

The current strict ConvNet family IPC=50 DSDM experiment is still running.

As of 2026-07-01 09:59 CST:

- `agent_0` packet distillation completed.
  - best evaluator top-1: 71.8
  - packet: `outputs/cifar100_4agent_25cls_conv_family_ipc50/agents/agent_0/packets/dsdm_packet.pt`
- `agent_1` packet distillation is running.
  - progress: about 496/10000 iterations, 4.96%
  - current best evaluator top-1: 68.8
- `agent_2` and `agent_3` packet distillation have not completed yet.

Therefore, strict ConvNet family IPC=50 does not yet have final social-learning metrics. Existing IPC=50 receiver tuning results are useful for recipe selection, but should not be reported as strict ConvNet-family final results.

## Upper Bounds

Centralized full-data upper bounds already recorded:

- ConvNet-3-IN strict DSDM recipe: 65.24
- ResNet-10-BN CIFAR recipe: 72.08
- ResNetAP-10-BN CIFAR recipe: 73.47

ConvNet family centralized upper bounds from previous diagnostics:

- ConvNet-3-w0.5: 59.67
- ConvNet-3-w1.0: 65.17
- ConvNet-4-w1.0: 65.70
- ConvNet-4-w1.5: 68.16

These should be used to separate model capacity from packet/social-transfer performance.

## Writing Guidance

Recommended framing:

1. Problem: heterogeneous socialized learning cannot reliably exchange parameters, gradients, or features across architectures.
2. Proposed carrier: input-space knowledge packets, compatible with heterogeneous receivers.
3. Compression: dataset distillation increases per-image knowledge density under low communication budgets.
4. Main evidence: CIFAR-100, 4 agents, 25 class-disjoint classes each, IPC=10.
5. Key comparison: DSDM / DSDM_LOGIT vs random real packet and full-real social transfer.
6. Limitation: receiver forgetting, especially when new-class absorption improves.
7. Avoid overclaiming IPC=50: random real subsets are strong at higher communication budgets, and strict ConvNet-family IPC=50 is still running.

