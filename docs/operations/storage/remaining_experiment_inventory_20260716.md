# Remaining experiment inventory, 2026-07-16

Physical-size attribution after packet deduplication (hard-linked content is counted only once
by the parent `du` traversal, so per-family values are approximate):

| Family | Approx. size | Status |
|---|---:|---|
| one-ResNet main + FAST | 7.56 GB | Current; keep |
| hetero4arch / VGG history | 7.18 GB | Historical; archive candidate |
| hetero6arch history | 5.86 GB | Historical; archive candidate |
| all-ConvNet IPC=50 history | 1.71 GB | Historical; archive candidate |
| ConvNet-family history | 1.63 GB | Historical; retain paper metrics, artifacts optional |
| no-VGG assembly history | 1.04 GB | Source artifacts reused by current runs; keep for now |
| other tuning / ablation | 1.02 GB | Historical; archive candidate |
| MASC smoke runs | 0.44 GB | Smoke-only; first deletion candidate if explicitly approved |

Largest individual historical runs:

| Run | Physical size |
|---|---:|
| `cifar100_4agent_25cls_hetero4arch_ipc50` | 1.69 GB |
| `cifar100_4agent_25cls_hetero4arch_ipc10` | 1.54 GB |
| `cifar100_4agent_25cls_ipc50_allconvnet` | 1.08 GB |
| `cifar100_4agent_25cls_hetero6arch_ipc10_seed1` | 1.01 GB |

No historical experiment directory was deleted. Before any future archive/delete pass, export
`metrics/`, `config/`, manifests, and the experiment registry, then verify that the current
one-ResNet queue does not reference the candidate as an artifact source.
