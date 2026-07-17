# Baseline Run Summary

Last updated: 2026-06-30T11:30:12Z

## Current Status

The three requested comparison baselines have been run under the project CIFAR-100 socialized-learning task setting:

- Dataset: CIFAR-100
- Agents: 4
- Class split: 25 class-disjoint classes per agent
- Labels: global CIFAR-100 labels `0-99`
- Output dimension: 100 classes
- Budgets: IPC=10 and IPC=50
- Evaluation rows: agent `0-3` plus `avg`

Important interpretation note: these are adapted baselines for this task setting, not untouched official full-pipeline reproductions from the original repositories.

## Adapter Boundary

The implementation used a wrapper/adapter:

```text
adapters/baseline_runner.py
```

External repository source files under `DESA/`, `FedRE/`, and `SL/` were not modified.

Adaptation details:

- `DeSA-adapted`: preserves the synthetic-anchor communication idea. Sender expert models produce IPC-limited synthetic anchors; receiver models train on the shared anchor pool and are evaluated with socialized-learning metrics.
- `FedRE`: preserves representation-level communication. Sender representation/logit units are generated under the IPC limit; receiver training uses the representation-unit identities/logits and is evaluated with socialized-learning metrics.
- `MASC-adapted-Conv3`: uses homogeneous Conv3 for all agents and IPC-limited real collaboration units. It does not use full external CIFAR-100 data in collaboration.

This means the numbers are directly comparable under the local task protocol, but should be described as adapted baselines.

## Data And Environment

Data source:

```text
/root/autodl-pub/cifar-100/cifar-100-python.tar.gz
```

Local data links:

```text
data/cifar-100-python -> /root/autodl-tmp/DatasetDistillationforSocialLearning/data/cifar-100-python
data/cifar-100-python.tar.gz -> /root/autodl-pub/cifar-100/cifar-100-python.tar.gz
```

Python environment:

```text
conda env: sp
torch: 2.3.1+cu121
torchvision: 0.18.1+cu121
GPU: NVIDIA GeForce RTX 3090
```

Repository commits recorded during the run:

```text
DESA:  efccf9bc9817305c01c9e6f5c62b8b43d56f6a9a
FedRE: 9e2164343ee0d76be60afe46e2b986b9f77ef1cb
SL:    efcca8e9be2df1a4892ff569e2a4eea860bcb361
```

The adapter workspace itself is not a git repository, so no adapter commit hash is available.

## Commands

Compilation check:

```bash
conda run -n sp python -m py_compile adapters/baseline_runner.py
```

Final experiment command:

```bash
conda run -n sp python adapters/baseline_runner.py \
  --method all --ipc 10 50 \
  --expert-epochs 20 --social-epochs 20 --anchor-steps 40 \
  --batch-size 128 --data-dir ./data --output-root ./outputs
```

## Output Files

Required CSV outputs:

```text
outputs/baselines_cifar100_conv_family/desa/ipc10/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/desa/ipc50/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/fedre/ipc10/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/fedre/ipc50/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/masc_conv3/ipc10/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/masc_conv3/ipc50/metrics/social_results.csv
outputs/baselines_cifar100_conv_family/main_table_baselines.csv
outputs/baselines_cifar100_conv_family/reproduction_log.md
```

Each per-run CSV has 5 rows: four agents plus `avg`. The merged table has 30 rows.

## Average Results

| method | ipc | global_acc | expert_acc | new_acc | balanced_avg | directionality | forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeSA-adapted | 10 | 19.6450 | 41.8900 | 12.2300 | 27.0600 | 29.6600 | 23.8800 |
| DeSA-adapted | 50 | 33.2525 | 37.4300 | 31.8600 | 34.6450 | 5.5700 | 28.0700 |
| FedRE | 10 | 20.5375 | 37.6300 | 14.8400 | 26.2350 | 22.7900 | 27.8700 |
| FedRE | 50 | 33.4350 | 37.3300 | 32.1367 | 34.7333 | 5.1933 | 28.4700 |
| MASC-adapted-Conv3 | 10 | 20.4100 | 36.5600 | 15.0267 | 25.7933 | 21.5333 | 28.3100 |
| MASC-adapted-Conv3 | 50 | 33.7400 | 40.6400 | 31.4400 | 36.0400 | 9.2000 | 24.4100 |

## Validation Notes

Validated after final run:

- All six required `social_results.csv` files exist.
- Each `social_results.csv` has the required columns.
- Each `social_results.csv` has rows for agents `0`, `1`, `2`, `3`, and `avg`.
- `main_table_baselines.csv` exists and has 30 rows.
- Logs, configs, commits, checkpoints, and communication packets were saved under each method/IPC directory.

## Follow-Up Guidance

If stricter original-method reproduction is required later:

- DeSA should be revisited by integrating more of the original decentralized FL loop and documenting any REG/KD changes for class-disjoint clients.
- FedRE should be revisited by running the original server/global-classifier pipeline with a strict 4-client class-disjoint CIFAR-100 data adapter.
- MASC should be revisited by adapting its original generalist/student stage while preserving IPC-limited collaboration data.

Until then, use the method names and descriptions as adapted baselines, not untouched official baselines.
