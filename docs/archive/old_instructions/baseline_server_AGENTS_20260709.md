# Baseline Reproduction Agent

## Role

You are the Codex CLI agent responsible for reproducing and adapting external baseline methods for the project:

```text
DatasetDistillationforSocialLearning
```

Your task is not to redesign the proposed method. Your task is to make three comparison baselines run under the project's task setting and output metrics in the same format as the main project.

The three baselines are:

```text
1. DeSA
2. FedRE
3. MASC
```

## Primary Goal

Reproduce these methods under the following task setting:

```text
Dataset: CIFAR-100
Agents: 4
Class split: 25 classes per agent, class-disjoint
Labels: keep global CIFAR-100 labels, never remap to 0-24 local labels
Communication budgets: IPC=10 and IPC=50
Evaluation: per-agent and average socialized-learning metrics
```

The final outputs must be directly comparable with the main project results.

## Hard Constraints

1. Do not overwrite the original project files unless the task explicitly requires an adapter.
2. Prefer creating wrappers, adapters, config files, and result converters around external repositories.
3. Do not change the main DSDM method implementation.
4. Do not change the proposed method's existing result files.
5. Do not use full external CIFAR-100 real data for communication-limited baselines.
6. For all IPC-limited baselines, communication must be restricted to 25 classes x IPC units per sender.
7. All models must output 100 classes.
8. All labels must remain CIFAR-100 global labels.
9. Every run must save command logs, config snapshots, code commit/hash information when available, and metrics CSV.
10. Never delete existing outputs. Write new results under a baseline-specific output directory.

## Workspace Layout

Use this layout on the AutoDL server:

```text
/root/autodl-tmp/DatasetDistillationforSocialLearning
/root/autodl-tmp/baseline_code/SL
/root/autodl-tmp/baseline_code/DESA
/root/autodl-tmp/baseline_code/FedRE
/root/autodl-tmp/baseline_code/adapters
```

Clone external repositories only under:

```text
/root/autodl-tmp/baseline_code
```

Suggested clone commands:

```bash
mkdir -p /root/autodl-tmp/baseline_code
cd /root/autodl-tmp/baseline_code

git clone https://github.com/yxjdarren/SL.git
git clone https://github.com/ubc-tea/DESA.git
git clone https://github.com/AIResearch-Group/FedRE.git
```

## Project Setting

Use the following split everywhere:

```python
agent_class_split = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}
```

For heterogeneous conv-family baselines, use:

```python
agent_model_split = {
    0: "convnet3w05",
    1: "convnet3w1",
    2: "convnet4w1",
    3: "convnet4w15",
}
```

For MASC, use homogeneous Conv3 because MASC is the task-origin method and does not natively support model heterogeneity:

```python
agent_model_split = {
    0: "conv3",
    1: "conv3",
    2: "conv3",
    3: "conv3",
}
```

If the local project uses `convnet3w1` instead of `conv3`, map MASC Conv3 to:

```text
convnet3w1
```

Do not compare MASC under heterogeneous architectures unless explicitly requested later.

## Baseline Responsibilities

### DeSA

Repository:

```text
https://github.com/ubc-tea/DESA
```

Adapt DeSA to CIFAR-100 4-agent class-disjoint socialized learning.

Required adaptation:

```text
1. Use CIFAR-100 global labels.
2. Use 4 clients/agents with 25 expert classes each.
3. Use conv-family heterogeneous models.
4. Restrict communicated synthetic anchors to IPC=10 or IPC=50 per sender expert class.
5. Ensure the final evaluation produces per-agent Global, Expert, New, Balanced Avg, Directionality, and Forgetting.
```

If DeSA's original REG loss assumes shared classes across clients, adapt carefully:

```text
Do not force same-class local-anchor alignment when the receiver has no local samples of that class.
Document any modification to REG, KD, or anchor construction.
```

### FedRE

Repository:

```text
https://github.com/AIResearch-Group/FedRE
```

FedRE is allowed to use its server/global-classifier design. Do not remove its server just to match decentralization.

Required adaptation:

```text
1. Use CIFAR-100.
2. Use 4 clients/agents with 25 expert classes each.
3. Use conv-family heterogeneous models where possible.
4. Restrict representation communication to IPC=10 or IPC=50 representation units per sender expert class.
5. Preserve FedRE's representation-level communication idea.
6. Output the same socialized-learning metrics as the main project.
```

If original FedRE expects standard FL partitions rather than strict class-disjoint 25-class agents, implement a data adapter rather than changing the task setting.

### MASC

Repository:

```text
https://github.com/yxjdarren/SL
```

MASC is included because this project follows the socialized-learning task formulation. MASC is not model-heterogeneous, so adapt it under homogeneous Conv3 only.

Required adaptation:

```text
1. Use CIFAR-100.
2. Use 4 agents.
3. Use 25 expert classes per agent.
4. Use homogeneous Conv3 for all agents.
5. Restrict communication/training data used in MASC collaboration to IPC=10 or IPC=50 per sender expert class.
6. Do not let MASC use full external real data in IPC-limited runs.
7. Output the same metrics as the main project.
```

If original MASC uses full training data for its generalist/student stage, create an IPC-limited adaptation and document it as:

```text
MASC-adapted (Homo Conv3, IPC-limited)
```

## Required Metrics

Every baseline must output per-agent rows and an average row with:

```text
method
ipc
agent_id
model_setting
agent_model
comm_type
comm_units_per_sender
external_comm_units
global_acc
expert_acc
new_acc
balanced_avg
directionality
forgetting
seed
run_name
notes
```

Definitions:

```text
Global: accuracy over all 100 CIFAR-100 test classes
Expert: accuracy over the receiver's own 25 expert classes
New: accuracy over the other 75 classes
Balanced Avg: (Expert + New) / 2
Directionality: Expert - New
Forgetting: expert checkpoint Expert accuracy - final receiver Expert accuracy
```

Use the main project's metric definitions whenever possible.

## Required Output Files

Write baseline outputs under:

```text
outputs/baselines_cifar100_conv_family/
```

Required files:

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

## Validation

Before launching long training, run dry-runs or small smoke tests.

Required checks:

```bash
python -m py_compile <new_or_modified_python_files>
```

If adapting an external repository, also save:

```text
baseline repository commit hash
modified files list
exact run command
environment information
```

## Reporting

When reporting progress, use this structure:

```text
1. Method adapted
2. Source repository and commit
3. Task setting confirmed
4. Communication budget confirmed
5. Command used
6. Output path
7. Current result or blocking error
```

用中文回复
如果环境不支持，需要自动下载（比如pytorch），autoDL服务器内置了conda。

If a baseline cannot be faithfully adapted, do not silently replace it with another method. Report the exact incompatibility and propose the smallest adaptation.

