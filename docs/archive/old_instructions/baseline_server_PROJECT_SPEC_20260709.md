# Codex CLI Project: Baseline Reproduction for CIFAR-100 Conv-Family Socialized Learning

## Purpose

This project file is for a separate AutoDL server dedicated to reproducing comparison baselines.

Do not overwrite the main project task files. Use this file as the working project brief for Codex CLI on the baseline server.

## Target Baselines

Reproduce and adapt exactly these three external comparison methods:

```text
1. DeSA
2. FedRE
3. MASC
```

These are the methods currently selected for the main comparison table.

## Main Task Setting

All adapted baselines must be evaluated under the same CIFAR-100 socialized-learning setting:

```text
Dataset: CIFAR-100
Number of agents: 4
Expert classes per agent: 25
Class split: class-disjoint
Output dimension: 100
Labels: keep global CIFAR-100 labels
Budgets: IPC=10 and IPC=50
```

Class split:

```python
agent_class_split = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}
```

Never remap labels to local 0-24 labels.

## Model Settings

### Heterogeneous Conv-Family Setting

Use this setting for:

```text
DeSA-adapted
FedRE
```

Model split:

```python
agent_model_split = {
    0: "convnet3w05",
    1: "convnet3w1",
    2: "convnet4w1",
    3: "convnet4w15",
}
```

If an external repository does not have these exact model names, implement equivalent ConvNet family variants:

```text
convnet3w05: 3-layer ConvNet, width 0.5
convnet3w1 : 3-layer ConvNet, width 1.0
convnet4w1 : 4-layer ConvNet, width 1.0
convnet4w15: 4-layer ConvNet, width 1.5
```

All variants must output 100 classes.

### Homogeneous Conv3 Setting for MASC

Use this setting for:

```text
MASC-adapted
```

MASC is the task-origin socialized-learning method, but it does not natively support heterogeneous architectures. Therefore:

```text
Use homogeneous Conv3 for all 4 agents.
```

If the external MASC repository calls this model `conv3`, use `conv3`.

If adapting inside the main project model pool, map Conv3 to:

```text
convnet3w1
```

Do not force MASC to use heterogeneous models in this task.

## External Repositories

Clone external code into:

```text
/root/autodl-tmp/baseline_code
```

Commands:

```bash
mkdir -p /root/autodl-tmp/baseline_code
cd /root/autodl-tmp/baseline_code

git clone https://github.com/ubc-tea/DESA.git
git clone https://github.com/AIResearch-Group/FedRE.git
git clone https://github.com/yxjdarren/SL.git
```

Keep the main project at:

```text
/root/autodl-tmp/DatasetDistillationforSocialLearning
```

Use wrappers/adapters where possible rather than heavily rewriting external repositories.

## Baseline Adaptation Details

### 1. DeSA-adapted

Source:

```text
https://github.com/ubc-tea/DESA
```

Role in paper:

```text
Nearest synthetic-anchor baseline.
```

Required adaptation:

```text
Dataset: CIFAR-100
Agents: 4
Split: 25 class-disjoint expert classes per agent
Models: heterogeneous conv-family
Budgets: IPC=10 and IPC=50 synthetic anchors per sender expert class
Output: per-agent socialized-learning metrics
```

Important notes:

```text
DeSA originally targets decentralized FL with synthetic anchors for data/model heterogeneity.
Our adaptation must evaluate whether synthetic-anchor communication works under class-disjoint socialized learning.
If DeSA's original REG loss assumes shared label space, modify it carefully and document the change.
Do not let DeSA use full real external data.
```

Expected method name in result CSV:

```text
DeSA-adapted
```

### 2. FedRE

Source:

```text
https://github.com/AIResearch-Group/FedRE
```

Role in paper:

```text
Recent model-heterogeneous FL baseline with representation-level communication.
```

Required adaptation:

```text
Dataset: CIFAR-100
Agents/clients: 4
Split: 25 class-disjoint expert classes per agent
Models: heterogeneous conv-family
Budgets: IPC=10 and IPC=50 representation units per sender expert class
Output: per-agent socialized-learning metrics
```

Important notes:

```text
FedRE is allowed to use its server/global-classifier design.
Do not remove the server just to make it decentralized.
The point of this baseline is to compare against a strong server-assisted model-heterogeneous FL method.
```

If FedRE's original CIFAR-100 pipeline uses a different partition:

```text
Add a class-disjoint 4-agent data adapter.
Do not change the target task.
```

Expected method name in result CSV:

```text
FedRE
```

### 3. MASC-adapted

Source:

```text
https://github.com/yxjdarren/SL
```

Role in paper:

```text
Task-origin socialized-learning baseline.
```

Required adaptation:

```text
Dataset: CIFAR-100
Agents: 4
Split: 25 class-disjoint expert classes per agent
Models: homogeneous Conv3
Budgets: IPC=10 and IPC=50 communicated samples per sender expert class
Output: per-agent socialized-learning metrics
```

Important notes:

```text
MASC is not a model-heterogeneous method.
Evaluate it under homogeneous Conv3, which is a favorable setting for MASC.
Restrict MASC collaboration/generalist training data to the same IPC communication budget.
Do not allow MASC to use full external real data in IPC-limited runs.
```

Expected method name in result CSV:

```text
MASC-adapted-Conv3
```

## Communication Budget Rules

Use IPC budgets:

```text
IPC=10
IPC=50
```

For each sender:

```text
num_sender_expert_classes = 25
communicated_units_per_sender = 25 * IPC
```

Therefore:

```text
IPC=10: 250 communicated units per sender
IPC=50: 1250 communicated units per sender
```

For each receiver, external communication from 3 other senders:

```text
IPC=10: 750 external communicated units
IPC=50: 3750 external communicated units
```

Define communication units by method:

```text
DeSA: synthetic anchor images
FedRE: entangled representation units
MASC: communicated training samples used by collaboration/generalist stage
```

If the method communicates non-image representations, record:

```text
comm_type = representation
external_comm_units = number of representation units
external_images = 0
```

If the method communicates images/anchors/samples, record:

```text
comm_type = image_or_anchor
external_comm_units = number of images or anchors
external_images = same as external_comm_units
```

## Required Metrics

Every method must output:

```text
Global
Expert
New
Balanced Avg
Directionality
Forgetting
Communication units
```

Definitions:

```text
Global = accuracy over all 100 CIFAR-100 test classes
Expert = accuracy over receiver's own 25 expert classes
New = accuracy over the other 75 classes
Balanced Avg = (Expert + New) / 2
Directionality = Expert - New
Forgetting = expert checkpoint Expert accuracy - final receiver Expert accuracy
```

If a baseline does not naturally produce a receiver checkpoint, create a compatible evaluation wrapper that evaluates the final per-agent model/head under these metrics.

## Required Result Schema

Each `social_results.csv` must contain these columns:

```text
method
ipc
agent_id
model_setting
agent_model
comm_type
comm_units_per_sender
external_comm_units
external_images
global_acc
expert_acc
new_acc
balanced_avg
directionality
forgetting
seed
run_name
source_repo
source_commit
notes
```

Also create one merged table:

```text
outputs/baselines_cifar100_conv_family/main_table_baselines.csv
```

This merged table must include all IPC=10 and IPC=50 rows for:

```text
DeSA-adapted
FedRE
MASC-adapted-Conv3
```

## Required Output Paths

Use these paths:

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

## Suggested Implementation Order

### Phase 1: Environment and Source Setup

```text
1. Clone DeSA, FedRE, and SL.
2. Record repository commit hashes.
3. Install only the minimal dependencies needed for each baseline.
4. Run each repository's shortest smoke test if available.
```

### Phase 2: Data Adapter

Create one shared CIFAR-100 4-agent adapter if possible:

```text
agent 0: classes 0-24
agent 1: classes 25-49
agent 2: classes 50-74
agent 3: classes 75-99
```

The adapter must preserve global labels.

### Phase 3: Metric Adapter

Create a shared evaluation utility that computes:

```text
Global
Expert
New
Balanced Avg
Directionality
Forgetting
```

Prefer reusing the main project's `social_metrics.py` if practical.

### Phase 4: Method Adaptation

Adapt in this order:

```text
1. MASC-adapted-Conv3
2. DeSA-adapted
3. FedRE
```

Reason:

```text
MASC validates the task-origin baseline first.
DeSA validates synthetic-anchor communication next.
FedRE validates representation-level model-heterogeneous FL last.
```

### Phase 5: Full Runs

Run:

```text
MASC-adapted-Conv3 IPC=10
MASC-adapted-Conv3 IPC=50
DeSA-adapted IPC=10
DeSA-adapted IPC=50
FedRE IPC=10
FedRE IPC=50
```

## Verification Requirements

Before long runs:

```bash
python -m py_compile <all new adapter files>
```

For each method, provide a dry-run or tiny smoke run proving:

```text
1. CIFAR-100 data loads.
2. 4-agent split is correct.
3. labels remain global 0-99.
4. model output dimension is 100.
5. IPC communication unit count is correct.
6. metrics CSV can be written.
```

## Final Report Requirements

When complete, produce:

```text
1. main_table_baselines.csv
2. reproduction_log.md
3. brief summary of each baseline adaptation
4. exact commands used
5. known deviations from original methods
6. blocking issues, if any
```

The final report must explicitly state:

```text
MASC is evaluated with homogeneous Conv3 because it does not support heterogeneous architectures.
FedRE is evaluated as a server-assisted model-heterogeneous FL baseline.
DeSA is evaluated as the nearest synthetic-anchor communication baseline.
All three are restricted to the same IPC=10/50 communication budget whenever applicable.
```

## Current Run Record

The latest completed adapted-baseline run is documented in:

```text
BASELINE_RUN_SUMMARY.md
```

This record includes the adapter boundary, command, environment, output paths, average metrics, and the note that the current numbers are adapted baselines rather than untouched official full-pipeline reproductions.
