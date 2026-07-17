# External Baselines

本目录统一管理论文对比方法的外部源码和适配记录。

## 管理原则

```text
1. external_baselines/repos/ 存放从 GitHub 拉取的第三方源码，但该目录不纳入本项目 git。
2. sources.yaml 记录每个 baseline 的 URL、commit、入口文件和适配口径。
3. 适配代码优先放在本项目自有 scripts / adapters 中，不直接大改第三方源码。
4. 跑实验时统一输出到本项目 outputs/ 下，指标统一为 acc_global / acc_new / acc_expert / forgetting。
5. 所有 baseline 都放到 CIFAR-100、4 agents、25 classes/agent 的 class-disjoint socialized setting 下评估。
```

## 当前已拉取 Baselines

```text
FAST     : CVPR 2026 DNN-free coreset selection
FedProto : Federated Prototype Learning across Heterogeneous Clients
MASC_SL  : Socialized Learning / MASC official code
DeSA     : Decentralized synthetic anchors
FedRE    : CVPR 2026 model-heterogeneous FL
```

## 当前比较口径

### FAST

FAST 作为真实图核心集选择对比，使用官方 commit
`6a218fcfdc93838634921399b0de6a36cdd29756`。适配口径：

```text
模式：pixels / per-class / minmax
每个 sender 对自己的 25 个 expert classes 独立选择
预算：IPC=10 或 IPC=50，严格每类等额
通信：被选中的原始真实图 + hard labels
训练：无 logits，factor=1，选中什么就直接用于 receiver 训练
```

官方首个 commit 的 graph-aware Hungarian matching 存在一处 NumPy 广播错误；
本项目只应用等价广播修复，补丁记录在：

```text
baseline_adapters/patches/fast_numpy_broadcast.patch
```

运行入口：

```bash
bash scripts/run_fast_one_resnet.sh
```

当前 one-ResNet 三 seed 结果：

```text
IPC=10: global=20.30±0.31, new=15.58±0.16, expert=34.44±1.00, forgetting=32.60±1.01
IPC=50: global=31.06±0.36, new=27.18±0.46, expert=42.70±0.58, forgetting=24.34±0.42
```

### MASC

MASC 作为原始 socialized learning 范式对比，不强行适配模型异构。实验口径：

```text
CIFAR-100 / 4 agents / 25 classes per agent
IPC=10 或 IPC=50：每个 agent 只向 MASC generalist 提供 25*IPC 张真实图
评价：acc_global, acc_new, acc_expert, forgetting
```

核心目的：

```text
在同一 socialized learning 任务和同一 IPC 图像预算下，比较 MASC 原始范式和本文 distilled packet 的知识传递效率。
```

完整复现入口为 `baseline_adapters/run_masc_complete.py`。该入口使用官方
`Netwider(13)` expert 和 `Netwider_multi(13)` student，补齐论文中的 CC、RA、
HFE/energy routing 三阶段。正式队列先运行 full-data sanity gate；只有 expert、
global 和 new-class 三项均通过最低合理阈值后，才继续 IPC=10/50。

### DeSA

DeSA 作为最接近的 synthetic-anchor decentralized heterogeneous FL baseline。实验口径：

```text
CIFAR-100 / 4 agents / 25 classes per agent / hetero6arch
IPC=10 或 IPC=50：每个 agent 生成 25*IPC 张 synthetic anchors
如交换 anchor logits，单独统计 logit bytes
评价：acc_global, acc_new, acc_expert, forgetting
```

类别增量适配入口为 `baseline_adapters/run_desa_cil.py`，正式名称记作
`DeSA-CIL`。它保留官方 random-ConvNet distribution matching、迭代 KD 和
SupCon regularization；针对 class-disjoint expertise 只做必要修改：

```text
1. 四个 sender 的 anchor 按全局类标签拼接，不再对同一 slot 求平均；
2. 每张 anchor 的 soft target 只来自拥有该类的 owner；
3. receiver 从已有 expert checkpoint 初始化，分类头始终为全局 100 类；
4. 每个 seed 严格使用 one-ResNet 主实验的 AlexNet/Conv3/Conv4/ResNet10/18。
```

### FedProto

FedProto 作为经典 prototype-based model/statistical heterogeneous FL baseline。实验口径：

```text
faithful adaptation first：先按 FedProto 完整流程适配到本任务，不先限制 IPC
通信对象：class prototypes
需记录：rounds, prototype dimension, total prototype bytes
评价：acc_global, acc_new, acc_expert, forgetting
```

### FedRE

FedRE 作为近两年 model-heterogeneous FL baseline。实验口径：

```text
faithful adaptation first：先按 FedRE 完整流程适配到本任务，不先限制 IPC
通信对象：entangled representations / label encodings / global classifier 相关信息
需记录：rounds, transmitted tensors, total bytes
评价：acc_global, acc_new, acc_expert, forgetting
```

## 后续适配优先级

```text
1. MASC-IPC10 / MASC-IPC50：先跑原始 socialized learning 范式的低通信版本。
2. DeSA-IPC10 / DeSA-IPC50：复用 synthetic anchor 生成逻辑，适配到 class-disjoint CIFAR-100。
3. FedProto faithful adaptation：优先得到经典 prototype baseline。
4. FedRE faithful adaptation：优先跑通官方 Cifar100/model-hetero 流程，再替换为本项目 split。
```

## 统一 Split 导出

新增本项目自有适配层：

```text
baseline_adapters/export_cifar100_splits.py
```

用途：

```text
1. 从项目 yaml 读取固定 CIFAR-100 / 4 agents / 25 classes split。
2. 为 MASC 导出 ImageFolder 目录，训练集按 IPC=10/50 限制。
3. 为 FedRE 导出 per-client npz，保留每个 agent 的 25 个 expert classes。
4. 为 FedProto / DeSA 导出 split_metadata.json，后续 runner 直接读取，避免各 repo 重新随机采样。
```

示例命令：

```bash
/root/miniconda3/envs/sp/bin/python -m baseline_adapters.export_cifar100_splits \
  --config configs/main_cifar100_hetero6arch_ipc10_seed0.yaml \
  --ipc 10 \
  --target all

/root/miniconda3/envs/sp/bin/python -m baseline_adapters.export_cifar100_splits \
  --config configs/main_cifar100_hetero6arch_ipc50_seed0.yaml \
  --ipc 50 \
  --target all
```

导出位置：

```text
external_baselines/outputs/splits/<config_name>/ipc<IPC>/
```

当前适配状态：

```text
MASC    : 数据格式已适配；提供 eval template；需要 checkpoint.pth 后才能运行官方 inference。
FedRE   : 数据格式已适配；提供 run_fedre_social.sh；已验证官方 read_client_data 可读取 4-agent split。
FedProto: 提供 run_fedproto_social.py / run_fedproto_social.sh；
          已替换官方随机 user_groups/classes_list 为固定 social split。
DeSA    : 提供 run_desa_social.py / run_desa_social.sh；
          使用 DeSA 官方网络/增强工具，严格为每个 sender 的 25 expert classes 生成 25*IPC anchors。
```

## 运行入口

### FedRE

```bash
ROUNDS=100 DEVICE_ID=0 \
  bash scripts/run_fedre_social.sh \
  external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10
```

### FedProto

```bash
ROUNDS=100 GPU=0 \
  bash scripts/run_fedproto_social.sh \
  external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10
```

### DeSA

```bash
IPC=10 GPU=0 \
  bash scripts/run_desa_social.sh \
  external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10
```

当前 one-ResNet 正式队列统一使用：

```bash
bash scripts/run_masc_desa_complete_queue.sh
```

### MASC

旧的 inference template 仍要求外部 checkpoint；正式复现不再依赖该 template，
而是由 `run_masc_complete.py` 从头训练论文网络。旧 template 仅保留源码核查用途：

```bash
export MASC_STUDENT_CKPT_DIR=/path/to/student_checkpoint_dir
export MASC_TEACHER_CKPT_DIR=/path/to/teacher_checkpoint_dir
TASK_ID=0 AGENT_ID=0 GPU=0 \
  bash scripts/run_masc_eval_template.sh \
  external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10
```

注意：

```text
MASC 用作原始 socialized learning reference，不声称模型异构。
FedProto/FedRE 的通信对象不是图像 IPC，因此正式报告时必须单独统计 rounds 和 tensor bytes。
DeSA-social runner 按本文公平 IPC 口径限制为 25*IPC anchors/sender。
```
