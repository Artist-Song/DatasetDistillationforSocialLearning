# PROJECT_SPEC.md

最后更新：`2026-07-17`

本文件是项目研究定位、当前实验协议、最优超参数和结果解释的唯一事实来源。
运行级结果明细见 `RESULTS.md`，历史过程见 `EXPERIMENT_LOG.md`。

## 项目定位

项目名称：`DatasetDistillationforSocialLearning`

暂定题目：

```text
Distilled Knowledge Packets for Communication-Efficient Heterogeneous Socialized Learning
```

本文研究 class-disjoint agents 如何在架构异构和通信受限条件下，通过输入空间
knowledge packet 保持自身 expert knowledge 并学习其他 agent 的 non-expert classes。
通信对象是 distilled images 及其可选 soft annotations，不是模型参数、梯度或中间特征。

这不是模型聚合式联邦学习：当前任务没有中心服务器平均参数、同步全局模型或同构梯度聚合。

## 研究问题

```text
RQ1  输入空间 packet 能否作为异构模型的共享知识接口？
RQ2  DSDM 能否在相同 raw-image budget 下提高单位通信知识密度？
RQ3  Receiver 能否在吸收新类时控制 expert forgetting？
RQ4  Sender-specific packet quality 如何随 guide backbone 和通信预算变化？
```

核心指标：

```text
acc_global   : 100 类全局准确率
acc_new      : receiver 未拥有的 75 类准确率
acc_expert   : receiver 原有 25 类准确率
forgetting   : acc_expert_before - acc_expert_after
```

## 当前唯一主实验

### 数据与类别

```text
Dataset: CIFAR-100
Agents: 4
Class split: 4 × 25 class-disjoint expertise
Global labels: 0-99, never remapped
Classifier output: 100 for every model
Train samples per sender: 25 × 500 = 12,500
```

```python
agent_class_split = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}
```

### one-ResNet 架构分配

当前 no-VGG 设定中，每个 seed 恰好有一个 compact ResNet：

| Seed | Agent 0 / 0-24 | Agent 1 / 25-49 | Agent 2 / 50-74 | Agent 3 / 75-99 |
|---:|---|---|---|---|
| 0 | AlexNet | ConvNet-3-w1.0 | ResNet-18 compact | ConvNet-4-w1.5 |
| 1 | ConvNet-4-w1.5 | AlexNet | ResNet-10 compact | ConvNet-3-w1.0 |
| 2 | AlexNet | ResNet-18 compact | ConvNet-3-w1.0 | ConvNet-4-w1.5 |

配置 ID `resnet10/resnet18` 使用 CIFAR 3x3 stem、base width 32：

```text
ResNet-10 compact: 1,252,100 parameters
ResNet-18 compact: 2,820,740 parameters
```

它们不是约 11M 参数的 canonical ResNet-18。`resnet_cifar_standard` 仅用于诊断。

## 当前方法

当前主方法是 sender-specific、backbone-aligned self-distillation：

```text
Sender private expert data
→ locally trained guide pool using the sender task-backbone architecture
→ DSDM condensation
→ sender-specific distilled image packet
→ sender expert-class logits
→ heterogeneous receivers
```

每个 guide 只接触本 sender 的 25 类数据；不同 agent 不共享 guide 权重。统一 ConvNet
packet encoder 是重要对照方向，但尚未成为当前主方法。

### Logits

Logits 是 image packet 的增强注释：只保存 sender 25 个 expert classes 上的 logits，
不保存 full 100-class logits、参数、梯度或 feature maps。Ours 指完整 `DSDM + Logits`；
DSDM without logits 用于分离 synthetic image 和 soft annotation 的贡献。

## 通信协议

| IPC | Raw/sender | External raw/receiver | DSDM effective/receiver | Full Real 比例 |
|---:|---:|---:|---:|---:|
| 10 | 250 | 750 | 3,000 | 1/50 |
| 50 | 1,250 | 3,750 | 15,000 | 1/10 |
| Full Real | 12,500 | 37,500 | 37,500 | 1 |

DSDM `factor=2` 表示一张 raw image decode 为 4 张训练图；通信量始终按 raw image 统计。

## 最优主实验超参数

### DSDM 公共参数

| Parameter | Value |
|---|---:|
| niter | 10,000 |
| factor | 2 |
| init / decode | mix / single |
| augmentation | color_crop_cutout |
| match / metric | semantic / mse |
| batch_real / batch_syn_max | 256 / 256 |
| smooth_iter / smooth_factor | 2,000 / 0.99 |
| cov_weight / h_p_weight | 50 / 0.2 |
| guide pool | 10 models, 20 pretrain epochs |
| evaluation checkpoints | legacy fixed schedule for current three seeds |

### Guide-specific DSDM 参数

| Task backbone | f_idx / idx_from | lr_img |
|---|---:|---:|
| ConvNet-3 | 2 | 0.1 |
| ConvNet-4 | 3 | 0.1 |
| AlexNet | 7 | 0.005 |
| ResNet-10/18 | 5 | 0.01 |

### Receiver 参数

| IPC | Epochs | LR | lambda_fr | lambda_kd | T | Scheduler |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 60 | 0.01 | 0.20 | 0.60 | 2 | multistep 39/51, gamma 0.2 |
| 50 | 225 | 0.001 | 0.05 | 0.50 | 2 | none |

Receiver 从 expert checkpoint 初始化并训练完整 backbone；当前 `self_data_mode=packet`，
配置中的 `self_real_per_class=20` 不会产生 self-real replay（结果中 `self_real_images=0`）。

## Baseline 协议

| Method | 主要通信对象 | Soft information | 当前用途 |
|---|---|---|---|
| Expert Only | none | none | before-social baseline |
| Full Real | all real images | hard labels | communication oracle |
| Heuristic | random real images | hard labels | primary image baseline |
| FAST | selected real images | hard labels | coreset baseline |
| MASC-complete | real CC images + model interaction | teacher KD | homogeneous social-learning baseline |
| DeSA-CIL | synthetic anchors | iterative owner logits | adapted class-incremental baseline |

Heuristic 和 FAST 的监督/通信条件一致，可以直接比较 selection policy。MASC 使用同构
Netwider 且约有 452 MB 参数交互；DeSA-CIL 使用 owner-aware 类增量适配。二者必须带脚注。

## 当前正式结果

以下为 seed-level 4-agent mean 的 mean ± population std，单位为百分比。

### IPC=10（complete, 3 seeds）

| Method | Global | New | Expert | Forgetting |
|---|---:|---:|---:|---:|
| Heuristic hard | 22.35 ± 0.14 | 18.14 ± 0.33 | 34.98 ± 0.73 | 32.06 ± 0.49 |
| FAST | 20.30 ± 0.31 | 15.58 ± 0.16 | 34.44 ± 1.00 | 32.60 ± 1.01 |
| Ours DSDM + Logits | **32.78 ± 0.29** | **29.11 ± 0.35** | **43.81 ± 0.69** | **23.24 ± 0.60** |
| Full Real | 51.16 ± 0.32 | 50.40 ± 0.47 | 53.44 ± 0.42 | 13.60 ± 0.19 |

Ours 相对 Heuristic：global `+10.43`、new `+10.97`、expert `+8.83`、forgetting
降低 `8.82`。该差值支持完整方法，但不能全部归因于 synthetic images，需结合 w/o logits。

### IPC=50（Ours interim）

| Method | Seeds | Global | New | Expert | Forgetting |
|---|---:|---:|---:|---:|---:|
| Heuristic hard | 3 | 32.58 ± 0.50 | 29.24 ± 0.80 | 42.57 ± 0.87 | 24.47 ± 0.68 |
| FAST | 3 | 31.06 ± 0.36 | 27.18 ± 0.46 | 42.70 ± 0.58 | 24.34 ± 0.42 |
| Ours DSDM + Logits | 2/3 | 35.60 ± 0.56 | 33.48 ± 0.56 | 41.96 ± 0.59 | 25.14 ± 0.87 |

Ours IPC=50 在 seed1 完成前只能标记 interim，不能与三 seed baseline 写最终配对差值。

### 最新外部 baseline（single seed）

| Method | IPC | Global | New | Expert | Forgetting |
|---|---:|---:|---:|---:|---:|
| MASC-complete | 10 | 9.96 | 2.76 | 31.55 | 39.69 |
| MASC-complete | 50 | 22.81 | 14.79 | 46.89 | 24.35 |
| DeSA-CIL | 10 | 19.54 | 4.24 | 65.41 | 0.53 |
| DeSA-CIL | 50 | 15.02 | 0.96 | 57.20 | 8.74 |

MASC Full validation 为 `65.43 global / 65.03 new / 66.62 expert / 4.62 forgetting`；
one-ResNet Full Real 三 seed为 `51.16 / 50.40 / 53.44 / 13.60`。两者协议不同，不能横向排序。

## 重要中间证据

以下实验不是无效结果，应在 registry 中保留为 `historical_evidence` 或 `diagnostic`：

1. Conv-family IPC=10/50：证明 DSDM 低预算知识密度并形成 receiver 超参基础。
2. all-ConvNet IPC=50：高预算通信与 receiver recipe 诊断。
3. hetero4arch：暴露 VGG/AlexNet stability 问题和 feature-index 敏感性。
4. hetero6arch：ConvNet-family packet 跨架构复用证据。
5. Self-guided vs ConvNet-guided ResNet receiver 对照：说明 packet guide 与预算交互。
6. Centralized upper bounds：ConvNet-3 65.24、ResNet-10 72.08、ResNetAP-10 73.47。
7. ResNet compact/standard 与 feature-index 诊断：用于区分容量、recipe 和 packet 质量。

大量 guardian/receiver sweep 只保留按同协议选出的最优配置及汇总，其余进入删除候选清单。

## Tiny-ImageNet 计划（未实施）

Tiny-ImageNet 是下一数据集扩展，不属于当前完成结果。计划使用 200 类、4 agents × 50
class-disjoint expertise、全局 0-199 标签。Task backbone 首选标准 64x64 模型组合；必须先跑
centralized upper bound。Tiny 实验需要同时比较 self-guided 与 ConvNet local encoder，最终
以 packet quality、new-class absorption 和 expert stability 决定主消融结论。

## 结果选择与追溯

1. Registry 角色为 `main/baseline/ablation/diagnostic/historical_evidence/tuning/invalid`。
2. 同协议优先最高 global；差值小于 0.5 时优先 new 更高且 forgetting 更低者。
3. 每个正式数字必须记录 config、metrics、seed 数、packet guide 和 caveat。
4. `RESULTS.md` 和 `paper_tables/` 是生成产物；原始事实来自 CSV 和 packet provenance。
5. 活动实验状态写入 `docs/operations/current_status.md`，不写入 `AGENTS.md`。
