# PROJECT_SPEC.md

最后更新：`2026-07-19`

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

### IPC=50（complete, Ours 2 seeds）

| Method | Seeds | Global | New | Expert | Forgetting |
|---|---:|---:|---:|---:|---:|
| Heuristic hard | 3 | 32.58 ± 0.50 | 29.24 ± 0.80 | 42.57 ± 0.87 | 24.47 ± 0.68 |
| FAST | 3 | 31.06 ± 0.36 | 27.18 ± 0.46 | 42.70 ± 0.58 | 24.34 ± 0.42 |
| Ours DSDM + Logits | 2 | 35.60 ± 0.56 | 33.48 ± 0.56 | 41.96 ± 0.59 | 25.14 ± 0.87 |

Ours IPC=50 固定使用 seed0 和 seed2 作为最终两种子结果；seed1 经用户决定停止。
Heuristic 和 FAST 保持各自原有的三种子聚合，不因 Ours 的 seed1 缺失而改写。

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

## PAT-style 类别划分扩展（seed0 验证运行中）

该扩展只借用 PAT 的 class-disjoint 类别分配思想，不复用 FedRE 的模型、优化器、服务器或
训练/测试重划分。CIFAR-100 官方 50,000 张训练集和 10,000 张测试集保持不变；seed0 用
`numpy.random.RandomState(0).permutation(100)` 固定类别顺序，再比较：

```text
PAT5 : 5 agents x 20 classes，五种 backbone 各出现一次
PAT10: 10 agents x 10 classes，五种 backbone 各出现两次
Models: ConvNet-3 / ConvNet-4 / AlexNet / compact ResNet-10 / compact ResNet-18
```

PAT5 的每个 20 类集合等于 PAT10 中相邻两个 10 类集合的并集。除 agent 数量与类别集合外，
两组均沿用当前 IPC=10 主方法：sender task-backbone-guided DSDM、sender expert-class logits、
receiver recipe 和 global/new/expert/forgetting 指标。每个 receiver 的 global 在原始 CIFAR-100
全类测试集上计算；PAT5 的 expert/new 测试样本数为 2,000/8,000，PAT10 为 1,000/9,000。
两组属于不同 `comparability_group`，不能与 4-agent 主表直接计算方法差值。

PAT seed0 使用稀疏 DSDM 合成集验证点 `100/500/1000/2000/3000/5000/7500/10000`，蒸馏
优化仍完整运行 10,000 iterations。最佳图片只在这8个固定候选中选择，并通过 immutable
history 保留；该协议与旧 one-ResNet 每100轮验证一次的100候选协议不得混合解释。

PAT10 首次稀疏运行中，AlexNet agent 2 在约 iteration 4,056 出现数值爆炸；此前保存的
iteration 100 packet 为有限值且本地评估 `64.9`，但该未完整蒸馏的 packet 只作 diagnostic，
不得进入正式通信结果。恢复运行保持 AlexNet 历史最优 `lr_img=0.005`，仅为 PAT10 AlexNet
启用 `grad_clip_norm=100` 截断异常梯度尖峰，并记录实际裁剪次数和最大梯度范数。agent 2
完整跑满 10,000 iterations 后的最佳评估不得低于 `64.9`；未通过该门槛时不运行 agent 7，
也不进入 logits、communication 或 receiver 阶段。PAT5 按用户要求继续原运行，不受该修改影响。

## CIFAR-100 teacher / guide 质量校准（seed0 已完成）

该校准只回答两个问题：不同成熟度 guide 能否生成高质量 DSDM 图片，以及独立充分收敛
expert 能否提供可信的 sender logits。它不运行 communication 或 receiver，也不改写已完成
主实验。seed0 复用 PAT5 的 `5 agents x 20 class-disjoint classes`，使五种 backbone 各占一个
sender；所有分类头仍为全局 100 维，packet 只保存 sender 的 20 个 expert-class logits。

```text
ConvNet-3/4 guide: 10 trajectories, snapshots at 20/50/100/200 epochs
AlexNet guide:     10 trajectories, snapshots at 20/50/100/200 epochs
Standard R10/R18: 10 trajectories, fixed 200-epoch guide
DSDM per candidate: IPC=10, factor=2, 10,000 iterations
Evaluation checkpoints: 100/500/1000/2000/3000/5000/7500/10000
```

guide 轨迹保持对应 DSDM 训练形式，仅改变成熟度；这样 guide-epoch 扫描不混入额外 augmentation
变量。logit teacher 与 guide 完全分离：ConvNet 使用当前 centralized upper-bound 已验证的
500-epoch DSDM-style DiffAug + CutMix recipe，AlexNet 使用 500-epoch crop/flip recipe，标准
ResNet-10/18 使用 200-epoch crop/flip + cosine recipe。每个 teacher 从本地训练集按类固定留出
10% validation，每 5 epochs 检查一次，最高 validation accuracy 的最早 epoch 被选中，再从头
使用完整本地训练集重训相同 epoch 数；官方 expert-class test 不参与 epoch 选择。

正式质量门禁要求：DSDM 完整跑满、raw images 与 decoded logits 全部有限、raw image count 和
logit 维度正确、teacher checkpoint 带 SHA-256 且 `test_used_for_selection=false`。每个架构按
synthetic evaluator accuracy 选择 guide maturity，并同时报告 teacher validation/test accuracy、
packet-label agreement、logit entropy、top-1 margin 和幅值。性能门禁暂定为成熟 teacher 本地
test accuracy 不低于 70%、packet-label agreement 不低于 70%，以及最佳 synthetic accuracy
相对同一 PAT5 sender 的旧结果下降不超过 2 个百分点。ResNet 的旧参考来自 compact 版本，
因此只作为最低质量下限，不作为 standard-vs-compact 公平增益结论。

质量校准的完整证据位于 `outputs/teacher_quality_seed0_summary/summary.json`，五个 backbone
均通过结构、teacher、logit agreement 和 image quality 门禁。该校准使用 PAT5 的 20 类 sender，
因此其 packet 不能直接当作 100 类图片池。

## CIFAR-100 backbone-specific 全类 DSDM 图片池（运行中）

该阶段用于验证并保存五种 backbone 各自的全类图片池，不运行 social communication 或 receiver。
每个独立 run 使用完整 CIFAR-100 训练集的 `0-99` 类，保留 global class index；这是一种用于
前期逻辑验证的可复用图片池协议，不等同于主实验中 sender 只见自己 expert classes 的本地
DSDM，后续论文必须明确该协议差异。

```text
Backbones: ConvNet-3, ConvNet-4, AlexNet, standard ResNet-10, standard ResNet-18
Agents: 1 per run, agent_0 classes = 0-99, classifier output = 100
Guide pool: 10 models, selected maturity = 200 epochs
DSDM: IPC=10, factor=2, 10,000 iterations
Evaluation checkpoints: 100/500/1000/2000/3000/5000/7500/10000
Run names: cifar100_fullclass_dsdm_<backbone>_ipc10_seed0
```

该图片池只保存 synthetic images、global class indices、best-iteration manifest 和 packet
provenance，不在池构建阶段附加 sender logits；未来通信时按接收方需要的类别从对应
backbone 池索引图片。每个池的 `class_ids` 必须严格等于 `0-99`，每类 raw image 数为 10，
factor decode 后每类训练视图为 40。

## Tiny-ImageNet 扩展（50 类诊断完成，all-200 运行中）

Tiny-ImageNet 是下一数据集扩展，不属于当前完成的正式结果。目标协议为 200 类、4 agents ×
50 个 class-disjoint expertise、全局 `0-199` 标签和所有分类头 200 维输出。

centralized-200 backbone 第一层 seed0 已完成，clean validation top-1 为：标准 ResNet-18
`66.43`、AlexNet-Tiny `59.96`、标准 MobileNetV2 `48.64`、ConvNet-4-IN-w1.5 `46.83`。
clean validation 排除完整性报告发现的 7 个 train/val 完全重复且标签冲突的验证样本；训练集
保持原始 100,000 张不变。该结果支持先用标准 ResNet-18 验证 DSDM，暂不增加 ResNet-34/50。

第二层 sender 0（全局类 `0-49`）packet-quality 诊断已完成，不运行 logits、receiver 或
social learning：

```text
Guide: standard ResNet-18-Tiny, 11,271,432 parameters, output 200
Guide pool: sender-local 10 models x 100 epochs; two methods share byte-identical guides
Distillation: IPC=10, factor=2, 10,000 iterations, f_idx/idx_from=5
Evaluation checkpoints: 100/500/1000/2000/3000/5000/7500/10000
Synthetic evaluator: 300 epochs, repeat=1
Pair: pure DSDM vs DSDM + PCBN(all 20 BN layers, normalized, weight=10,000)
```

PCBN 权重 `10,000` 是初始量级校准值：在混合初始化 smoke 中，加权 PCBN loss 约占总 loss
的 9%，不是已完成超参搜索后的“最优值”。两组只比较 PCBN 开关/权重，其余配置、seed、
初始化逻辑和 guide pool 相同。

50 类单 seed clean expert-class synthetic self-evaluation 结果为：

| Iteration | Pure DSDM | DSDM + PCBN |
|---:|---:|---:|
| 100 | 31.3 | 30.3 |
| 500 | 31.5 | 31.5 |
| 1,000 | 32.5 | 32.2 |
| 2,000 | **32.5460** | 32.9 |
| 3,000 | 32.4 | 32.3 |
| 5,000 | 32.1 | **33.5869** |
| 7,500 | 31.9 | 30.5 |
| 10,000 | 31.8 | 32.6 |

PCBN 最佳值相对 pure DSDM 提高 `1.0408` 个百分点，但该结果仍是单 sender、单 seed 的
packet-quality diagnostic，不能写成完整 social-learning 提升。pure DSDM 最优快照来自
iteration 2,000，PCBN 最优快照来自 iteration 5,000；二者均没有被较低的最终评估覆盖。

50 类配对完成后 all-200 scaling diagnostic 已开始，继续使用 IPC=10，因此 raw synthetic
budget 从 `50 × 10 = 500` 增加到 `200 × 10 = 2,000`，factor=2 后为 8,000 张训练视图。
all-200 会重新训练 10 个只使用完整 200 类训练集的 ResNet-18 guides，不能复用只见过
`0-49` 类的当前 guide pool；纯 DSDM 与 PCBN 仍共享 byte-identical guides。PCBN 首轮固定
同一权重 10,000 用于规模扩展对照，该设置仍不代表最优超参。

每次验证分数刷新最优值时，同时原子更新 `synthetic/data_best.pt`，并在
`synthetic/history/best_iter_XXXXX.pt` 保留不可被后续验证覆盖的迭代快照；manifest 记录
best accuracy、iteration 和 PCBN provenance。后续仍需比较 self-guided 与 ConvNet local
encoder，并以 packet quality、new-class absorption 和 expert stability 决定主消融结论。

## 结果选择与追溯

1. Registry 角色为 `main/baseline/ablation/diagnostic/historical_evidence/tuning/invalid`。
2. 同协议优先最高 global；差值小于 0.5 时优先 new 更高且 forgetting 更低者。
3. 每个正式数字必须记录 config、metrics、seed 数、packet guide 和 caveat。
4. `RESULTS.md` 和 `paper_tables/` 是生成产物；原始事实来自 CSV 和 packet provenance。
5. 活动实验状态写入 `docs/operations/current_status.md`，不写入 `AGENTS.md`。
