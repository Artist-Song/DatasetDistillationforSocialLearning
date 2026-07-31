# PROJECT_SPEC.md

最后更新：`2026-07-29`

本文件是项目研究定位、当前实验协议、最优超参数和结果解释的唯一事实来源。
运行级结果明细见 `RESULTS.md`，历史过程见 `EXPERIMENT_LOG.md`。

## 项目定位

项目名称：`DatasetDistillationforSocialLearning`

暂定题目：

```text
Distilled Knowledge Packets for Decentralized Heterogeneous Socialized Learning
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

当前 ICLR 2027 计划的论文主指标：

```text
acc_global   : 100 类全局准确率
acc_new      : receiver 未拥有的类别准确率；首轮5-agent协议为80类
acc_expert   : receiver 原有类别准确率；首轮5-agent协议为20类
```

`Global` 是首要指标，`New` 和 `Expert` 是共同主指标。`forgetting = acc_expert_before -
acc_expert_after` 只作内部诊断，不进入论文主表或方法排名。隐私保护和复杂网络拓扑都不是当前
主要贡献，不作强声明。

## 当前计划/正在验证的 ICLR 2027 协议

本节定义当前研究计划，暂定方法名为 `Distilled Knowledge Packet (DKP)` 和
`DKP-based Socialized Learning (DKP-SL)`。本节内容是正在验证的协议，不是已完成正式结果；
在五个 receiver 全部完成并通过聚合门禁之前，不得把单 receiver、partial run 或诊断数值写入
正式结果表。

### 任务与扩展标准

```text
Dataset: CIFAR-100
Current validation: 5 agents x 20 mutually disjoint classes
Planned agent-count extensions: 10 agents x 10 classes; 20 agents x 5 classes
Global labels: 0-99, never persistently remapped
Classifier output: 100 for every model
Communication: one DKP per sender, one logical all-share
Aggregation: no parameter averaging, no shared global model, no model-homogeneity requirement
Infrastructure: no central training server
```

5/10/20-agent 是当前扩展标准；首轮只运行 CIFAR-100、5 agents、IPC=10、seed0。10-agent 和
20-agent 在首轮完整验证结束前不得启动。seed0 类别来自
`numpy.random.RandomState(0).permutation(100)`，当前5组为：

```python
agent_class_split = {
    0: [26, 86, 2, 55, 75, 93, 16, 73, 54, 95, 53, 92, 78, 13, 7, 30, 22, 24, 33, 8],
    1: [43, 62, 3, 71, 45, 48, 6, 99, 82, 76, 60, 80, 90, 68, 51, 27, 18, 56, 63, 74],
    2: [1, 61, 42, 41, 4, 15, 17, 40, 38, 5, 91, 59, 0, 34, 28, 50, 11, 35, 23, 52],
    3: [10, 31, 66, 57, 79, 85, 32, 84, 14, 89, 19, 29, 49, 97, 98, 69, 20, 94, 72, 77],
    4: [25, 37, 81, 46, 39, 65, 58, 12, 88, 70, 87, 36, 21, 83, 9, 96, 67, 64, 47, 44],
}
```

五组必须各含20类、两两互斥且并集严格为 `0-99`。首轮模型映射固定为：

| Agent | Backbone | ResNet definition |
|---:|---|---|
| 0 | ConvNet-3-w1.0 | - |
| 1 | ConvNet-4-w1.5 | - |
| 2 | AlexNet | - |
| 3 | ResNet-10-CIFAR standard | 3x3 stride-1 stem, base width 64, blocks `[1,1,1,1]` |
| 4 | ResNet-18-CIFAR standard | 3x3 stride-1 stem, base width 64, blocks `[2,2,2,2]` |

历史 compact ResNet 继续保留在4-agent/PAT历史证据中，但当前新 run 禁止将 compact checkpoint
或图片池静默映射为 standard ResNet。

### 图片池与蒸馏解耦

首轮不重新蒸馏。通信验证从每种 backbone 已完成、完整性通过且 synthetic evaluator 性能最高的
IPC=10 全100类图片池中，按 sender 的20类集合切片。图片池在构建时见过全100类，因此本轮必须
标为 `backbone-specific full-class pool slice communication diagnostic`，不能冒充 sender-local
DSDM 正式主结果。只有完整性验证失败时才允许考虑重蒸馏。

| Sender backbone | Selected full-class source | Pool diagnostic accuracy | Source packet SHA-256 |
|---|---|---:|---|
| ConvNet-3-w1.0 | pure-200 | 46.54 @10000 | `07db12b9462ee6726b01ad8f78a469f9d887d742f06d75d4bbc44240a9438bf5` |
| ConvNet-4-w1.5 | pure-200 | 48.57 @9500 | `294e2eb86dcc393c2ee7f476d12ae2714878c04fa2540c9ddf0440c1104f25d0` |
| AlexNet | historical-e20 recovery | 46.05 @5500 | `75e9ef5f0cb525b4e793a043c3737ba6e84b97b84e58bd00749e25ccb3f849ce` |
| ResNet-10 standard | model-specific + PCBN w960 | 39.22 @5000 | `e2339df5d22531fd4d54ce2040f8d4f4b550e99a0cef53b1c87b7bcdd97e9959` |
| ResNet-18 standard | model-specific recovery | 39.90 @7500 | `5a79190b64f552e00490596cb3792a196cf00e56f74ecda515c7d9d6e3e7a3ce` |

表中 accuracy 只是图片池选择证据，不是 DKP-SL 通信结果。中断的 R10 PCBN `w1300/w2100`
没有完整10,000 iterations和 completion provenance，不得参与“最高图片池”选择。
首轮切片固定使用 catalog `configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml` 和
pool protocol `backbone_specific_fullclass_pool_slice_v2_dkp`；catalog 将每个 backbone 绑定到
精确的 source model id、source packet SHA-256、best iteration及best snapshot SHA-256。

### Expert、guide 与 DKP

DSDM guide pool 只负责生成蒸馏图片。每个 agent 的最终 expert checkpoint 与 guide pool 分离；
同一个最终 expert checkpoint 的 SHA-256 必须同时对应：

1. receiver 通信后模型初始化；
2. receiver 的冻结 FR feature 教师；
3. sender 生成自身20类 expert-class logits 的教师。

DKP 的逻辑内容为 raw distilled images、global hard labels 和 sender expert-class logits。每个
sender 为 `20 x IPC10 = 200` 张 raw images；`factor=2` 确定性解码后为每类40张、共800张训练图。
通信图片数仍按200张 raw images 计。logits 只保存 sender 自己20类的列，并逐行对应800张
deterministic decoded images；不得保存完整100类 logits。`sender_logit_class_ids` 必须严格等于
该 sender 配置中的有序20类集合。首轮诊断将对齐后的 logits 持久化并传输为 `float16`，因此每个
sender 为 `800 x 20 x 2 = 32,000` bytes；每个 receiver 从四个外部 sender 接收
`128,000` logit bytes。validator 必须按实际 tensor dtype/shape 重新计算，不能只信配置值。

### Cosine expert classifier

Cosine classifier 是当前方法的核心新设计，从 expert 训练开始启用。五种 backbone 使用完全相同
的分类头定义：

```text
logits = s * normalize(penultimate_feature) @ normalize(class_weight)^T
bias = none
s = softplus(raw_scale)
initial s = 10.0
one learnable scalar s per model; s is excluded from weight decay
```

分类头始终有100行。expert 训练 CE 只在本地20类列上计算，标签在临时索引中参与 loss，但 artifact
和数据中的标签始终保持全局 `0-99`，不得持久化局部标签重映射。Local expert 只在本地20类 mask
内评估；未训练的外部80类权重不得参与 expert-class argmax。

本轮不做 cosine-specific LR/scale sweep。沿用已验证 optimizer/data recipe，把500/200仅作为
validation 候选时域；固定留出10%本地分层 validation、每5 epochs检查、并列取最早最高点，随后
从头用全部本地真实数据重训至所选 epoch。test 不参与选择。

| Backbone | Optimizer/data recipe | Validation ceiling |
|---|---|---:|
| ConvNet-3/4 | SGD, lr=0.01, momentum=0.9, wd=5e-4, batch=128, DSDM DiffAug+CutMix, MultiStep 333/416 gamma0.2 | 500 |
| AlexNet | SGD, lr=0.01, momentum=0.9, wd=5e-4, batch=128, crop/flip, MultiStep 333/416 gamma0.2 | 500 |
| standard R10/R18 | SGD, lr=0.1, momentum=0.9, wd=5e-4, batch=128, crop/flip, cosine LR with `T_max=200` | 200 |

旧 linear-head 的最佳 epoch `385/400/385/140/180` 只作候选时域证据，不能声明为新 cosine head
的最优 epoch，也不能直接复用旧 linear expert checkpoint。

### Prototype initialization 与 receiver 双流

通信训练前，receiver 使用冻结 expert encoder 对每个外部类别的全部40张 deterministic decoded
图提取 penultimate feature；先求类均值，再 L2 normalize，写入该 external class 的 cosine weight。
本地20类权重必须逐元素保持 expert checkpoint 值。全部80个 new-class prototype finite、单位范数
且本地权重未被覆盖后，才解除全类 mask。

每个 optimizer step 同时使用两个 class-balanced stream：

```text
local stream:    64张 receiver 本地真实图；一个epoch先完整遍历20 x 500张本地训练图，
                 再从该epoch的seeded顺序补48张，形成157个固定64张batch
external stream: 64张其他四个 sender 的 decoded DKP；较短loader耗尽后重排并循环
steps/epoch:     由local stream决定
```

视图和 loss 分工固定为：

```text
local crop/flip view:
  CE_local + frozen/current penultimate feature cosine FR on the identical view

external deterministic decoded view:
  sender-class KD only; fixed teacher logits never supervise a random crop

two independent external crop/flip views:
  mean CE_external + SupCon on normalized penultimate features
```

SupCon 不增加 projection head；同一图像的另一视图和 batch 内同标签视图都是正样本。CE 按类别数
固定为 `0.2 * CE_local + 0.8 * CE_external`，使20个 local classes 与80个 external classes 的
单类期望权重相同。FR 不再叠加 expert-logit MSE；KD 只在对应 sender 的20列上计算。总损失为：

```text
CE + lambda_FR * FR + lambda_KD * KD + lambda_SC * SupCon
```

### 首轮诊断运行

第一轮覆盖全部五个 receiver，并依次运行：

1. `Local cosine expert`：只评本地20类；无通信。
2. `DKP images + CE-only`：保留 cosine classifier、prototype initialization和双流 CE，关闭
   FR/KD/SupCon；逻辑通信计 `logit bytes=0`。
3. `DKP-SL full`：启用 CE、FR、sender-class KD 和 SupCon；记录实际 transmitted logit bytes。

固定诊断参数不是当前协议的正式最优超参：

| Parameter | Value |
|---|---:|
| receiver epochs / batch per stream | 60 / 64 |
| receiver lr | 0.01 |
| lambda_FR / lambda_KD / KD temperature | 0.2 / 0.6 / 2 |
| lambda_SC / SupCon temperature | 0.1 / 0.07 |
| scheduler | MultiStep 39/51, gamma 0.2 |

`lambda_SC=0.1` 和 temperature `0.07` 明确标为 `diagnostic provisional`，不得写成正式最优值。
首轮不做 test-set 超参搜索。三组完成前不追加 linear-head消融；三组全部通过后再运行同协议
linear-head内部消融以隔离分类头贡献。MASC、DeSA、FedRE 等外部 baseline 和完整 IPC/seed/agent
网格在首轮完成前全部暂停。

### 首轮完成状态（通信诊断，非正式结果）

seed0 首轮于 `2026-07-27` 完整结束。五个 local expert、五个 CE-only receiver 和五个 full
DKP-SL receiver 均完成，launcher 和全部 receiver 子进程 `exit=0`。对应不可覆盖 run 为：

```text
cifar100_5agent20cls_dkp_cosine_experts_seed0_v1
cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2
cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2
```

下表是 full-class pool slice 协议的完整 seed0 通信诊断，不是 sender-local DSDM 正式论文结果，
不得进入正式主表或与历史4-agent结果混合：

| Receiver | Local expert (20类) | CE-only G / N / E | Full G / N / E | Full - CE Global |
|---|---:|---:|---:|---:|
| ConvNet-3 | 79.05 | 18.950 / 4.538 / 76.600 | 20.830 / 7.038 / 76.000 | +1.880 |
| ConvNet-4 | 83.20 | 23.760 / 9.163 / 82.150 | 28.290 / 15.000 / 81.450 | +4.530 |
| AlexNet | 83.80 | 33.370 / 23.150 / 74.250 | 37.330 / 29.000 / 70.650 | +3.960 |
| ResNet-10 standard | 86.20 | 36.060 / 26.775 / 73.200 | 35.070 / 26.063 / 71.100 | -0.990 |
| ResNet-18 standard | 85.15 | 36.940 / 29.175 / 68.000 | 33.800 / 24.350 / 71.600 | -3.140 |
| 五 receiver 均值 | 83.480 | 29.816 / 18.560 / 74.840 | 31.064 / 20.290 / 74.160 | +1.248 |

每个 receiver 均为 `9420` optimizer steps、接收800张 external raw images；CE-only 的 transmitted
logit bytes 为0，full 为128,000。一次逻辑全共享的唯一 sender payload 是1000张 raw images和
160,000 logit bytes；汇总文件中的4000 images和640,000 bytes是按五个 receiver incidence 求和，
不能当作唯一网络 payload。full 的平均训练计算时间为538.93秒/receiver；并行 receiver 的时间求和
不是队列 wall-clock。

完整方法相对 CE-only 的 seed0 均值提高 `Global +1.248`、`New +1.730`，同时 `Expert -0.680`。
三个非 ResNet receiver 的 Global 都提高，而 standard R10/R18 分别下降0.99/3.14；这是后续固定
loss组件消融要定位的 backbone-dependent 现象，不是 packet/provenance 门禁失败。首轮汇总位于
`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/first_round_seed0_summary.json`，其中
明确记录 `status=complete_diagnostic`、`formal_result=false`、`paper_eligible=false`。最终十个
receiver checkpoint 的 strict-load、全 tensor finite、`[2,100]` 输出、finite feature、loss/metric
恒等式和 source SHA 复验记录在同目录的 `final_receiver_checkpoint_audit.json`，状态为 `passed`。

### 首轮后的固定内部诊断计划

分类头贡献使用重新训练的 matched linear expert 进行端到端对照，不能复用 cosine checkpoint，也
不能只把 receiver 配置改成 `linear`。linear 外部类的 weight imprinting 固定为：

```text
p_c = normalize(mean frozen-linear-expert feature of all 40 decoded images for class c)
alpha = mean L2 norm of the 20 trained local classifier rows
beta = mean bias of the 20 trained local classifier rows
W_external[c] = alpha * p_c
b_external[c] = beta
```

本地20类的 `W/b` 必须逐元素不变，`alpha` 必须有限且为正，外部80类行范数必须等于 `alpha`。
这一定义是无 test 调参的 norm-matched linear ablation；linear expert 仍输出100维、使用全局标签和
本地20类 masked CE，并沿用相同 data/optimizer/validation-selection recipe。

FR/KD/SupCon 只做固定二元组件消融，不搜索 lambda 或 temperature。已有 `000=CE-only` 和
`111=full`，后续诊断补 `100/010/001/110/101/011`；启用的组件继续使用
`lambda_FR=0.2`、`lambda_KD=0.6`、`lambda_SC=0.1`。只有 `KD=1` 的 run 传输 sender logits并计
128,000 bytes/receiver，`KD=0` 必须为0 bytes。所有新增 run 使用新 run_name、
`checkpoint_retention=final_only` 和 `paper_eligible=false`。receiver RNG repeat 如运行，必须明确
标为固定 dataset/class/pool/expert seed0 下的训练稳定性诊断，不能冒充独立 dataset/distillation seed。
在这些内部诊断闭合前，继续不启动外部 baseline 或10/20-agent及完整IPC/seed网格。

### 夜间固定内部诊断完成状态（非正式结果）

`2026-07-28` 完成 matched-linear 端到端对照和预先固定的 FR/KD/SupCon
`2^3` 组件矩阵。所有新 run 都是 seed0、IPC10、full-class pool slice 通信诊断，
`formal_result=false` 且 `paper_eligible=false`；它们不是 sender-local DSDM 正式论文结果。

Matched-linear 使用独立重训的 linear expert 和已固定的 norm/bias-matched imprinting：

| Receiver | Local linear expert | Linear CE-only G / N / E | Linear full G / N / E | Cosine full - linear full Global |
|---|---:|---:|---:|---:|
| ConvNet-3 | 77.600 | 18.730 / 4.700 / 74.850 | 20.420 / 7.013 / 74.050 | +0.410 |
| ConvNet-4 | 82.500 | 21.830 / 6.888 / 81.600 | 27.730 / 14.738 / 79.700 | +0.560 |
| AlexNet | 83.000 | 33.540 / 25.763 / 64.650 | 33.870 / 23.713 / 74.500 | +3.460 |
| ResNet-10 standard | 85.700 | 33.090 / 23.675 / 70.750 | 34.000 / 24.275 / 72.900 | +1.070 |
| ResNet-18 standard | 84.650 | 35.090 / 29.700 / 56.650 | 36.230 / 30.500 / 59.150 | -2.430 |
| 五 receiver 均值 | 82.690 | 28.456 / 18.145 / 69.700 | 30.450 / 20.048 / 72.060 | +0.614 |

Cosine full 相对 matched-linear full 的五 receiver 均值差为 `Global +0.614 / New +0.243 /
Expert +2.100`，但 R18 的 Global 为 `-2.430`；因此当前只能说 cosine head 在 seed0 上有较小的
平均改善，且存在 backbone-dependent 例外。这是重训整条 expert/receiver pipeline 的处理效应，
不是在同一 feature checkpoint 上直接换头的局部测量。

Cosine expert 下的固定 loss 矩阵如下；三位二进制顺序为 `FR/KD/SC`：

| FR/KD/SC | 五 receiver Global / New / Expert | Global - CE-only | Logit bytes / receiver |
|---|---:|---:|---:|
| 000 | 29.816 / 18.560 / 74.840 | +0.000 | 0 |
| 001 | 30.696 / 20.648 / 70.890 | +0.880 | 0 |
| 010 | 30.232 / 19.698 / 72.370 | +0.416 | 128,000 |
| 011 | 30.506 / 20.958 / 68.700 | +0.690 | 128,000 |
| 100 | 31.662 / 20.503 / 76.300 | +1.846 | 0 |
| 101 | 32.172 / 21.405 / 75.230 | +2.356 | 0 |
| 110 | 30.832 / 19.700 / 75.360 | +1.016 | 128,000 |
| 111 | 31.064 / 20.290 / 74.160 | +1.248 | 128,000 |

固定矩阵中 `101=FR+SupCon` 的 seed0 均值最高，相对 `111=full` 为
`Global +1.108 / New +1.115 / Expert +1.070`，且不传输 logits。但该观察不均匀：ConvNet-3 和
R10 的最高 Global 来自101，ConvNet-4 来自100，AlexNet 来自111，R18 来自010：

| Receiver | 固定矩阵中最高 Global 条件 | Global / New / Expert |
|---|---:|---:|
| ConvNet-3 | 101 | 21.180 / 6.950 / 78.050 |
| ConvNet-4 | 100 | 29.110 / 15.575 / 83.250 |
| AlexNet | 111 | 37.330 / 29.000 / 70.650 |
| ResNet-10 standard | 101 | 38.990 / 30.925 / 71.250 |
| ResNet-18 standard | 010 | 37.300 / 30.475 / 64.600 |

这些都是完成固定矩阵后对 test metrics 的回顾性观察，不能把101追认为正式最优方法；
需要先在独立 seed 上预注册确认，再决定是否修订 full 协议。

Matched-linear 的10个和 loss matrix 的30个最终 receiver checkpoint 均通过 strict-load、
finite state/output/feature、`[2,100]` 输出、loss/metric 恒等式、checkpoint/provenance SHA 和
`final_only` 保留审计。汇总与审计证据分别为：

```text
outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_head_seed0_summary.json
outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_final_receiver_checkpoint_audit.json
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_v1.json
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_final_checkpoint_audit_v1.json
```

两组诊断都只有一个 dataset/class/pool/expert seed，没有独立 seed 方差，也不能消除
full-class pool 构建时见过全100类的 caveat。在独立 seed 复核前，不启动外部 baseline、
10/20-agent 或完整 IPC 网格。

### Receiver 域平衡与固定步数诊断（已完成内部诊断，非正式结果）

首轮与固定 loss 矩阵显示 ConvNet-3/4 的 New 明显低于同一 seed0 5x20 split 的历史 PAT5，
而 prototype 初始化后、联合训练前的 New 并不低。当前诊断检验两个预先提出的机制：
`9420` 次更新是否使 decoded packet 过度重复，以及“local real=expert classes、external
synthetic=new classes”的数据域与类别组完全绑定是否形成域捷径。该诊断不重新蒸馏、不改变
cosine expert、prototype、packet、类别划分或通信预算，也不是 test-set 超参 sweep。

固定使用 loss 条件 `FR=0.2 / KD=off / SupCon=0.1`，因为它是已完成二元矩阵中 Global 最高的
内部诊断条件；该选择仍不是正式方法最优声明。三个 local CE 来源为：

```text
S (real):                CE_local = CE(local real)
U (packet):              CE_local = CE(receiver self-DKP)
H (real_packet_50_50):   CE_local = 0.5 CE(local real) + 0.5 CE(receiver self-DKP)
all:                     CE = 0.2 CE_local + 0.8 CE_external
                         FR = local real only
                         SupCon = external two-view only
```

receiver self-DKP 必须是该 receiver 作为 sender 的唯一 packet：20类、raw 200、factor=2 后每类
40张且共800张 decoded images。它是本地已持有的 sender artifact，不计 external communication，
也不进入 KD、SupCon或80类 prototype 初始化。U 中 local real 仍只用于 FR；H 中同一 local-real
增强视图同时用于一半 local CE 和 FR。S/U/H 的 external communication 均保持800张 raw images、
0 transmitted logit bytes。

三组均已精确完成 `3780` optimizer steps，以匹配历史 PAT5 的 `60 x 63` 更新数；MultiStepLR 在完成
第 `2457/3213` 次更新后以 `gamma=0.2` 衰减，即影响下一次更新。15个 receiver 均 exit 0，严格
packet/provenance/SHA、final checkpoint finite state 和五 receiver coverage 检查通过。以下数字仅是
`paper_eligible=false` 的 seed0 内部机制诊断，不进入 `RESULTS.md` 或论文正式表：

| Local CE 条件 | Global | New | Expert |
|---|---:|---:|---:|
| S: real | 31.922 | 22.1075 | 71.180 |
| U: self-DKP | 38.708 | 34.2700 | 56.460 |
| H: 50% real + 50% self-DKP | 33.116 | 22.4100 | 75.940 |

U 相对历史同一 seed0 5x20 split 的 PAT5 为 `Global +8.798 / New +6.8075 / Expert +16.760`；
但 ConvNet-3 的 Expert 仍比其 PAT5 低3.65点。H 恢复 Expert 的同时使 New 大幅退回，因此下一步只在
U 附近检验较小 real-CE 权重；当前证据不支持先加大 FR。

### Packet-heavy real-CE 权重曲线（已完成内部诊断，非正式结果）

预先固定 `local_ce_real_fraction = 0.05/0.10/0.20/0.30`，四组都覆盖五个 receiver，不按中间
test 指标提前筛选或停止。这里的 fraction 是
`CE_local = fraction * CE(local real) + (1-fraction) * CE(self-DKP)` 的 loss 权重，不是样本抽样
比例；两条 local CE stream 每个 optimizer step 各取一个类平衡 batch。其余设置与U完全相同：
cosine head、80类 prototype initialization、`FR=0.2`、`KD=off`、`SupCon=0.1`、3780 steps、
step milestones 2457/3213、IPC10、seed0和相同 expert/packet provenance。四个 run 使用独立
`run_name`、`checkpoint_retention=final_only`、`paper_eligible=false`，不得依据本曲线声称正式最优参数。

r05/r10/r20/r30 的 self-DKP forward 结构完全一致，因此四者是受控的 loss-weight 曲线。S 不读取
self-DKP CE，而 U/H 会额外对 self-DKP forward；在 AlexNet/standard ResNet 上这也会改变 BatchNorm
running-stat exposure，所以 S 与 U/H 不能解释为纯 loss-weight 因果消融。若后续需要消除该混杂，
必须以新 run_name 统一冻结 BN 或统一 forward exposure，不能修改或重解释本轮协议。

四组各5个 receiver 均已完成，20个进程全部 exit 0；strict summary 和独立 final-checkpoint
strict-load/output/get_feature/cosine/SHA 审计均通过。以下仍是 seed0 内部诊断，不是正式结果：

| real-CE loss 权重 | Global | New | Expert |
|---:|---:|---:|---:|
| 0.05 | 37.930 | 30.4725 | 67.760 |
| 0.10 | 36.704 | 28.1325 | 70.990 |
| 0.20 | 35.232 | 25.6875 | 73.410 |
| 0.30 | 34.186 | 23.9650 | 75.070 |

r05 相对旧 PAT5 为 `Global +8.020 / New +3.010 / Expert +28.060`，但相对U为
`Global -0.778 / New -3.7975 / Expert +11.300`。曲线显示加入少量 real CE 的确能恢复 Expert，
但5%已使 Global 相对U的差超过0.5点，且 ConvNet-3 的 New=29.725 仍低于其旧 PAT5=31.600。

### r02 曲线后确认（已完成内部诊断，非正式结果）

为检验U与r05之间是否存在更合适的折中，追加唯一一个 `local_ce_real_fraction=0.02` 的五 receiver
确认。该点是在查看完成曲线后依据 `Global = 0.2 Expert + 0.8 New` 选出的自适应诊断，不是预注册
超参，也不得用于正式最优声明；必须在独立 seed 重新预注册后才可能进入方法协议。局部线性估计为
相对U `Global -0.31 / New -1.52 / Expert +4.52`，这里只把它作为待验证假设，不写成结果。
r02 保持相同 expert、packet、cosine/prototype、FR+SupCon、KD off、3780 steps、调度与通信计量，
使用独立新 run_name，并完整覆盖五个 receiver，不按中间 test 指标停止。五个任务全部 exit 0，联合
strict summary 和25个 curve+r02 final checkpoint 审计通过：

| Receiver | r02 Global | r02 New | r02 Expert |
|---|---:|---:|---:|
| ConvNet-3 | 37.940 | 34.0750 | 53.400 |
| ConvNet-4 | 40.830 | 34.9250 | 64.450 |
| AlexNet | 38.550 | 33.5375 | 58.600 |
| ResNet-10 standard | 39.520 | 32.2625 | 68.550 |
| ResNet-18 standard | 35.660 | 27.1875 | 69.550 |
| 五 receiver 均值 | 38.500 | 32.3975 | 62.910 |

r02 相对U为 `Global -0.208 / New -1.8725 / Expert +6.450`，Global 差在0.5点以内；相对r05为
`Global +0.570 / New +1.925 / Expert -4.850`；相对旧 PAT5 为
`Global +8.590 / New +4.935 / Expert +23.210`。r02 的每个 receiver 三项均高于旧 PAT5 对应项，
但这不能消除 standard/compact ResNet 定义不同的可比性 caveat。

相对U，ConvNet-3/4/AlexNet 的 r02 Global 为 `+0.54/+0.49/+0.06`，R10/R18为
`-0.36/-1.77`，R18 是当前主要 backbone 例外。只读 group-masked 诊断显示，R18 的 New 类内
accuracy 从37.1875降到35.400，且 New 样本预测到本地 Expert 类的比例从26.9%升到31.5%；因此
主要问题仍是跨组校准向 Expert 偏移，并非 prototype 新类整体失效。r02 可作为下一独立 seed 的
预注册候选，但由于它是查看 seed0 test 曲线后的自适应点，当前不能升格为正式方法或最优超参。

### r02 KD-on 配对确认（已完成通信诊断，非正式结果）

在 r02 KD-off 完整闭合后，追加一个五 receiver 的 `full` 配对确认：除启用 sender-class KD
(`lambda_KD=0.6`, `T=2`) 和对应 `800 x 20 float16` sender logits 外，其余 expert、raw images、
local real fraction、FR、SupCon、3780 steps和调度必须完全相同。每个 receiver 的 transmitted
logit bytes 必须为128,000，raw external images仍为800。该确认用于判断域平衡后完整 DKP-SL 是否
优于 r02 FR+SupCon，不是新 sweep，也仍是查看 seed0 诊断后的自适应实验。

五个 receiver 均已完成并通过最终 checkpoint 审计。完整方法的
`Global/New/Expert=39.632/34.290/61.000`，相对严格配对的 KD-off r02 为
`+1.132/+1.8925/-1.910`。逐 receiver 如下：

| Receiver | Full Global | Full New | Full Expert |
|---|---:|---:|---:|
| ConvNet-3 | 38.910 | 35.6250 | 52.050 |
| ConvNet-4 | 40.880 | 35.0750 | 64.100 |
| AlexNet | 39.510 | 35.3375 | 56.200 |
| ResNet-10 standard | 40.550 | 33.8375 | 67.400 |
| ResNet-18 standard | 38.310 | 31.5750 | 65.250 |

该 run 为
`cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2`，状态仍是
`complete_diagnostic`、`formal_result=false`、`paper_eligible=false`，不能把 seed0 自适应确认
写成正式最优参数或多 seed 正式结果。汇总和最终 checkpoint 审计为：

```text
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_v2.json
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_final_checkpoint_audit_v2.json
```

用户已确认：蒸馏阶段用 evaluator 比较候选 snapshot 并选择知识价值最高的蒸馏图片，是本项目接受
的数据集蒸馏选择流程，不作为后续通信验证的阻断项。后续固定复用这批图片并保留选择 provenance，
不得在多 seed 或多 agent 运行中重新按 receiver test 指标选择图片或覆盖 source artifact。

### 下一阶段：IPC10 多 seed 与多 agent（当前计划/正在验证）

下一阶段保持 r02 full 的方法、图片池和通信超参固定，先做新的5-agent类别划分/训练 seed，再扩到
10-agent和20-agent。新 seed 必须训练 fresh cosine expert；seed0 expert、sender logits、prototype
和 receiver checkpoint 均不得跨类别划分复用。五个 full-class IPC10 pool 可按新 sender 类别集合
重新切片，不重新蒸馏。

独立复核种子预注册为 `seed1/seed2/seed3`；类别排列分别使用
`numpy.random.RandomState(seed).permutation(100)`。seed0 已用于方法诊断和 r02 自适应确认，不进入
该独立多 seed 估计。每个 agent 数和 seed 的 fresh expert、DKP-SL、Heuristic、FAST、Full Real
配置均使用独立 run name；45份配置由 `scripts/prepare_iclr2027_dkp_scaling.py` 生成并要求幂等一致。

10-agent和20-agent采用与5-agent嵌套的类别划分：每个原20类 macro-agent 分别拆为2个10类或
4个5类子 agent，子 agent 保持其 macro-agent 对应 backbone。因此五种 backbone 在10-agent中
各出现2次，在20-agent中各出现4次。CE 类别权重按实际类别数动态计算：

```text
5 agents:  0.20 * CE_local + 0.80 * CE_external
10 agents: 0.10 * CE_local + 0.90 * CE_external
20 agents: 0.05 * CE_local + 0.95 * CE_external
```

IPC10、factor=2 和 float16 sender-class logits 下的精确单次 all-share 计量为：

| Agents × classes | Raw / sender | Decoded / sender | Sender-logit shape | External raw / receiver | Logit bytes / receiver |
|---|---:|---:|---:|---:|---:|
| 5 × 20 | 200 | 800 | `800 × 20` | 800 | 128,000 |
| 10 × 10 | 100 | 400 | `400 × 10` | 900 | 72,000 |
| 20 × 5 | 50 | 200 | `200 × 5` | 950 | 38,000 |

三种规模的唯一 sender raw payload 都是1000张；factor decode 不增加通信图数。Heuristic 和 FAST
只传相同 IPC10 真实图与 hard labels，不带 logits；Full Real 每类500张并使用独立 full-data
comparability group。三类 hard-label 配置沿用同一 fresh expert、r02双流 receiver和3780步训练，
用于主方法阶段完成后的配对对比。MASC/DeSA/FedRE 仍按各自参数、anchor/logit或表示通信对象单独
计量，不写成 image-only，也不与 hard-label image-budget group 混合。

用户在主网格继续前指定先运行一个 `5-agent seed1` Heuristic sanity baseline。该唯一提前运行的
内部 image baseline 已完整覆盖5个receiver；packet 为每类按 seed1 确定性随机抽取10张 sender-local
真实训练图，hard labels、无 logits，总计1000张 raw/train images。五个 packet 重新抽样后逐张
张量完全一致，packet-hub/source SHA和复用 expert SHA均一致。五个 receiver 均完成预注册的3780步，
最终 checkpoint strict-load、100维输出、finite state/output、CSV/provenance SHA和独立 checkpoint
重评均为 `5/5` 通过。

| seed1 method | Global | New | Expert | Forgetting（诊断） |
|---|---:|---:|---:|---:|
| DKP-SL full | 38.810 | 33.1475 | 61.460 | 21.560 |
| Heuristic hard | 33.660 | 25.9925 | 64.330 | 18.690 |
| DKP-SL - Heuristic | +5.150 | +7.1550 | -2.870 | +2.870 |

因此按registry预先固定的首要 Global 指标，seed1 sanity gate 通过；Global/New分别有4/5个配对
receiver更高。该结果并非全指标支配：Expert平均较低且诊断 forgetting 较高，standard R10 的
Global/New也分别低0.38/0.8875点。Full DKP-SL另外传输每receiver 128,000 bytes sender logits，
Heuristic不传 logits；两者属于相同图片预算比较，但不是相同总字节或纯图片选择消融。该单seed
结果保持 `formal_result=false`、`paper_eligible=false`，必须等待 seed2/3 后才形成多seed估计；
证据为
`outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed1_heuristic_sanity_v1.json`。

用户在启动 baseline 前更新并最终确认当前执行矩阵。内部方法中，Ours、Heuristic 和 FAST 均运行
`5/10/20 agents × seeds1/2/3`，用于检验方法优势与真实图 coreset 在 sender 粒度变化下是否稳定；
Full Real 作为高通信量 oracle，只运行标准 `5 agents × seeds1/2/3`。因此内部结果矩阵共30个
method-scale-seed 单元：Ours 9、Heuristic 9、FAST 9、Full Real 3；每个单元必须覆盖该规模全部
receiver。已经生成的10/20-agent Full Real配置继续保留为未调度候选，不删除、不进入当前队列。

外部 baseline 另运行27个独立单元：DeSA-CIL、同构MASC-complete和FedRE各自运行
`5/10/20 agents × seeds1/2/3`，但不与 image-only comparability group 混合。DeSA使用当前五种
异构task backbone及对应fresh expert初始化，并通信1000张合成anchor和每轮owner-class logits；
MASC使用官方同构Netwider13、IPC10且总计1000张CC真实图，同时单列expert上传和collective student
下发的参数字节；FedRE保持官方训练核心、RAP、512x100共享head和100 rounds，5-agent取官方HtM10
模型列表前5个、10-agent使用完整列表、20-agent将该列表按原顺序重复两次。FedRE只替换为当前
seed1/2/3的5/10/20-agent互斥全局类别划分，并保留CIFAR-100官方train/test。

截至`2026-07-31 03:46 UTC`，完整研究矩阵的57个method-scale-seed单元均已完成：Ours 9、
Heuristic 9、FAST 9、Full Real 3、DeSA-CIL 9、MASC-complete 9、FedRE 9。每个seed先覆盖该规模
全部receiver再求seed内均值；表中为三个seed-level mean的均值和population std，单位为百分比：

| Method | 5 agents Global / New / Expert | 10 agents Global / New / Expert | 20 agents Global / New / Expert |
|---|---|---|---|
| Ours | 39.18 +/- 0.52 / 33.72 +/- 0.53 / 61.05 +/- 0.91 | 34.02 +/- 0.17 / 31.39 +/- 0.20 / 57.75 +/- 0.18 | 29.67 +/- 0.21 / 28.23 +/- 0.22 / 57.14 +/- 0.08 |
| Heuristic | 33.60 +/- 0.59 / 25.91 +/- 0.75 / 64.40 +/- 0.09 | 26.76 +/- 0.45 / 23.98 +/- 0.51 / 51.76 +/- 0.89 | 19.22 +/- 0.63 / 16.82 +/- 0.66 / 64.87 +/- 0.21 |
| FAST | 31.67 +/- 0.29 / 23.54 +/- 0.44 / 64.19 +/- 0.36 | 24.15 +/- 0.12 / 21.14 +/- 0.15 / 51.29 +/- 0.22 | 17.25 +/- 0.06 / 14.80 +/- 0.07 / 63.65 +/- 0.84 |
| Full Real | 58.07 +/- 1.28 / 54.45 +/- 2.07 / 72.57 +/- 1.88 | - | - |
| DeSA-CIL* | 14.64 +/- 0.83 / 1.17 +/- 0.12 / 68.51 +/- 3.65 | 9.70 +/- 0.07 / 2.31 +/- 0.13 / 76.18 +/- 0.78 | 6.39 +/- 0.07 / 2.27 +/- 0.03 / 84.64 +/- 1.19 |
| MASC-complete* | 8.96 +/- 0.12 / 1.96 +/- 0.25 / 37.00 +/- 1.60 | 6.24 +/- 0.24 / 2.11 +/- 0.36 / 43.48 +/- 1.15 | 4.75 +/- 0.08 / 1.80 +/- 0.12 / 60.87 +/- 1.86 |
| FedRE | 11.69 +/- 0.02 / 0.00 +/- 0.00 / 58.41 +/- 0.12 | 6.31 +/- 0.03 / 0.00 +/- 0.00 / 63.06 +/- 0.28 | 3.61 +/- 0.05 / 0.00 +/- 0.00 / 72.14 +/- 0.94 |

Global是首要指标。Ours在三个规模上均优于相同IPC10图像预算的Heuristic和FAST；5-agent的Expert
低于二者，必须作为权衡报告。Full Real是高通信oracle，只运行5-agent。DeSA-CIL*和
MASC-complete*为明确标注的适配协议；FedRE保留原始异构联邦训练核心和round-100报告点。它们的
native通信对象不同，因此不参与image-only预算下的方法差值计算。

全部19个聚合条目已进入`experiments/registry.yaml`白名单并标记complete/paper-eligible；
`scripts/build_project_results.py`从原始CSV/JSON重新聚合并生成`RESULTS.md`、
`paper_tables/iclr2027_scaling_results.*`、`paper_tables/iclr2027_scaling_communication.*`和完整
provenance JSON。内部静态配对条件仍为相同class/model split与seed、相同cosine expert来源、
3780-step schedule以及hard-label方法零logits。FAST固定HEAD commit
`6a218fcfdc93838634921399b0de6a36cdd29756`及已记录的NumPy兼容补丁SHA；Full Real保持独立
full-data comparability group。统一队列已以`48/48`完成，最终状态位于
`logs/iclr2027_baseline_matrix/queue_status.json`。

### 论文方法写作交接：定义、公式与代码映射

本节是给论文写作 agent 的方法交接入口。论文方法叙事固定为 **agent self-distillation + one-shot
DKP socialized learning**；代码、实验简化和证据边界必须同时遵守本节与前述协议。不得从历史归档
或单个诊断 run 反推另一套方法定义。

#### 方法的一句话定义

每个异构 agent 先把自己的私有类别知识自蒸馏成一个小型 Distilled Knowledge Packet；所有 agent
只进行一次逻辑 all-share，随后每个 receiver 从自己的本地 expert 出发，使用 prototype 初始化、
packet-heavy 双流训练、特征保持、sender-class KD 和监督对比学习，在不聚合参数、不训练统一全局
模型的条件下扩展到全局类别空间。

#### 问题设定与不变量

设有 `N` 个 agent、全局类别数 `C=100`。agent `i` 只持有类别集合 `C_i` 和私有数据 `D_i`；当前
5/10/20-agent平衡协议中 `|C_i|=q=C/N`，各集合两两互斥且并集为全局 `0-99`。不同 agent 的
backbone 可以不同，但所有 task model 都满足：

```text
classifier output dimension = C
persisted labels = global labels
parameter aggregation = none
shared global model = none
model homogeneity requirement = none
communication topology = one logical all-share
```

全局100维输出和全局标签是异构模型能够解释同一个packet的接口契约，不是要求它们共享参数空间。

#### 阶段A：agent自蒸馏与DKP构造

1. agent `i` 用 `D_i` 训练本地 cosine task expert `E_i`。训练损失只在 `C_i` 对应的输出列上计算，
   但checkpoint分类头始终为100维。
2. agent `i` 只使用自己的 `D_i` 在本地训练DSDM guide pool，并把本地数据自蒸馏为每类 `K` 张
   raw synthetic images；当前 `K=10`。guide只服务于图片生成，不进入通信包，也不等同于最终task
   expert `E_i`。
3. factor `f=2` 的确定性decode在接收端产生 `f^2=4` 个训练视图。通信图片量只按raw images计，
   decode不增加传输图片数。
4. 同一个最终expert checkpoint对decoded images生成sender logits，但只保留有序列集合 `C_i`，
   不保存完整100类logits。当前持久化dtype为`float16`。

因此逻辑packet定义为：

```text
P_i = (X_i_raw, y_i_global, Z_i_local, C_i, provenance_i)
```

其中 `provenance_i` 至少绑定sender id、backbone id、expert checkpoint SHA-256、raw/decoded tensor
摘要、logit逐行对齐摘要和协议版本。guide权重、receiver权重和原始私有训练集都不属于DKP。

**实验实现边界：** 方法定义使用sender-local agent self-distillation。当前5/10/20-agent扩展为了将
昂贵的图片生成与receiver验证解耦，复用了通过完整性门禁的backbone-specific full-class pool并按
`C_i`切片；用户已确认该工程解耦不影响当前receiver结论。论文可以按agent自蒸馏描述方法，但实验
协议或附录必须如实披露pool reuse，且不得把该slice结果冒充“重新执行了sender-local DSDM”的直接
实验证据。

#### 阶段B：一次通信与receiver初始化

每个sender只生成一个packet并向其他agent逻辑共享一次。receiver `j` 不下载其他模型，也不平均
参数，而是：

1. 用自己的最终expert `E_j` 初始化当前模型；同一checkpoint同时冻结为FR教师。
2. 对每个外部类别 `c`，用receiver自己的冻结encoder `phi_j`提取该类全部decoded packet images的
   penultimate features，并进行cosine prototype imprinting：

```text
p_j,c = normalize(mean_{x in P_external,c} phi_j(x))
W_j[c] = p_j,c
```

3. 本地类别 `C_j` 的classifier rows必须逐元素保持expert值；只有外部类别rows由prototype写入。
4. 全部prototype finite、单位范数并通过门禁后，才解除100类输出mask并开始社会化训练。

Prototype initialization解决新类分类头的冷启动，但当前没有完全隔离它的正式因果消融，因此论文
应将其写为稳定初始化机制，不单独声称它贡献了某个精确提升值。

#### 阶段C：packet-heavy双流社会化训练

当前多seed协议固定为r02 full。每个optimizer step同时组织本地知识和外部知识，但不同loss使用
严格匹配的视图：

```text
local real crop/flip view:
  2% local CE contribution + frozen/current penultimate feature cosine FR

receiver self-DKP view (本地已有，不计外部通信):
  98% local CE contribution

external deterministic decoded view:
  sender-class KD; teacher logits只约束其sender自己的有序类别列

two independent external crop/flip views:
  mean external CE + SupCon on normalized penultimate features
```

令 `rho=0.02`，则本地CE为：

```text
L_local = rho * CE(local real) + (1-rho) * CE(receiver self-DKP)
```

为使每个全局类别具有相同的期望CE权重，平衡划分下按类别数而不是按loader长度加权：

```text
L_CE = (q/C) * L_local + ((C-q)/C) * L_external
```

FR在同一local-real增强视图上比较冻结expert与当前模型的penultimate feature cosine distance；KD仅
在发送方类别子空间内用温度 `T=2` 蒸馏；SupCon不增加projection head。当前总损失固定为：

```text
L = L_CE + 0.2 * L_FR + 0.6 * L_KD + 0.1 * L_SupCon
```

所有5/10/20-agent receiver固定执行3780个optimizer steps，step milestones为2457/3213。训练过程
不会更新冻结expert，也不会产生可供其他agent继续聚合的新全局模型。

#### 通信量公式

平衡划分、IPC=`K`、factor=`f`、float16 logits下：

```text
raw images / sender              = Kq
unique raw sender payload        = KC
external raw images / receiver   = K(C-q)
decoded images / sender          = f^2 Kq
logit bytes / receiver           = 2 f^2 K q (C-q)
```

当前 `C=100, K=10, f=2`，因此唯一sender raw payload在所有agent规模下均为1000张；5/10/20-agent
的receiver外部raw images分别为800/900/950，sender-logit bytes分别为128,000/72,000/38,000。
图片数和logit bytes必须分开报告，不能把decode后的视图重复计为通信，也不能把外部baseline的参数
或表示通信换算成image-only预算。

#### 模块在什么位置起作用

以下是已完成seed0内部诊断支持的机制解释，不等价于独立多seed因果结论：

| 模块 | 主要作用位置 | 当前证据与允许表述 |
|---|---|---|
| 自蒸馏DKP | sender到receiver的新类知识载体 | 当前端到端结果支持高知识密度packet有效；由于full方法还包含KD，不能把相对hard-label baseline的全部差值归因于图片本身 |
| self-DKP主导的local CE | receiver域平衡 | 从local-real S切换到self-DKP U时Global约`+6.786`、New约`+12.1625`；这是当前最大的receiver机制信号，但同时降低Expert |
| `rho=0.02` real mix | 新旧类校准 | 相对U，Global仅约`-0.208`而Expert约`+6.450`；该点由seed0诊断提出，随后在seed1/2/3固定复核 |
| FR | 本地表示保持 | 固定组件矩阵中FR-only相对CE-only Global约`+1.846`，是最稳定的单组件信号 |
| SupCon | 外部类别结构 | SupCon-only约`+0.880` Global；FR+SupCon约`+2.356` |
| sender-class KD | 发送方细粒度决策迁移 | r02配对中KD-on相对KD-off为Global约`+1.132`、New约`+1.8925`、Expert约`-1.910`，主要帮助新类并存在旧类权衡 |
| cosine classifier | 跨类尺度与prototype兼容 | matched-linear诊断中平均Global约`+0.614`且有backbone例外；应写为稳定器而非主要增益来源 |
| prototype initialization | 外部分类头冷启动 | 初始化后New能力并不低，但尚无完全隔离的正式消融；不得虚构独立提升值 |

当前最稳妥的机制总结是：主要性能来自 **高知识密度DKP与packet-heavy域平衡receiver的配套**；FR
和SupCon负责稳定新旧表示，sender-class KD进一步提高New/Global但可能牺牲部分Expert，cosine和
prototype主要改善初始化与跨类校准。

#### 论文claim边界

1. 不得把系统相对Heuristic/FAST的全部优势写成“synthetic image单模块贡献”；完整DKP还包含logits，
   receiver也包含FR/SupCon/prototype/domain balance。
2. `Global`是首要指标，`New/Expert`是共同主指标；5-agent上可能出现New显著提升而Expert下降，必须
   如实写成plasticity-retention trade-off，不得声称所有指标支配。
3. Full Real是每类500张的高通信oracle，不属于IPC10同预算比较。
4. MASC、DeSA、FedRE具有不同的同构、参数、anchor/logit或表示通信条件；只在各自
   `comparability_group`内解释，不用image-only数字强行排序。
5. 当前研究不证明形式化隐私，不把“未发送原始数据”扩写成隐私保证；复杂通信拓扑也不是贡献。
6. seed0组件结果是内部机制诊断；正式稳定性结论必须来自seed1/2/3的seed-level mean与population
   std，并且一个seed必须覆盖该规模全部receiver。

#### 代码到方法的映射

| 方法职责 | 主要代码入口 |
|---|---|
| 配置与5/10/20-agent嵌套划分 | `scripts/prepare_iclr2027_dkp_scaling.py`, `agent_data.py`, `config_adapter.py` |
| 统一pipeline编排 | `run_social_pipeline.py` |
| 本地cosine expert训练与选择 | `agent_trainer.py`, `DSDM/models/cosine_classifier.py` |
| agent自蒸馏图片生成 | `DSDM/DSDM.py`, `DSDM/dsdm_decode.py` |
| 当前图片池解耦适配 | `fullclass_pool_packets.py`, `scripts/materialize_fullclass_pool_packets.py` |
| sender-class logits与逐行对齐 | `packet_logits.py`, `packet_integrity.py` |
| packet manifest与原子结果写入 | `social_output_manager.py`, `output_manager.py` |
| receiver packet消费与factor decode | `packet_consumer.py` |
| prototype、r02双流、FR/KD/SupCon和3780-step训练 | `social_trainer.py`, `dkp_receiver.py` |
| packet/receiver/final checkpoint门禁 | `validate_packets.py`, `scripts/validate_iclr2027_dkp_scaling.py`, `scripts/audit_iclr2027_dkp_final_receivers.py` |
| hard-label baseline完整审计 | `scripts/validate_iclr2027_hard_label_baseline.py` |
| DeSA/MASC/FedRE适配与通信统计 | `baseline_adapters/`, `baseline_adapters/communication_accounting.py` |
| 单GPU完整baseline队列 | `scripts/run_iclr2027_baseline_matrix.py` |

论文写作agent建议按“问题设定 -> agent self-distillation -> DKP定义 -> one-shot通信 -> receiver
prototype与双流优化 -> 通信复杂度 -> 异构性讨论”的顺序组织方法部分；实验结果、消融数字和正式表格
仍分别以registry生成的`RESULTS.md`和`paper_tables/`为准。

### 强制启动与结果门禁

启动 receiver 前必须检查实时 GPU/进程/日志，不能把归档 PID 当实时状态，并满足：当前规模全部
expert checkpoint 可 strict-load、输出100维且本地 accuracy合理；初始化/FR/sender-logit teacher
SHA-256一致；类别集合等长、互斥且并集为0-99；每个 packet 的 raw/decoded/logit shape 符合上表且
标签保持全局类；关闭KD的 image-only方法必须没有 logits；decoded image/label/logit 数量、类别、逐行
顺序摘要一致且全有限；prototype finite、单位范数且不覆盖
本地权重；packet validator、模型 output/get_feature 和相关单元测试全部通过。任何不一致都必须停止
并报告，禁止静默修补。

每个 receiver 分别记录 `Global/New/Expert`、各 loss 的训练均值、wall-clock训练时间、optimizer
steps、raw communication images和transmitted logit bytes，再计算相应 seed 内全部 receiver 的算术
均值。Local expert 的主要输出是本地 expert classes 的 `Expert`；若registry保留 Global/New字段，
必须标明它们是在
local mask 下的通信前参考值而非全局分类模型结果。新协议使用独立 comparability group 和新
`run_name`，不覆盖或混合历史4-agent结果。`RESULTS.md` 和 `paper_tables/` 仍只由白名单registry
生成，不手工写正式数字。

## 历史4-agent正式协议（保留，不再作为当前计划）

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

## 历史4-agent方法

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

## 历史4-agent通信协议

| IPC | Raw/sender | External raw/receiver | DSDM effective/receiver | Full Real 比例 |
|---:|---:|---:|---:|---:|
| 10 | 250 | 750 | 3,000 | 1/50 |
| 50 | 1,250 | 3,750 | 15,000 | 1/10 |
| Full Real | 12,500 | 37,500 | 37,500 | 1 |

DSDM `factor=2` 表示一张 raw image decode 为 4 张训练图；通信量始终按 raw image 统计。

## 历史4-agent最优超参数

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

## 历史4-agent Baseline 协议

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

## 历史4-agent正式结果

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

## 历史 PAT-style 类别划分扩展

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

## CIFAR-100 backbone-specific 全类 DSDM 图片池（seed0 completed）

该阶段用于验证并保存五种 backbone 各自的全类图片池，不运行 social communication 或 receiver。
每个独立 run 使用完整 CIFAR-100 训练集的 `0-99` 类，保留 global class index；这是一种用于
前期逻辑验证的可复用图片池协议，不等同于主实验中 sender 只见自己 expert classes 的本地
DSDM，后续论文必须明确该协议差异。

ConvNet-3 性能门禁已经完成：pure-200 教师池的最佳 synthetic evaluator accuracy 为
`46.54`（iteration 10,000），单轨迹多 checkpoint 教师池为 `45.47`（iteration 5,000）；两组
均完成 10,000 iterations 并通过 packet validator。pure-200 达到论文 `46.2 +/- 0.3` 的量级，
因此放行 ConvNet-4 和 AlexNet 的同协议双模式对照。

六个 full-class run 均完成 10,000 iterations 并通过 packet validator；每个 packet 为1000张
raw synthetic images（每类10张），factor=2 后为4000张训练视图：

| Backbone | Pure-200 | Trajectory-20x10 |
|---|---:|---:|
| ConvNet-3 | 46.54 @10000 | 45.47 @5000 |
| ConvNet-4 | 48.57 @9500 | 46.38 @9500 |
| AlexNet | 32.35 @1000 | 36.81 @6000 |
| ResNet-10-CIFAR standard | 19.92 @2000 | 不运行 |
| ResNet-18-CIFAR standard | 20.26 @4500 | 不运行 |

这些是各自 backbone evaluator 下的 full-100 accuracy，不应直接与20类切片 evaluator 数值横向
比较。ResNet standard 的图片质量明显低于 ConvNet，说明“容量足够”不等于当前官方 DSDM
guide recipe 已适配；后续需单独分析 ResNet 的教师收敛、BN/优化设置和特征匹配稳定性。

```text
Backbones: ConvNet-3, ConvNet-4, AlexNet, standard ResNet-10, standard ResNet-18
Agents: 1 per run, agent_0 classes = 0-99, classifier output = 100
Guide pool: 10 models, selected maturity = 200 epochs
DSDM: IPC=10, factor=2, 10,000 iterations
Evaluation checkpoints: every 500 iterations from 500 through 10,000
Run names: cifar100_fullclass_dsdm_<backbone>_dsdmguidee0200_ipc10_seed0
```

本轮以作者官方仓库 `https://github.com/Li-Hongcheng/DSDM` 的 commit
`cb12851831e39da6b0169da84598166ad7706e01` 为代码基准。启动前必须运行
`scripts/audit_official_dsdm.py`：它检查本地 DiffAug、ConvNet、训练/验证核心函数与官方源码，
并直接执行官方 CIFAR-100 IPC=10 `--reproduce` 参数解析，与本地 ConvNet-3 resolved args
逐字段比较。允许的训练调整只有 guide epoch `20 -> 200`；不同 backbone 的必要适配只有
架构定义和 `f_idx/idx_from = 2/3/7/5/5`。每 500 iterations 评估一次只改变测量和最优图片
候选时刻，不进入蒸馏梯度更新。所有 backbone 统一沿用官方 `lr_img=0.1`、DiffAug、CutMix、
SGD、`train` mode guide、无 guide freeze、无 scheduler、无 gradient clipping 和其余 DSDM
参数。官方原生 `--reproduce` 只指定 ConvNet-3；ConvNet-4、AlexNet 和标准 ResNet 结果必须
标注为保持官方训练/蒸馏 recipe 的架构适配实验，不能写成官方原生配置。

先前运行名不含 `dsdmguidee0200` 的全类池使用了项目 teacher-quality recipe（包括模型专属
`lr_img`、非官方 guide 训练路径和局部稀疏评估点），已停止并只保留为 protocol-mismatch
diagnostic，不得与本轮或 DSDM 论文的 `46.2` 横向比较。

历史 AlexNet formal sweep 记录的 CIFAR-100 IPC=10 最佳值为 `46.00 @ iteration 5000`，对应
`f_idx=7`、`lr_img=0.005`、semantic/MSE、factor=2 和 10 个独立 epoch-20 DSDM-style guides。
原始目录 `/root/autodl-tmp/outputs/dsdm_arch_recipe_cifar100_ipc10/` 及其最佳图片已经不在本机，
Git 与回收站也没有副本。`2026-07-26` 使用新运行名
`cifar100_fullclass_dsdm_alexnet_historicale0020_ipc10_seed0` 重建该 recipe；guide 在蒸馏时保持
历史 `eval` mode 和参数冻结，验证间隔改为每500 iterations，但保留历史峰值候选5000。
该运行是历史项目 recipe 的恢复实验，不标注为官方 DSDM 原生 AlexNet 配置。

该图片池只保存 synthetic images、global class indices、best-iteration manifest 和 packet
provenance，不在池构建阶段附加 sender logits；未来通信时按接收方需要的类别从对应
backbone 池索引图片。每个池的 `class_ids` 必须严格等于 `0-99`，每类 raw image 数为 10，
factor decode 后每类训练视图为 40。

ConvNet-3、ConvNet-4 和 AlexNet 均比较两种 guide-pool design。单轨迹组只训练一条连续
200-epoch 轨迹，在 epoch `20/40/60/80/100/120/140/160/180/200` 保存 10 个 checkpoint；
它们按该顺序映射为 DSDM 的 `model_0...model_9`。DSDM 本身不做多模型拼接或损失改写，
每次迭代仍按官方逻辑等概率随机选择一个 `model_i`。除 guide pool 从
“10 个独立初始化的 epoch-200 模型”变为“同一初始化轨迹的 10 个成熟度节点”外，IPC、
factor、10,000 iterations、每 500 iterations 评估以及全部 DSDM 超参数均保持一致。

```text
Pure-200: 10 trajectories x checkpoint at epoch 200
Trajectory-20x10: 1 trajectory x checkpoints every 20 epochs through epoch 200
Backbones using both designs: ConvNet-3 / ConvNet-4 / AlexNet
Trajectory config pattern: configs/fullclass_dsdm/fullclass_<backbone>_trajectorye0020to0200step0020_ipc10_seed0.yaml
```

该对照属于 teacher-pool composition ablation，不得写成官方 DSDM 原生设置。最终 epoch-200
checkpoint 不在本阶段承担 sender logits teacher 角色。

后续新实验中的 ResNet 固定使用 `resnet10_standard` 和 `resnet18_standard`；历史 compact
ResNet 结果继续保留，但不再作为新实验 backbone。这里的 standard 是 CIFAR 输入适配的标准宽度
ResNet：3x3 stride-1 stem、base width 64，R10 blocks=`[1,1,1,1]`、R18
blocks=`[2,2,2,2]`，100 类输出时参数量分别为 `4,949,412` 和 `11,220,132`。它们具有标准
通道宽度和残差层数，但不是带 7x7 stem/max-pool 的逐字 torchvision ImageNet 实现。

标准 ResNet 不运行单轨迹多断点教师池；本轮只使用充分预训练完成的独立 epoch-200 教师 checkpoint。
全类池需要教师见过 CIFAR-100 全部 `0-99` 类，不能复用只在 PAT5 单 sender 20 类上训练的旧
teacher-quality checkpoint。全类标准 ResNet 教师训练完成后固定保存；复用时同时记录源文件与
映射文件 SHA-256，并要求加载后的 state dict 逐张量完全相同，后续 DSDM
或通信不重复训练教师。标准 R10/R18 全类 run 已在 ConvNet-4/AlexNet 对照之后完成；运行名分别为
`cifar100_fullclass_dsdm_resnet10_standard_dsdmguidee0200_ipc10_seed0` 和
`cifar100_fullclass_dsdm_resnet18_standard_dsdmguidee0200_ipc10_seed0`；两者均已完成并通过
validator。

标准 R10 随后使用模型专用 recipe 重做：10 个独立 epoch-200、crop/flip + cosine guides，
DSDM 使用 `eval`/冻结 guide、`f_idx=5`、`lr_img=0.01` 和相同8个稀疏评估点。恢复运行
`cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery` 已完成
10,000 iterations，最佳 synthetic evaluator accuracy 为 `38.86 @ iteration 3000`；100 类
各10张 raw images、factor 解码后各40张训练视图均通过 validator。

为判断 ResNet 的 BN 中间分布是否是图片质量瓶颈，新增同 seed、同初始化、同 guide state、同
DSDM recipe 的单变量 PCBN 对照。权重不是沿用 Tiny-ImageNet 的 `10,000`：在 R10 混合初始化上，
用 guides `0/5` 与类别 `0/20/40/60/80/99` 共12个样本校准，原始 DSDM/归一化 PCBN 平均损失为
`30.2868/0.0025595`。最终使用 `weight=960`、全部12个 BN 层、按层归一化，使加权 PCBN 初始约占
总 loss 的 `7.504%`；`620/1300` 分别对应约 `5%/10%`，只作为量级参考，不运行额外权重 sweep。
配置为 `fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml`，新 run 不覆盖
纯 DSDM。该对照是 packet-quality diagnostic，必须等待完整10,000 iterations和 validator 后再
判断是否提高 ResNet 性能。

`2026-07-27` 对 PCBN 实现进行了失效门禁与梯度审计。正常计算路径保持不变：真实分支统计断开
梯度，合成分支统计保留对图片的梯度；每类分别在相同增强后的真实/合成 batch 上匹配，统计维度为
batch 与空间维；guide 固定为 `eval` 且参数冻结；每次更换 guide 均移除并重挂 hook。新增门禁会在
权重非有限、层名无法匹配、模型无目标 BN 或 hook 收集数异常时直接失败，不再静默返回零损失；
packet provenance 同时记录层选择、按层归一化和实际 hook 数。

R10 的 `w960` 在训练 `100-1000/1000-3000/3000-6000/6000+` 区间中，PCBN 占记录总 loss 的
平均比例约为 `6.39%/3.46%/2.71%/2.50%`。独立梯度诊断在 mix 初始化和 `w960` iteration-5000
最佳图片上，使用 guides `0/5` 与类别 `0/20/60/99`：`w960` 的 PCBN 梯度范数平均约为 DSDM
梯度的 `1.45%/1.55%`；`w1300` 为 `1.96%/2.10%`，`w2100` 为 `3.17%/3.40%`。梯度夹角余弦
为正但较小，未发现相互抵消或数值爆炸风险。全部12层的直接 loss 均值会被 stem 的
`layer0.bn1` 主导，因此本轮保持实现不变，只把层尺度重平衡作为后续独立消融，不能与静态权重
sweep 混在同一结论。

权重 sweep 固定同 seed、初始化、guide state、DSDM recipe和8个评估点，并行运行 `w1300/w2100`；
只有 `w2100` 比 `w960/w1300` 中的最高值至少高 `0.10` 个百分点时，才追加 `w3400` 探索上边界。
所有权重按各自完整10,000 iterations的最佳合成集准确率比较；该过程是 single-seed tuning，不能
直接进入正式 social-learning 主表。

ConvNet-3 另设全类池切片诊断：从两个100类最佳池中只提取 PAT5 agent 0 的20个全局类别，
与 PAT5 agent 0 的本地20类 packet 在同一 fresh ConvNet-3 evaluator 上重测。类别固定为
`[26,86,2,55,75,93,16,73,54,95,53,92,78,13,7,30,22,24,33,8]`，标签保持全局编号，
分类头仍输出100维；每个输入均为 raw IPC=10、factor=2，评估统一使用1500 epochs、
DSA + CutMix 和 seed0。主要受控比较是 full100 pure-200 slice 与 local20 pure-200；同时报告
full100 trajectory slice 和实际 PAT5 20-epoch packet 作为辅助口径。该诊断已在 ConvNet-4
两条 pipeline 均 exit 0 后完成，不进入正式主表。

## 全类图片池通信逻辑验证（image-only packet 已完成）

为避免每个通信实验重复运行 DSDM，新增受控的 full-100 pool slice 协议。它先按 sender 的
task backbone 选择对应全类图片池，再严格按该 sender 的 expert classes 提取 IPC=10 图片，生成
sender-local packet。该协议模拟“sender 已用自身 backbone 蒸馏好本地类别图片”，但实际来源在
构池阶段见过完整 CIFAR-100，因此只用于通信逻辑和 receiver 消融验证，不能冒充当前主方法的
sender-local DSDM 结果。

首个隔离验证沿用4 agents × 25个连续互斥类别，模型为 AlexNet / ConvNet-3 /
standard ResNet-18 / ConvNet-4。标准 ResNet 图片池不会映射给历史 compact ResNet；适配器要求
model id 精确匹配。池目录由
`configs/packet_pools/cifar100_fullclass_ipc10_seed0.yaml` 白名单指定，并在切片前验证 source
packet SHA-256、完整10,000 iterations、最佳快照一致性、有限值、100类覆盖和每类10张。

```text
Run: cifar100_4agent_25cls_fullclass_pool_seed0_ipc10
Protocol: backbone_specific_fullclass_pool_slice_v1
Raw packet: 4 senders x 25 classes x IPC10 = 1,000 images
Decoded training views: 4,000 (factor=2)
Labels: global 0-99; each raw class count=10; each decoded class count=40
Validation: complete 4-sender manifest, packet validator warnings=[]
```

当前只完成 image-only packet materialization 和 communication manifest。完整 Ours 仍需先训练
各 sender 的独立收敛 expert，用它同时初始化对应 receiver 并为本 sender 的25类 packet 附加
25维 expert-class logits；附加 logits 后必须重新 build communication 和运行 packet validator，
才可启动 receiver。全类 guide checkpoint 不得作为 sender logits teacher。

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
切片重评结果如下（同一 fresh ConvNet-3、同一20类测试集）：

| Packet | Accuracy |
|---|---:|
| Full100 pure-200 slice | 60.55 |
| Full100 trajectory slice | 61.65 |
| PAT5 local20 main, guide-20 | 57.50 |
| PAT5 local20 pure-200 | 60.15 |

受控 pure-200 差值为 `60.55 - 60.15 = +0.40` 个百分点；因此全类池切片在该20类上的图片
质量与直接20类蒸馏基本一致。trajectory slice 相对 PAT5 guide-20 为 `+4.15` 个百分点，
但这同时改变了 guide maturity，只作为辅助诊断。
