# EXPERIMENT_LOG.md

本文件只记录关键实验、协议变化和结论。原始 stdout/stderr 保存在 `logs/` 或
`external_baselines/logs/`，完整逐 run 指标由 `experiments/registry.yaml` 索引。

## 日志格式

```text
日期 / experiment_id / role / 目的
配置或相对基准修改
状态与核心结果
结论、caveat、原始产物路径
```

## 2026-06-25 | centralized-upper-bounds | diagnostic

- 修正旧训练 recipe 后，ConvNet-3-IN 达到 65.24，ResNet-10 达到 72.08，
  ResNetAP-10 达到 73.47。
- 结论：旧 ResNet 低上界主要来自训练 recipe，不是残差架构本身无能力。
- 产物：`outputs/cifar100_4agent_25cls_upper_bound/centralized_full/`。

## 2026-07-01 | conv-family-and-allconv | historical_evidence

- Conv-family IPC=10/50 和 all-ConvNet IPC=50 用于建立 DSDM 可行性、通信预算趋势和
  receiver 超参基础。
- 这些是重要中间证据，不能标记 invalid；但不再作为当前 one-ResNet 主表。
- 相关汇总保存在旧 experiment registry 和对应 run metrics。

## 2026-07-05 | hetero4arch | diagnostic

- 完成 ConvNet/VGG/AlexNet 跨架构适配和 feature-index 核对。
- 观察到 VGG/AlexNet stability 对 packet 来源和 receiver loss 高度敏感。
- 该阶段推动 no-VGG one-ResNet 主设定形成。

## 2026-07-07 | hetero6arch-convnet-packet | historical_evidence

- 将 ConvNet-family DSDM packet 复用到异构 receiver，验证 input-space packet 可跨架构消费。
- ResNet receiver 的 self-guided/ConvNet-guided 差异随 IPC 改变，不能声称统一 guide 恒优。
- 历史 ConvNet-4 packet 存在旧 feature-index 版本，正式消融需带 provenance。

## 2026-07-10--2026-07-17 | one-resnet-main | main

- 主设定固定为 ConvNet-3、ConvNet-4、AlexNet 和恰好一个 compact ResNet。
- 主方法固定为 sender task-backbone-guided DSDM + expert-class logits。
- IPC=10 三 seed 完成：Ours global 32.78、new 29.11、expert 43.81、forgetting 23.24。
- IPC=50 最终采用 seed0/2 两种子 paired 结果；seed1 后续经用户决定停止并排除。

## 2026-07-16 | fast-hard-label | baseline

- FAST 使用官方 pixels/per-class/minmax，每类严格 IPC 张真实图，无 logits。
- 三 seed 下 FAST 在 global/new 上低于随机 Heuristic；该现象保留，不按预期篡改。
- Receiver RNG 尚未显式按 config seed 初始化，后续严格 paired comparison 需要统一重跑。

## 2026-07-17 | masc-desa-complete | baseline

- 最新 MASC 使用 complete CC + RA + HFE reproduction；旧 MASC official-flow 数值被 supersede。
- DeSA 使用 owner-aware class-incremental adaptation；旧 comparable_fixed 数值保留历史但不进主表。
- MASC Full validation 已通过，说明低 IPC 结果不是缺少完整流程导致。

## 2026-07-17 | documentation-consolidation | operations

- `AGENTS.md` 重构为指示矩阵；`PROJECT_SPEC.md` 成为唯一研究事实来源。
- 建立人工 registry，将主结果、baseline、ablation、diagnostic、historical evidence 分离。
- 旧指示和历史 spec 已先做 SHA-256 一致的归档副本。
- 删除 checkpoint/packet 前仅生成候选清单，等待用户确认。

## 2026-07-17 | one-resnet-seed1-stop | decision

- 用户决定停止 seed1 IPC=50 队列，并以 seed0/2 作为 IPC=50 最终两种子结果。
- 终止整个进程组后 GPU 降为 0% 利用率；后续 receiver、MASC 和 DeSA 未启动。
- seed1 agent0 完成、agent1 约 17% 时停止；其 partial artifact 保留但不进入结果聚合。
- 为保持 paired comparison，IPC=50 的 Ours、Heuristic 和 FAST 均只聚合 seed0/2。

## 2026-07-22 | pat-class-split-seed0 | validation

- 目的：在不引入 FedRE 训练机制的前提下，验证当前 DSDM + sender logits 方法扩展到
  `5 agents x 20 classes` 和 `10 agents x 10 classes` 的表现。
- 数据：保持 CIFAR-100 官方 train/test，不合并、不做 75/25 重划分；seed0 随机排列全局
  类别 `0-99` 后等分，类别之间互斥且完整覆盖。
- 模型：ConvNet-3、ConvNet-4、AlexNet、compact ResNet-10、compact ResNet-18；PAT5 各一次，
  PAT10 按相同顺序各两次。
- 其余协议：IPC=10、factor=2、10 guides x 20 epochs、DSDM 10,000 iterations、sender
  expert-class logits，以及现有 60-epoch receiver recipe，均与 seed0 one-ResNet 基准一致。
- 配置：`configs/pat_class_split/main_cifar100_pat5agent_seed0_ipc10.yaml`、
  `configs/pat_class_split/main_cifar100_pat10agent_seed0_ipc10.yaml`。
- 启动：`2026-07-22T04:50:16Z`，两条流程并行共享 RTX 4090；launcher PID/PGID `103753`，
  PAT5/PAT10 Python PID 分别为 `103762`/`103761`。
- 日志：`logs/pat_class_split_seed0/`；正式结果只有在所有 receiver 完成并通过 packet/结果
  完整性检查后才可登记，目前状态为 running。
- 首次启动继承了 base config 的空 `evaluate_iterations`，运行时实际解析成每100轮验证一次，
  共100个候选；两条流程在 agent 0 约 `2300/10000` 时按用户要求停止，未形成正式 packet。
- 旧式尝试日志归档到 `logs/pat_class_split_seed0/legacy_100checkpoints_stopped_20260722/`；
  36.76 MiB partial packet/synthetic/visuals 按
  `docs/operations/storage/pat_sparse_restart_cleanup_20260722.tsv` 清除，所有150个 guides 和
  15个 expert checkpoints 完整保留。
- 稀疏重启：`2026-07-22T05:54:17Z`，固定验证点为
  `100/500/1000/2000/3000/5000/7500/10000`；launcher PID/PGID `114206`，PAT5/PAT10
  Python PID 分别为 `114215`/`114214`。两条运行日志均确认 `Evaluation checkpoints (8)`。
- PAT10 AlexNet agent 2 在约 iteration 4,056 出现 loss/gradient 数值爆炸，随后 NaN 空转；
  guide checkpoints 与此前 iteration 100 保存的 `64.9` 最优 packet 均为有限值。PAT10 于
  `2026-07-22T07:54:13Z` 单独停止（exit 143），PAT5 保持运行。agent 2 的失败产物和 agent 3
  的中断产物非破坏性归档到各自
  `diagnostics/stopped_for_numerical_fix_20260722_0754/`，guide/expert 均保留。
- 数值修复不降低 AlexNet 历史最优学习率：仍使用 `lr_img=0.005`，仅 PAT10 AlexNet 增加
  `grad_clip_norm=100`；所有 DSDM 增加 loss/gradient/image/prototype finite gate，完成后 packet
  记录 `condense_complete/completed_iterations/grad_clip_count/max_grad_norm`。agent 2 完整结果以
  `64.9` 为硬性能下限，未通过则停止、不得运行 agent 7 或下游通信。
- PAT10 恢复队列于 `2026-07-22T08:04:33Z` 启动，PGID `119568`。顺序为非 AlexNet
  `3/4/5/6/8/9`，之后 AlexNet `2/7`；仅蒸馏 packet，不自动启动 logits、communication 或
  receiver。日志为 `logs/pat_class_split_seed0/pat10agent_seed0_ipc10_numerical_fix.log`。
- PAT5 完整 pipeline 于 `2026-07-22T11:22:03Z` exit 0，5 个 packet、logits、communication
  manifest 和 5 个 receiver 结果均已生成。PAT10 recovery 的 agent 3 完成 10,000 次蒸馏后，
  因旧代码重复拼接 `agents/agent_3` 的完成检查路径 exit 1；实际主 packet 已保存。该路径错误
  已修正，并由 teacher-quality smoke 确认不再产生嵌套 agent 目录；PAT10 未自动继续。

## 2026-07-22 | teacher-quality-role-separation | diagnostic-smoke

- 为下一轮成熟度校准拆分两类 checkpoint：DSDM guide pool 保存独立训练轨迹的指定 epoch
  快照；正式 `expert_model.pt` 单独充分训练，只按 sender 本地训练集的分层 validation 选择
  epoch，官方 test 只作最终报告。
- ConvNet 正式 expert 复用已验证的 500-epoch DSDM-style DiffAug + CutMix recipe；AlexNet
  使用 500 epochs，标准 ResNet-10/18 使用 200 epochs。选择最佳 validation epoch 后，从头
  使用完整 sender 训练集重训相同 epoch 数，避免测试集选择和固定最大 epoch 过拟合。
- ConvNet-3 最小闭环 smoke 使用 1 个 guide、2 epochs、2 次 DSDM iteration 和 2-epoch
  evaluator，仅验证路径与 provenance，不构成性能结果。输出确认 raw synthetic images
  `200 x 3 x 32 x 32`、factor 解码 logits `800 x 20`、两者均有限、蒸馏完成标记存在、教师
  `test_used_for_selection=false`，且未再生成重复嵌套的 `agents/agent_0` 路径。
- 首轮 smoke 产物：`outputs/cifar100_teacher_quality_conv3_seed0_smoke/`、
  `outputs/cifar100_tq_conv3_guidee0002_seed0_ipc10_smoke/`；最终 recipe/provenance smoke 使用
  新 run name 后缀 `_smoke_v2`，不覆盖首轮产物。其中 2-epoch teacher/test accuracy
  与 packet agreement 只用于证明度量链工作，不得用于选择正式 recipe。
- 用户确认后，正式 seed0 teacher-quality queue 于 `2026-07-22T14:57:49Z` 启动，launcher
  PID/PGID `130762/130762`。ConvNet-3、ConvNet-4、AlexNet、standard ResNet-10 和 standard
  ResNet-18 五个 teacher/guide 训练进程并行；全部完成后，14 个 DSDM maturity candidate 默认
  每批 3 个并行，只执行 `distill_packets + attach_logits + quality summary`。
- 正式日志与状态表：`logs/teacher_quality_seed0/`；launcher：
  `scripts/run_teacher_quality_seed0_parallel.sh`。启动检查为 GPU 100% utilization、约 5.97 GiB
  显存、无其他实验进程，数据盘剩余约 7.0 GiB。
- 正式 queue 于 `2026-07-23T01:22:44Z` 完成，5 个 teacher 和 14 个 DSDM maturity candidate
  均 exit 0；`outputs/teacher_quality_seed0_summary/summary.json` 报告 `passed=true`、无缺失或
  结构失败。最终选择的 guide 均为 200 epoch：ConvNet-3 图片 `59.85`、ConvNet-4 `68.00`、
  AlexNet `65.25`、standard ResNet-10 `59.35`、standard ResNet-18 `60.45`。对应 teacher
  official-test report-only accuracy 为 `77.65/82.65/83.55/85.20/85.85`，packet-label
  agreement 为 `93.88%/97.62%/93.12%/99.75%/99.75%`；所有图片、logits、checkpoint provenance
  和 10,000 iteration completion gate 均通过。
- 该结果只证明 packet image/logit quality，不自动替换主实验或启动 receiver。queue 完成后数据盘
  剩余约 `2.7 GiB`；所有 checkpoint/packet 暂时保留，删除需要单独生成清单并获得确认。

## 2026-07-23 | cifar100-fullclass-dsdm-pool | running

- 用户确认启动五种 backbone 的 CIFAR-100 全类 DSDM 图片池：ConvNet-3、ConvNet-4、AlexNet、
  standard ResNet-10 和 standard ResNet-18；每个 run 为独立 `agent_0`，类别严格为全局 `0-99`，
  分类头输出 100 维。
- 该阶段只构建 backbone-specific synthetic image pool，不启动 logits、communication 或
  receiver；每类 IPC=10、factor=2、10,000 DSDM iterations，固定稀疏评估点
  `100/500/1000/2000/3000/5000/7500/10000`。packet 必须保留 `class_ids` 和 best-iteration
  manifest，后续通信按接收方所需类别索引图片。
- 全类 guide pool 使用每个 backbone 10 个 200-epoch guides。为避免误复用 PAT5 的 20 类 guide，
  新配置移除了 `source_root`，从完整 CIFAR-100 训练集重新训练；不附加 sender logits。
- 配置：`configs/fullclass_dsdm/`；运行名：`cifar100_fullclass_dsdm_<backbone>_ipc10_seed0`；
  队列：`scripts/run_fullclass_dsdm_queue.sh`，最多两路并行，launcher PID/PGID `165235/165235`。
- 启动时 GPU 利用率约 86%，显存约 2.3 GiB，数据盘剩余约 12 GiB。ConvNet-3/4 guide 训练已
  开始，首个 20-epoch 日志正常；最终结果待全部 packet 和 `validate_packets.py` 完成后再登记。
