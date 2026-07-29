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

## 2026-07-28 | iclr2027-dkp-r02-full-and-storage-handoff | complete_diagnostic / blocked_on_storage

- 完成新 run
  `cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2`：五个 receiver
  全部 `exit=0`，均值 `Global/New/Expert=39.632/34.290/61.000`。严格配对的 r02 KD-off
  均值为 `38.500/32.3975/62.910`，因此 KD-on full 差值为
  `+1.132/+1.8925/-1.910`。
- 固定协议为 cosine expert/prototype、`real_packet_mix`、real CE loss fraction `0.02`、
  FR `0.2`、KD `0.6`/temperature `2`、SupCon `0.1`/temperature `0.07`、3780 optimizer
  steps、step milestones `2457/3213`。每个 receiver 接收800张 external raw images和
  128,000 sender-logit bytes。
- 五个最终 receiver checkpoint 均通过 strict-load、finite、100维输出、get_feature、cosine
  classifier 和 summary SHA 复验。汇总与审计分别为
  `outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_v2.json`
  和
  `outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_final_checkpoint_audit_v2.json`。
- 用户确认蒸馏阶段通过 evaluator 选择知识价值最高的候选图片是本项目接受的数据集蒸馏流程，
  后续不因该选择方式暂停通信验证。固定复用现有五个 full-class IPC10 pool，不重新蒸馏，
  继续保留全部 source/best-snapshot/catalog SHA provenance。
- `2026-07-28 11:07 UTC` 实时检查无训练进程，RTX 4090 空闲。数据盘约50 GiB、仅余
  1.1 GiB，容量98%；inode正常。下一阶段 5-agent 多 seed、10-agent、20-agent IPC10 加上
  原子保存余量要求至少10 GiB可用，若同盘继续IPC20/50建议新增至少30 GiB。
- 当前不启动新训练，状态记为 `blocked_on_storage_expansion`。扩容恢复步骤和精确 provenance
  已写入 `docs/operations/current_status.md`；未删除、移动或覆盖任何 artifact。扩容后先完成
  5/10/20-agent 动态门禁和单元测试，再跑多 seed，之后才扩10/20-agent；外部 baseline 暂停。

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
- 该首轮队列随后因 protocol audit 停止。停止时 ConvNet-3/4 的 synthetic best 分别约为
  `31.29/41.14`，只作错误 recipe 的 diagnostic；其输出不删除、不覆盖，也不进入正式比较。
- 复查改以作者官方 GitHub `Li-Hongcheng/DSDM` commit
  `cb12851831e39da6b0169da84598166ad7706e01` 为唯一基准。官方 README 的真实入口是
  `--reproduce`；本地新增启动门禁会直接运行官方参数解析，并要求 CIFAR-100 IPC=10 ConvNet-3
  的所有行为字段完全一致。
- 修正后只允许 guide epoch `20 -> 200` 和 backbone 所需 `f_idx=2/3/7/5/5`；每 500 次完整
  evaluator 只改变选优频率。撤销旧模型专属 `lr_img`、scheduler、guide eval/freeze 和数值裁剪，
  五个 backbone 均使用官方 `lr_img=0.1`、DiffAug + CutMix、10 guides、factor=2、10,000 iterations。
- 新 run 使用 `_dsdmguidee0200_` 后缀，与错误 recipe 的旧输出物理隔离；启动前源码/参数 audit、
  `py_compile` 和 40 个单元测试均通过。
- 修正版队列于 `2026-07-23T11:59:27Z` 启动，launcher PID/PGID `178099/178099`；官方源码和
  resolved-args 门禁通过后 ConvNet-3/4 两路 guide 训练开始。启动检查 GPU utilization 为
  `100%`、显存约 `3.9 GiB`，日志确认 net-update augmentation 为 `color_crop`。
- 按用户要求，本轮两路并行启动在 guide 尚未产生完整 checkpoint 前停止，短暂日志归档到
  `logs/fullclass_dsdm_dsdmguidee0200_seed0_parallel_stopped_20260723_1201/`；没有 synthetic
  packet 或 guide snapshot 被登记。
- 单 ConvNet-3 验证队列于 `2026-07-23T12:01:33Z` 重启，launcher PID/PGID `178765/178765`，
  环境 `MODELS=conv3 MAX_PARALLEL=1`。只验证官方 CIFAR-100 IPC=10 口径能否接近论文结果，
  其余 backbone 暂不启动。

## 2026-07-23 | cifar100-conv3-single-trajectory-guide-pool | running

- 用户确认新增与 pure-200 严格受控的 guide-pool composition 对照：只训练一条 ConvNet-3
  连续轨迹至 200 epochs，每 20 epochs 保存一次，共形成 10 个 checkpoint；不训练 10 条
  不同长度的轨迹，也不混合多个 checkpoint 同时计算一次 DSDM loss。
- checkpoint epoch `20/40/60/80/100/120/140/160/180/200` 依次映射到 DSDM 的
  `model_0...model_9`，蒸馏循环保持官方每 iteration 等概率随机选择一个 guide 的逻辑。
  对照其余设置与 pure-200 一致：CIFAR-100 全 100 类、IPC=10、factor=2、10,000 iterations、
  `lr_img=0.1`、官方 DiffAug/CutMix 以及每 500 iterations 完整评估。
- 新增配置、启动器和 provenance/test 门禁；官方 commit audit、`py_compile`、shell syntax 和
  14 项相关单元测试均通过。该设置明确登记为 DSDM teacher-pool ablation，不冒充官方原生设置。
- 并行队列于 `2026-07-23T13:30:07Z` 启动，launcher PID/PGID `182280/182280`；pure-200
  `PGID 178765` 保持运行。启动后 GPU utilization 约 `97%`、显存约 `3.6 GiB`，数据盘剩余
  约 `11 GiB`。日志确认 `trajectories=1` 及完整快照列表，epoch-20 checkpoint 已实际落盘。
- 运行名：`cifar100_fullclass_dsdm_conv3_trajectorye0020to0200step0020_ipc10_seed0`；日志：
  `logs/fullclass_dsdm_conv3_trajectorye0020to0200step0020_seed0/`。最终结果需等待 packet 完成、
  best manifest 和 validator 全部通过后登记。

## 2026-07-24 | cifar100-conv4-alexnet-guide-pool-comparison | launch

- ConvNet-3 两组均完成并通过 validator：pure-200 为 `46.54 @ iter 10000`，单轨迹 checkpoint
  池为 `45.47 @ iter 5000`。教师训练分别约 53 分钟和 7 分钟，总 pipeline 分别约 7 小时20分
  和 6 小时35分。
- 基于该闭环，ConvNet-4 和 AlexNet 各运行 pure-200 与 Trajectory-20x10 两组，共四个独立
  run。两组之间仅 guide pool composition、run name 和 provenance 不同；DSDM 均保持
  IPC=10、factor=2、10,000 iterations、每500次评估、官方 `lr_img=0.1`、无 gradient clipping。
- ConvNet-4 使用 `f_idx/idx_from=3`，AlexNet 使用 `7`；单轨迹 checkpoint 固定为
  `20/40/60/80/100/120/140/160/180/200`。官方 DSDM commit audit、配置结构化差异检查、
  shell syntax、py_compile 和相关协议测试均已通过。
- 新队列：`scripts/run_fullclass_conv4_alexnet_comparison.sh`；独立日志目录：
  `logs/fullclass_dsdm_conv4_alexnet_pool_compare_seed0/`。单张 RTX 4090 最多同时运行两组，
  先比较 ConvNet-4 pair，再比较 AlexNet pair。
- 队列于 `2026-07-24T03:24:34Z` 启动，launcher PID/PGID `194128/194128`。首轮运行态确认
  pure 组为 `models=10, epochs=200`、trajectory 组为 `trajectories=1` 和完整 checkpoint 列表；
  GPU utilization `100%`、显存约 `4.7 GiB`，无其他实验进程。
- 用户确认后续新实验的 ResNet 统一改用 CIFAR standard-width R10/R18；compact 结果仅作历史
  证据。standard R10/R18 只采用充分预训练的独立 epoch-200 教师 checkpoint，不运行多断点
  teacher pool；PAT5 20类教师不得冒充全100类教师。
- 标准 ResNet 后继队列使用 `scripts/run_fullclass_standard_resnet_queue.sh`，等待 Conv-4/AlexNet
  launcher `194128` 退出后自动开始。R10/R18 两路并行，每路从完整 CIFAR-100 训练 10 个独立
  epoch-200 终点教师，随后执行 DSDM、manifest 和 packet validation；没有 trajectory 版本。
- 后继 launcher 于 `2026-07-24T03:30:22Z` 启动，PID/PGID `194963/194963`；状态表已记录
  `dependency pid_194128 waiting`。等待期间只有 shell/sleep 进程，不占用 GPU。

## 2026-07-24 | fullclass-conv3-slice-vs-pat5-local20 | queued-diagnostic

- 目的：判断 ConvNet-3 先蒸馏全100类再按需切出20类，是否能保留直接在相同20类上蒸馏的
  packet 质量。PAT5 对应 agent 0，20个类别和全局标签完全保持不变。
- 统一评估：fresh ConvNet-3、100维分类头、相同20类 CIFAR-100 test subset、raw IPC=10、
  factor=2、1500 epochs、DSA + CutMix、seed0；每份 packet 都重新训练 evaluator，不直接比较
  原来的100类与20类 manifest accuracy。
- 输入包括 full100 pure-200 slice、full100 trajectory slice、实际 PAT5 local20 20-epoch packet、
  teacher-quality local20 pure-200 packet。主要受控差值为 pure-200 对 pure-200。
- `scripts/compare_fullclass_slice_pat5_conv3.py --dry-run` 已通过：四份输入均为每类10张、全有限，
  类别顺序、factor/decode 和源文件 SHA-256 已固定；切片单元测试通过。
- 执行器 `scripts/run_fullclass_slice_pat5_conv3_after_conv4.sh` 只在 Conv-4 pure200 和 trajectory
  两条 pipeline 都以 exit 0 完成后启动验证，等待期间不占 GPU。
- 等待器于 `2026-07-24T03:48:36Z` 启动，PID/PGID `195996/195996`；当前状态为
  `dependency_wait`，尚未创建 summary 或启动 evaluator。

## 2026-07-25 | fullclass-pools-and-slice-diagnostic | complete

- Conv-4/AlexNet 队列于 `2026-07-24T22:39:13Z` 完成；pure/trajectory 两组均 exit 0，四个 packet
  validator 均无 warning。最佳结果：Conv-4 `48.57/46.38`，AlexNet `32.35/36.81`（pure/trajectory）。
- 标准 ResNet 后继队列于 `2026-07-25T18:42:44Z` 完成；R10 `19.92 @2000`，R18 `20.26 @4500`，
  packet validator 均通过。两者只运行 independent epoch-200 teacher pool，没有 trajectory 版本。
- ConvNet-3 full100 slice 诊断于 `2026-07-24T13:40:38Z` 完成。使用 PAT5 agent0 的20类、
  100维分类头、fresh ConvNet-3、1500 epochs、DSA+CutMix、seed0 统一重评：full100 pure-200
  slice `60.55`，full100 trajectory slice `61.65`，PAT5 local20 guide-20 `57.50`，PAT5
  local20 pure-200 `60.15`。主要受控差值 `+0.40` 个百分点；结果只作 diagnostic，不进正式主表。
- 当前 GPU 已空闲，数据盘剩余约 `6.3 GiB`；所有 full-class outputs、slice summary 和日志均保留，
  删除仍需单独生成清单并经用户确认。

## 2026-07-26 | fullclass-dsdm-model-specific-e0200 | running

- 目的：用此前验证过的模型专用 DSDM recipe 重做 CIFAR-100 全 100 类图片池，区分前一轮统一
  `lr_img=0.1` recipe 导致的性能下降。新 run 不覆盖旧 full-class 结果。
- AlexNet：`f_idx=7`、`lr_img=0.005`、guide `lr=0.01`/batch `128`/无增强/multistep `[100,150]`。
- 标准 ResNet-10/18：`f_idx=5`、`lr_img=0.01`、guide `lr=0.1`/batch `128`/增强/cosine。
- 公共设置：IPC=10、factor=2、init=mix、decode=single、match=semantic、metric=mse、
  10 个独立 guide、guide epoch=200、10,000 DSDM iterations；评估点为
  `100/500/1000/2000/3000/5000/7500/10000`。
- 新 run：`cifar100_fullclass_dsdm_alexnet_modelbest_e0200_ipc10_seed0`、
  `cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0`、
  `cifar100_fullclass_dsdm_resnet18_standard_modelbest_e0200_ipc10_seed0`。
- 配置生成器：`scripts/prepare_fullclass_dsdm_model_specific.py`；队列：
  `scripts/run_fullclass_dsdm_model_specific_queue.sh`；日志：
  `logs/fullclass_dsdm_model_specific_e0200_seed0/`。
- 队列已启动，两路并行运行 AlexNet 与 ResNet-10，ResNet-18 在前两路完成后接入；启动时
  GPU 利用率约 `90%`，磁盘剩余约 `6.3 GiB`。当前处于 guide 训练阶段，蒸馏 evaluator 尚未开始；
  首个 100/500 次评估完成后再判断是否继续长跑。

## 2026-07-26 | fullclass-alexnet-historical-e0020-recovery | running

- 全盘核验确认历史 `46.00 @ iter 5000` AlexNet formal-sweep 图片已不在原目录、当前 outputs、
  Git 或系统回收站。现存 full-100 AlexNet 图片池最高为 trajectory-e200 的 `36.81`，不能冒充。
- model-specific e200 AlexNet 分支在 `30.3 @ iter 500` 且 e200 guide 全类 test mean `60.95` 时停止；
  已有诊断 artifact 保留。用户随后明确只重跑 AlexNet，因此同队列的 ResNet-10 分支也已停止，
  排队中的 ResNet-18 未启动；此前已经完成的 standard ResNet 结果保持不变。
- 新运行 `cifar100_fullclass_dsdm_alexnet_historicale0020_ipc10_seed0` 恢复历史 recipe：10个独立
  DSDM-style e20 guides、`f_idx=7`、`lr_img=0.005`、semantic/MSE、factor=2、eval/frozen guide、
  10,000 iterations。验证每500次执行并包含历史最佳候选 iteration 5000。
- 配置生成器为 `scripts/prepare_fullclass_alexnet_historical_e0020.py`，执行器为
  `scripts/run_fullclass_alexnet_historical_e0020.sh`，日志目录为
  `logs/fullclass_alexnet_historicale0020_seed0/`；新 run 不覆盖任何已有结果。
- 用户澄清 ResNet 正常运行应继续。停止前 R10 model-specific 运行最佳为 `38.14 @ iter 100`，
  iteration约790时 loss稳定在9–12，无数值异常。由于该实现不保存 DSDM optimizer/iteration，
  不从最佳图片追加10000次，而是新建 `_recovery` run，复用原R10 guide checkpoint并从头蒸馏；
  R18保持在R10完成后启动。恢复执行器为
  `scripts/run_fullclass_resnet_model_specific_recovery.sh`。

## 2026-07-26 | cifar100-r10-pcbn-control | queued

- 标准 R10 model-specific recovery 已完成：`38.86 @ iter 3000`，完整运行10,000 iterations，
  packet 为100类各10张 raw images，factor=2 后各40张训练视图；build communication 和 packet
  validator 均 exit 0。纯组 run 为
  `cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery`。
- 新增 PCBN 单变量对照，保持 seed0、mix 初始化、10个独立 epoch-200 guides、`f_idx=5`、
  `lr_img=0.01`、semantic/MSE、factor=2、10,000 iterations 和8个稀疏评估点完全一致。
  教师源与纯组一致；源 checkpoint 和 DSDM 映射 checkpoint 的序列化 SHA-256 分别记录，加载后
  state dict 已逐张量验证完全相同。
- 权重校准使用 guides 0/5 和类别 0/20/40/60/80/99 共12个初始样本。DSDM 主损失均值
  `30.2868458`，按12层归一化的未加权 PCBN 均值 `0.00255954`；选择 `weight=960` 后加权
  PCBN 均值 `2.45716`，占初始总 loss `7.504%`。候选量级 `620/960/1300` 对应约
  `5%/7.5%/10%`，本轮只运行中间的保守权重，不进行事后最优 sweep。
- 配置：
  `configs/fullclass_dsdm/fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml`；
  校准证据：`experiments/diagnostics/cifar100_r10_pcbn_weight_calibration_seed0.json`；
  执行器：`scripts/run_cifar100_r10_pcbn_control.sh`；日志：
  `logs/cifar100_r10_pcbn_control_seed0/`。
- pair validator 已通过：标准 R10 参数量 `4,949,412`、输出100维、penultimate feature `[B,512]`、
  PCBN 挂接全部12个 BN 层。PCBN、ResNet variant、稀疏评估和 atomic artifact 共15项测试通过。
  队列在纯 R10 完成后建立，并等待 AlexNet historical launcher `284624` 退出后开始，以维持最多
  两路 GPU 任务；届时会与正在进行的 R18 队列并行。等待 launcher 于
  `2026-07-26T15:08:54Z` 启动，PID/PGID `301541/301541`；preflight JSON 已写入日志目录且
  `status=passed`，等待阶段不占用 GPU。

## 2026-07-27 | cifar100-r10-pcbn-implementation-audit-and-weight-sweep | queued

- PCBN 正常数值路径复核通过：真实统计使用 `no_grad`，合成统计保留梯度；BN 输入按
  `[batch, spatial]` 求每通道 mean/variance；每类使用同一次 DiffAug 后的真实/合成数据；guide
  为 eval/frozen；12个 BN hook 每次 guide iteration 结束均关闭。原实现未发现会使现有结果失真的
  主逻辑错误。
- 修复的是 silent-failure 风险，不改变正常 loss：非有限权重、未知层名、零 BN hook、收集数量异常
  现在显式报错；best synthetic 与 packet provenance 增加 `pcbn_layers`、
  `pcbn_normalize_layers`、`pcbn_hook_count`。PCBN/atomic/evaluation/ResNet 共19项测试通过。
- `w960` 训练后半段 PCBN/total loss 约 `2.5%`。两种图片状态、2个guide、4个类别的梯度诊断显示，
  PCBN/DSDM gradient norm：`w960=1.45-1.55%`、`w1300=1.96-2.10%`、
  `w2100=3.17-3.40%`；梯度 cosine 为正。结果保存于
  `experiments/diagnostics/cifar100_r10_pcbn_gradient_diagnostic_seed0.json`。
- 全层直接均值中 `layer0.bn1` 在抽样中始终是最大项，mix初始时平均占单样本 PCBN loss 最大层比例
  约79%，iteration-5000图片约58%。本轮不修改层间归一化；层尺度重平衡另作消融，避免与权重
  sweep 混淆。
- 权重搜索固定其它条件，只并行运行 `1300/2100`；`2100` 至少比 `960/1300` 最优高0.10点时
  自动追加 `3400`，否则停止。等待器 PID/PGID `327701/327701`，先等待当前 `301541/285660`
  两条队列完成，再启动两路；等待阶段不占 GPU。
- 配置位于 `configs/fullclass_dsdm/*pcbn_w1300.yaml`、`*pcbn_w2100.yaml`、
  `*pcbn_w3400.yaml`；执行器 `scripts/run_cifar100_r10_pcbn_weight_sweep.sh`；日志
  `logs/cifar100_r10_pcbn_weight_sweep_seed0/`。三个候选结构门禁均通过，`w3400` 默认不运行。

## 2026-07-27 | storage-cleanup-inventory | awaiting-confirmation

- 数据盘约50 GB，已用48 GB，剩余约3.0 GB；仓库约49.6 GB，其中 `outputs/` 约42.8 GB，
  `external_baselines/` 约5.30 GB。
- 最安全的大项为完成的 FedRE 实验模型目录：task1三种子约4.08 GB、task3三种子约0.24 GB，
  合计磁盘块 `4,319,723,520` bytes；每个seed的 status/summary/protocol和CSV结果均保留。
  已生成 `docs/operations/storage/fedre_model_cleanup_candidates_20260727.tsv`，未执行删除，等待用户确认。
- 仓库外另有两个旧工程副本约0.99 GB；需用户确认是否仍有用途。历史 Tiny all200 与旧全类错误
  recipe 的可重训 guide/dsdm_pretrained checkpoint 约4.23 GB，列为C档，只在当前ResNet队列结束并
  重新核对 provenance 后考虑；活动R10-PCBN/R18及当前有效全类图片池完全排除。
- 分级总表：`docs/operations/storage/storage_cleanup_plan_20260727.tsv`。通用 tuning cleanup builder
  只找到约10 MB，不是本次磁盘压力主因。
- 用户确认后执行保守A0子集：只删除完成的teacher-quality、被替代full-class诊断、cross-arch和
  no-VGG诊断中的220个 `dsdm_pretrained/*.pth` 可重建缓存；原始guide、当前图片池、活动队列、
  packet、指标与配置均保留。释放约3.37 GiB，执行清单已标记为 `A0-executed`。

## 2026-07-27 | fullclass-pool-communication-gate | image-packets-complete

- 五个候选全类池逐项核验通过：packet 与 `synthetic/data_best.pt` 张量完全一致，best history
  存在，condensation 完整10,000 iterations，图片有限，标签为全局0-99且每类raw IPC=10。
- 新增 `backbone_specific_fullclass_pool_slice_v1` 适配器及池白名单。它要求 task model id 与
  pool model id 精确匹配，拒绝把 standard ResNet 图片池静默用于 compact ResNet。
- 首个隔离 run 为 `cifar100_4agent_25cls_fullclass_pool_seed0_ipc10`：AlexNet/ConvNet-3/
  standard R18/ConvNet-4 分别提取0-24/25-49/50-74/75-99，形成4个sender packet。
- `build_communication` 和 image-only packet validator 均 exit 0：1000张raw、4000张decoded，
  100类的raw/decoded计数严格为10/40，manifest包含4个唯一sender。
- 当前 `total_sender_logit_bytes=0`，不启动 receiver。PCBN sweep 结束后使用 expert-only 入口训练
  独立收敛本地expert，再执行 attach logits、重建manifest、validator和receiver训练。

## 2026-07-28 | iclr2027-dkp-ipc10-scaling-seed1-3 | running

- 预注册独立复核种子固定为 `1/2/3`，agent规模固定为 `5x20 / 10x10 / 20x5`。三种规模使用
  nested class split 和对应 macro-agent backbone；r02 full、3780 optimizer steps、IPC10、
  factor2、cosine head、prototype初始化及FR/KD/SupCon权重全部冻结，不允许按途中test结果修改。
- 生成45份互不覆盖配置：每个 agent规模/seed 各有fresh expert、DKP-SL full、Heuristic、FAST和
  Full Real。重复生成全部为 `reused_identical`。5/10/20-agent CE local/external权重分别为
  `0.20/0.80`、`0.10/0.90`、`0.05/0.95`。
- 动态 strict packet、expert reuse、receiver prototype和最终checkpoint审计已实现。DeSA-CIL的
  iterative owner-logit通信量及Full Real单sender图片数已从历史4x25硬编码改为按实际split计算；
  Heuristic/FAST继续保持hard labels且不带logits。
- 启动前全量 `176/176` tests通过；py_compile、项目文档冲突检查、registry结果表重建、
  `git diff --check`、45配置幂等性及pool catalog SHA检查均通过。数据盘已扩容至100 GiB，启动时
  可用52 GiB。
- 主队列于 `2026-07-28T12:26:27Z` 启动，PID/PGID=`5370/5370`，最大并发5。首任务为
  `5agent_seed1` 的五个fresh cosine experts；首个稳定GPU采样为利用率100%、显存5,838 MiB、
  功耗304.83 W。状态与日志位于 `logs/iclr2027_dkp_scaling/`。
- 队列按5-agent三个seed、10-agent三个seed、20-agent三个seed顺序fail closed。九个主方法任务
  全部完成前不启动MASC/DeSA/FedRE或其它外部baseline，也不启动IPC20/IPC50。

## 2026-07-28 | iclr2027-5agent-seed1-heuristic-sanity | complete-interim

- 首次 `5agent_seed1` 主方法训练本身全部成功，旧严格汇总器因把 canonical
  `receiver_agent_<id>` 路径误写成 `receiver_<id>` 而在训练后报错。修复汇总器与queue resume
  检查后，38项相关测试和177项全量测试全部通过；五个已有receiver均经只读完整性检查复用，未重训。
  严格汇总写入
  `outputs/cifar100_5agent20cls_dkp_r02_full_steps3780_ipc10_seed1_v1/metrics/scaling_seed_summary.json`，
  DKP-SL seed1均值为 Global/New/Expert=`38.810/33.1475/61.460`。
- 用户要求在继续主网格前先测随机 Heuristic。主队列在 `5agent_seed2` fresh expert 启动约20秒后
  按完整进程组安全终止；未生成或登记 seed2 结果，后续仍须由 `--resume` 完整性门禁恢复。
- Heuristic 使用同一 seed1 类别划分、五种 backbone、fresh cosine expert SHA、r02 local mix、
  FR=0.2、SupCon=0.1、3780 optimizer steps和相同IPC10 raw image budget；仅 packet 改为每类随机
  10张 sender-local真实图、hard labels、KD关闭且无 logits。五个packet均可由 seed1随机抽样精确
  张量复现，validator记录1000张raw/train images、每类10张、logit bytes=0且无warning。
- 五个receiver并行运行12分21秒至12分40秒，退出码均为0。Heuristic的五agent均值为
  Global/New/Expert=`33.660/25.9925/64.330`，forgetting=`18.690`；DKP-SL相对差值为
  `+5.150/+7.1550/-2.870`，forgetting差值为`+2.870`。Global/New为4/5 receiver胜出，Expert为
  2/5；因此按预注册Global首要口径 sanity gate通过，但不声称全指标支配。
- 五个最终checkpoint均通过严格state-dict加载、finite检查、100维cosine head、config/packet/
  checkpoint/provenance/expert SHA检查、loss与metric恒等式检查；从checkpoint独立重评五个模型与CSV
  的Global/New/Expert逐项完全一致。诊断证据：
  `outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed1_heuristic_sanity_v1.json`。
- 这是相同图片预算而非相同总字节比较：DKP-SL每receiver另传128,000 bytes sender logits，
  Heuristic按baseline定义不传logits。该单seed结果保持 `formal_result=false`、
  `paper_eligible=false`，不进入正式RESULTS表。

## 2026-07-28 | iclr2027-scaling-matrix-scope-freeze | validated

- 用户确认不运行所有方法与所有agent规模的机械笛卡尔积。最终内部执行范围冻结为：Ours与
  Heuristic各运行5/10/20-agent × seeds1/2/3，FAST与Full Real只运行5-agent × seeds1/2/3，
  共24个method-scale-seed单元。10/20-agent FAST与Full Real配置保留但不调度。
- 对24个选中配置执行只读静态配对审计并通过：相同class/model split与runtime seed、正确fresh
  cosine expert来源、3780固定步数、FR/KD/SupCon开关、hard-label方法零logits、FAST官方
  pixels/per-class/minmax commit=`6a218fcfdc93838634921399b0de6a36cdd29756`和算法seed0、Full Real
  每类500张及独立full-data comparability group均符合协议。
- 审计时Ours队列仍健康运行于`5agent_seed2` experts，RTX4090利用率100%；没有Heuristic、FAST、
  Full Real、MASC、DeSA或FedRE进程。活动主队列结束前不修改其config/output/packet source或脚本。

## 2026-07-29 | iclr2027-complete-baseline-matrix | ready-to-launch

- 本条记录替代上一条24单元的旧执行范围。用户最终确认FAST也必须运行多agent；完整研究矩阵冻结为
  57个method-scale-seed单元：已完成Ours 9个，baseline队列48个。baseline包括Heuristic 9、
  FAST 9、Full Real 3、DeSA-CIL 9、同构MASC-complete 9和FedRE 9；Heuristic 5-agent seed1已完成，
  因此实际尚需训练47个单元。
- FedRE保留官方训练核心、RAP、共享head和100轮：5-agent严格取官方HtM10模型列表前5个，10-agent
  使用完整10个，20-agent按原顺序重复该10模型列表两次；只将数据切分替换成项目seed1/2/3的全局
  标签互斥5/10/20-agent CIFAR-100 split。DeSA使用当前异构task backbone；MASC使用同构Netwider13，
  IPC10总计1000张CC真实图并单列参数交互。
- Heuristic/FAST均为sender级multi-agent hard-label协议、无logits。FAST固定官方HEAD
  `6a218fcfdc93838634921399b0de6a36cdd29756`，并把实际NumPy兼容补丁SHA
  `73dc25a3daf930a9aa3f1fdbbb5a994983e275d5dff6bbe10a7330ef70ac900c`写入packet provenance和cache key。
- 外部适配的5-agent与20-agent短smoke均成功；FedRE 5-agent官方训练路径及数据manifest验证成功。
  启动前182项全量测试、py_compile、45份配置验证、48任务dry-run、项目文档检查和
  `git diff --check`全部通过。统一单GPU fail-closed队列为
  `scripts/run_iclr2027_baseline_matrix.py`，最大并发receiver数为5。
- 队列于`2026-07-29T02:56:29Z`启动，master PID/PGID=`105288/105288`。它首先只读复核并登记已完成
  的Heuristic 5-agent seed1，随后为seed2生成五个hard-label、零logit packet，通过communication、
  packet validator和完整preflight后启动五个receiver（PID `105348-105352`）。连续五次GPU采样均为
  utilization 99%、显存7,648 MiB、功耗207-213 W，未发现Traceback、OOM或non-finite日志。

## 2026-07-29 | method-handoff-and-code-sync | validated

- `PROJECT_SPEC.md`新增唯一的论文方法写作交接章节，固定“agent self-distillation + one-shot DKP
  socialized learning”叙事，并记录问题定义、DKP内容、prototype初始化、r02双流损失、通信量公式、
  模块证据、claim边界及代码入口映射。README只增加导航，不建立第二份研究事实。
- 交接明确区分方法定义与实验工程解耦：方法按sender-local自蒸馏描述；当前扩展复用经过门禁的
  backbone-specific full-class pool slice，实验协议必须披露，不能冒充重新运行的sender-local DSDM。
- 代码同步前审计未发现待提交checkpoint、packet、数据压缩包或密钥模式；`outputs/`、`logs/`、
  数据集和模型权重继续由`.gitignore`排除。活动baseline队列未停止或修改，提交快照时为`16/48`
  完成，20-agent Heuristic seed2运行中。
