# AGENTS.md

最后更新：`2026-07-27`

本文件是当前仓库唯一有效的 Codex 指示文件。研究定义、实验协议和正式结果口径以
`PROJECT_SPEC.md` 为准；历史指示位于 `docs/archive/`，不得覆盖当前口径。

## 阅读顺序

```text
README.md
AGENTS.md
PROJECT_SPEC.md
RESULTS.md
EXPERIMENT_LOG.md（仅在追溯历史时读取）
```

## 指示矩阵

| ID | 范围 | 级别 | 指示 | 验证来源 |
|---|---|---|---|---|
| A-01 | 文档 | MUST | 当前研究事实只写入 `PROJECT_SPEC.md` | 文档冲突检查 |
| A-02 | 文档 | MUST | `RESULTS.md` 与 `paper_tables/` 由白名单 registry 生成 | `experiments/registry.yaml` |
| D-01 | 数据 | MUST | 当前 CIFAR-100 扩展标准为5/10/20 agents；首轮只运行 seed0 的5 agents × 20个互斥类别 | current configs |
| D-02 | 标签 | MUST | 标签保持全局 `0-99`，禁止持久化映射为局部标签 | packet validator |
| D-03 | 模型 | MUST | 所有 task backbone 分类头固定输出 100 维 | model tests |
| M-01 | 方法 | MUST | 当前计划方法为 DKP/DKP-SL；不聚合参数、不训练统一全局模型、不要求架构同构 | `PROJECT_SPEC.md` |
| M-02 | 方法 | MUST | DSDM guide只生成图片；同一个最终expert checkpoint同时承担receiver初始化、FR教师和sender-logit教师 | checkpoint SHA provenance |
| M-03 | Logits | MUST | 首轮只保存 sender 自身20个expert-class logits，不保存完整100类logits | packet validator |
| M-04 | 模型 | MUST | 当前新设计从expert阶段使用统一cosine classifier和可学习正全局scale | model tests |
| M-05 | 方法 | MUST | full-class pool slice 只用于通信逻辑/receiver 诊断，不得冒充 sender-local DSDM 主结果 | `PROJECT_SPEC.md` |
| M-06 | 模型 | MUST | pool source model id 必须与 sender task backbone 精确匹配，standard/compact ResNet 禁止混用 | pool adapter |
| C-01 | 通信 | MUST | DSDM 按 raw synthetic image 计量，factor decode 不增加通信图数 | packet validator |
| C-02 | 通信 | MUST | 同时记录 image count 与 logit bytes，不能把 MASC 参数交互写成 image-only | result provenance |
| C-03 | 通信 | MUST | pool slice 保持全局标签并记录 source packet SHA-256、best snapshot 和 pool protocol；不传播 guide 权重 | pool provenance |
| C-04 | 通信 | MUST | `use_sender_logits=true` 时，所有 sender logits 完成并重建 manifest、validator 通过后才能启动 receiver | communication gate |
| R-01 | 结果 | MUST | 只有 `paper_eligible: true` 的 complete/明确标注 interim 记录可进入正式表 | registry |
| R-02 | 结果 | MUST | 当前协议一个seed必须包含5个receiver；聚合先求seed内五agent均值；历史4-agent结果保持原聚合 | result builder |
| R-03 | 结果 | MUST | 主表均值和标准差基于 seed-level mean，当前使用 population std | result builder |
| R-04 | 结果 | MUST | 不跨 `comparability_group` 计算方法差值 | registry |
| R-05 | 超参 | MUST | 当前协议以Global为首要指标，New/Expert为共同主指标；forgetting只作内部诊断；禁止test-set调参 | registry selection |
| B-01 | Baseline | MUST | Heuristic/FAST 使用 hard labels、无 logits | configs/packets |
| B-02 | Baseline | MUST | MASC/DeSA 必须注明同构、参数通信、owner logits 等适配条件 | baseline registry |
| H-01 | 历史 | MUST | Conv-family/all-ConvNet 属于重要中间证据，不得标记 invalid | registry role |
| H-02 | 历史 | SHOULD | hetero4/hetero6 保留汇总、配置和 provenance；大 artifact 只列删除候选 | cleanup inventory |
| O-01 | 运行 | MUST | 活动队列结束前不得移动其 config、output、packet source 或脚本 | process/log check |
| O-02 | 存储 | MUST | 删除 checkpoint/packet 前必须生成清单并获得用户确认 | cleanup candidates |
| O-03 | Git | MUST | 不覆盖或回滚用户已有变更，不执行 destructive reset/checkout | git status |
| O-04 | 运行 | MUST | pool source、当前图片池和活动队列产物保持 immutable；切片 packet 使用新 run_name，不覆盖 source | pool adapter |

## 当前主实验速查

```text
Dataset: CIFAR-100
Current gate: seed0, 5 agents x 20 class-disjoint classes, IPC=10
Planned expansion standards: 10 agents x 10 classes; 20 agents x 5 classes
Heterogeneity: ConvNet-3 / ConvNet-4 / AlexNet / standard ResNet-10 / standard ResNet-18
Method: cosine expert + prototype initialization + dual-stream DKP-SL
Primary metrics: Global / New / Expert (Global first)
Diagnostic only: forgetting
```

历史4 agents × 25 classes正式结果继续保留，但不再作为当前计划。当前模型分配、训练 recipe、
结果和 caveat 只在 `PROJECT_SPEC.md` 与 `RESULTS.md` 维护。

## Full-class pool 通信验证速查

```text
Protocol: backbone_specific_fullclass_pool_slice_v2_dkp
Role: communication/receiver diagnostic, separate comparability_group
Source: validated full-class image pool, sliced by sender expert classes
Current gate: seed0 cosine experts, five sender packets/logits, CE-only and full receivers complete
Evidence: outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/first_round_seed0_summary.json
```

该验证协议不能与主实验 seed mean 混合。任何 standard ResNet pool 都不能写入 compact ResNet
sender；任何 image-only manifest 都不能在 logits-enabled 配置下直接进入 receiver。

## 代码与实验规则

1. 优先复用 `run_social_pipeline.py`，不大规模重构 DSDM。
2. 新实验使用新 `run_name`，不得覆盖已有结果。
3. 修改模型后验证参数量、输出维度和 `get_feature` 层。
4. 修改 `f_idx` 时同步验证 `idx_from/idx_to`。
5. 修改 packet 后运行 `validate_packets.py`。
6. 新协议不得与旧协议混合计算 seed mean。
7. 当前机器只有一张 RTX 4090；重蒸馏前先检查 GPU 任务。
8. 第三方 repo 保持在 `external_baselines/repos/`，适配优先写 wrapper。
9. 复用全类图片池时使用 `fullclass_pool_packets.py`/对应 wrapper，不修改 DSDM 蒸馏核心。
10. pool slice 后运行 `validate_packets.py`；附加 sender logits 后必须重新 build communication 和再次验证。
11. 当前五个完整IPC10 full-class pool优先复用；完整性门禁失败前不得重新蒸馏。
12. receiver启动前必须验证五个expert/packet/logit/prototype及其SHA provenance；不一致直接停止。
13. seed0首轮Local cosine expert、DKP images + CE-only和完整DKP-SL已完成；后续固定组件/分类头诊断必须使用新run_name，保持`paper_eligible: false`，不得覆盖首轮artifact。
14. 在首轮诊断审计和固定内部消融闭合前，不启动外部baseline或完整IPC/seed/agent网格。

## 检查命令

```bash
ps -eo pid,ppid,stat,etime,cmd \
  | rg 'run_one_resnet|run_social_pipeline|iterative_desa|run_masc' \
  | rg -v 'rg '

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

/root/miniconda3/envs/sp/bin/python -m py_compile \
  run_social_pipeline.py agent_data.py config_adapter.py DSDM/DSDM.py

/root/miniconda3/envs/sp/bin/python -m unittest \
  tests.test_dsdm_evaluation_schedule \
  tests.test_resnet_variants \
  tests.test_atomic_artifact_writes \
  tests.test_fullclass_pool_packets -v

OMP_NUM_THREADS=1 /root/miniconda3/envs/sp/bin/python \
  scripts/materialize_fullclass_pool_packets.py \
  --config <pool-communication-config> \
  --pool-catalog <fullclass-pool-catalog> --dry-run

python scripts/build_main_results_table.py
python scripts/validate_project_docs.py
```
