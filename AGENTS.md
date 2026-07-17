# AGENTS.md

最后更新：`2026-07-17`

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
| D-01 | 数据 | MUST | CIFAR-100 使用 4 agents × 25 个互斥类别 | current configs |
| D-02 | 标签 | MUST | 标签保持全局 `0-99`，禁止映射为局部 `0-24` | packet validator |
| D-03 | 模型 | MUST | 所有 task backbone 分类头固定输出 100 维 | model tests |
| M-01 | 方法 | MUST | 当前主方法是 sender 自身 task backbone 指导的本地 DSDM | packet provenance |
| M-02 | 方法 | MUST | 每个 sender 只用自己的 expert data 训练 guide；不共享 guide 权重 | checkpoints/config |
| M-03 | Logits | MUST | 只保存 sender 的 25 个 expert-class logits | packet validator |
| M-04 | 对照 | MUST | 统一 ConvNet/local encoder 目前是消融候选，不是当前主方法 | registry role |
| C-01 | 通信 | MUST | DSDM 按 raw synthetic image 计量，factor decode 不增加通信图数 | packet validator |
| C-02 | 通信 | MUST | 同时记录 image count 与 logit bytes，不能把 MASC 参数交互写成 image-only | result provenance |
| R-01 | 结果 | MUST | 只有 `paper_eligible: true` 的 complete/明确标注 interim 记录可进入正式表 | registry |
| R-02 | 结果 | MUST | 正式 seed 必须包含 4 个 receiver；聚合先求每 seed 四 agent 均值 | result builder |
| R-03 | 结果 | MUST | 主表均值和标准差基于 seed-level mean，当前使用 population std | result builder |
| R-04 | 结果 | MUST | 不跨 `comparability_group` 计算方法差值 | registry |
| R-05 | 超参 | MUST | 同协议以 global 为第一选择指标；差值 <0.5 时兼顾 new 与 forgetting | registry selection |
| B-01 | Baseline | MUST | Heuristic/FAST 使用 hard labels、无 logits | configs/packets |
| B-02 | Baseline | MUST | MASC/DeSA 必须注明同构、参数通信、owner logits 等适配条件 | baseline registry |
| H-01 | 历史 | MUST | Conv-family/all-ConvNet 属于重要中间证据，不得标记 invalid | registry role |
| H-02 | 历史 | SHOULD | hetero4/hetero6 保留汇总、配置和 provenance；大 artifact 只列删除候选 | cleanup inventory |
| O-01 | 运行 | MUST | 活动队列结束前不得移动其 config、output、packet source 或脚本 | process/log check |
| O-02 | 存储 | MUST | 删除 checkpoint/packet 前必须生成清单并获得用户确认 | cleanup candidates |
| O-03 | Git | MUST | 不覆盖或回滚用户已有变更，不执行 destructive reset/checkout | git status |

## 当前主实验速查

```text
Dataset: CIFAR-100
Agents: 4
Expertise: 25 class-disjoint classes per agent
Heterogeneity: ConvNet-3 / ConvNet-4 / AlexNet / exactly one compact ResNet
Method: sender task-backbone-guided DSDM + sender expert-class logits
Budgets: IPC=10 and IPC=50
Primary metrics: global / new / expert / forgetting
```

当前模型分配、训练 recipe、结果和 caveat 只在 `PROJECT_SPEC.md` 与 `RESULTS.md` 维护。

## 代码与实验规则

1. 优先复用 `run_social_pipeline.py`，不大规模重构 DSDM。
2. 新实验使用新 `run_name`，不得覆盖已有结果。
3. 修改模型后验证参数量、输出维度和 `get_feature` 层。
4. 修改 `f_idx` 时同步验证 `idx_from/idx_to`。
5. 修改 packet 后运行 `validate_packets.py`。
6. 新协议不得与旧协议混合计算 seed mean。
7. 当前机器只有一张 RTX 4090；重蒸馏前先检查 GPU 任务。
8. 第三方 repo 保持在 `external_baselines/repos/`，适配优先写 wrapper。

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
  tests.test_atomic_artifact_writes -v

python scripts/build_main_results_table.py
python scripts/validate_project_docs.py
```
