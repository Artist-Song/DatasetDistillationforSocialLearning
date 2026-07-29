# DatasetDistillationforSocialLearning

面向架构异构社会化学习的高知识密度蒸馏通信包研究代码库。

## 文档入口

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | Codex 操作约束和指示矩阵 |
| `PROJECT_SPEC.md` | 当前研究定义、方法、协议、超参数和正式口径 |
| `RESULTS.md` | registry 生成的正式结果、中间证据和 provenance |
| `EXPERIMENT_LOG.md` | 追加式实验与决策日志 |
| `experiments/registry.yaml` | 实验人工白名单和分类 |
| `docs/operations/current_status.md` | 当前后台队列和不可移动路径 |

历史文档位于 `docs/archive/`，不得作为当前方法或结果来源。

## 协议入口

当前计划、正在验证的协议、历史4-agent正式协议和结果可比性边界均只以
`PROJECT_SPEC.md` 为准；README 不复制实验口径，避免与研究规范产生两个事实来源。
论文方法写作请直接从 `PROJECT_SPEC.md` 的“论文方法写作交接：定义、公式与代码映射”一节开始，
该节同时给出agent自蒸馏叙事、DKP-SL数学定义、实现入口、模块证据和claim边界。

## 主要入口

```bash
python run_social_pipeline.py --config <config> --stage <stage> --packet-method <method>
python validate_packets.py --config <config> --packet-method <method>
python scripts/build_main_results_table.py
python scripts/build_cleanup_candidates.py
python scripts/validate_project_docs.py
```

运行新任务前先检查 GPU 和 `docs/operations/current_status.md`，不要覆盖正在运行的输出。
