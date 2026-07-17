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

## 当前主设定

```text
CIFAR-100 / 4 class-disjoint agents / one compact ResNet per seed
Sender task-backbone-guided DSDM images + sender expert-class logits
IPC=10 primary / IPC=50 extension
```

## 主要入口

```bash
python run_social_pipeline.py --config <config> --stage <stage> --packet-method <method>
python validate_packets.py --config <config> --packet-method <method>
python scripts/build_main_results_table.py
python scripts/build_cleanup_candidates.py
python scripts/validate_project_docs.py
```

运行新任务前先检查 GPU 和 `docs/operations/current_status.md`，不要覆盖正在运行的输出。
