# Operations

本目录保存会变化的运行状态、存储审计和删除候选，不承载论文结论。

| File | Purpose |
|---|---|
| `current_status.md` | 当前后台队列、活动配置和不可移动路径 |
| `cleanup_candidates_*.md` | 人工复核用清理摘要，不代表已删除 |
| `deletion_candidates_*.csv` | 输出目录级候选清单 |
| `config_candidates_*.csv` | 与候选输出对应的旧配置清单 |
| `deleted_configs_*.csv` | 已按批准策略删除的非最优配置路径记录 |
| `consolidation_changes_*.md` | 文档整合中归档、删除和保留事项记录 |
| `storage/` | 历史磁盘审计与 hard-link 去重记录 |

重新生成审计：

```bash
python scripts/build_cleanup_candidates.py
```

删除 checkpoint、packet 或输出目录前，必须核对 hard-link/source 依赖并再次取得用户确认。
