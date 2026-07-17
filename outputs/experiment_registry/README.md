# Legacy Discovery Registry

本目录是旧版全量扫描结果，包含 main、tuning、diagnostic、legacy 和重复 run。

它只用于历史检索，**不是论文结果白名单**。当前唯一有效的人工 registry 是：

```text
experiments/registry.yaml
```

当前正式结果由以下命令生成：

```bash
python scripts/build_main_results_table.py
```

旧表中的最高 global 不能自动视为当前最优结果，因为它可能来自不同模型池、receiver
recipe、packet source 或不完整 seed。重要的 Conv-family、all-ConvNet、hetero4/6 结果
仍保留为历史证据，不应删除或标记 invalid。
