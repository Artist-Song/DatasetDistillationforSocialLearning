# AGENTS.md

## 项目目标
实现一个基于 PyTorch 的多 agent 社会化学习框架，将数据集蒸馏引入社会化学习中，用 social packet 替代参数交互，以缓解模型异构下的知识传递问题。

## v1 范围
- 数据集：CIFAR10 / CIFAR100
- 数据划分：direct / social
- 模型池：conv / resnet18 / vit_tiny
- social packet：P = {X_distill, y, q}
- 通信策略：all-to-all
- baselines:
  1. local_only
  2. raw_share
  3. masc_style_baseline
  4. packet_x_only
  5. packet_x_q

## 工程原则
- 优先交付最小可运行版本
- 每一阶段都要有独立入口脚本
- local anchor 与 DSDM packet 必须支持离线缓存
- v1 不实现动态路由
- v1 不实现复杂 projector
- v1 优先保证 direct split 跑通，再做 social split

## 指标
- expert_accuracy
- general_accuracy
- average_accuracy
- bytes_per_packet
- total_comm_bytes
- accuracy_per_mb
- heterogeneity_gap

## 沟通规则
- 默认中文
- 每次修改前先给简短计划
- 每次完成后说明：
  - 改了哪些文件
  - 当前能运行到哪一步
  - 还缺什么