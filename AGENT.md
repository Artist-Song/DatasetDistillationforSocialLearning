# AGENTS.md

## 项目定位

本项目名为 `social-packet-learning`。

当前项目以 DSDM 源码为起点，第一阶段目标不是重写 DSDM，而是在尽量保留 DSDM 原始逻辑的基础上，做轻量工程化改造，使其后续能够接入社会化学习中的 packet 构建、packet 传输和多 agent 实验。

当前 DSDM 源码位于：

```text
DSDM/
```

## 当前阶段目标

当前阶段是第一阶段：DSDM 工程适配。

只做以下事情：

1. 增加统一配置入口。
2. 保留 DSDM 原始算法逻辑。
3. 统一实验输出目录。
4. 将 DSDM 蒸馏结果保存为 packet。
5. 增加两个对比 packet 方法：
   - Heuristic：每类随机选择真实训练样本。
   - Importance：每类选择低置信度难样本。
6. 三种 packet 使用统一评估协议。
7. 所有实验结果追加保存到全局结果表。
8. 为未来 t-SNE 可视化、多 agent、通信实验预留目录和配置字段，但第一阶段不实现。

当前阶段不要实现：

1. multi-agent split。
2. agent communication。
3. social learning。
4. receiver adaptation。
5. heterogeneous agent learning。
6. t-SNE 具体逻辑。
7. 联邦学习或模型参数传输。
8. soft target / logits / gradient 传输。

## 代码修改原则

1. 尽量不改 DSDM 主体算法。
2. 不重写 DSDM 的蒸馏循环、loss、模型池和验证逻辑。
3. 优先通过配置适配器、输出管理器、wrapper 实现新功能。
4. 保留 DSDM 原始命令行运行能力。
5. 新增功能尽量放在 DSDM 外层，不要污染核心代码。
6. packet 中禁止保存模型参数、梯度、logits、概率、soft target。
7. 所有新增模块应当尽量小而清晰，避免一次性大重构。
8. 每次修改后优先做 smoke test，不要直接运行完整长实验。

## 中文要求

本项目所有工程说明、Codex prompt、代码注释、汇报说明都使用中文。

## 注释要求

每个新增或修改的函数、类都必须有简短中文注释。

要求：

1. 类注释说明这个类负责什么。
2. 函数注释说明这个函数输入什么、输出什么、主要做什么。
3. 注释要短，不要写长篇解释。
4. 不要求每一行代码都有注释，只要求函数和类级别有中文说明。
5. 如果修改 DSDM 原函数，也要补充或保留简短中文注释，不能破坏原逻辑。

示例：

```python
class OutputManager:
    """统一管理实验输出路径和结果保存。"""
    ...

def save_packet(...):
    """保存统一格式的社会化学习 packet。"""
    ...
```

## Git 工作流要求

用户在本地使用 git 做版本管理，并在 AutoDL 上通过 `git pull` 同步代码。

Codex 每次修改代码后必须：

1. 不执行 `git push`。
2. 不自动执行 `git commit`，除非用户明确要求。
3. 修改完成后运行必要的 smoke test。
4. 汇报 `git status`。
5. 给出建议的 `git add` / `git commit` / `git push` 命令，但由用户手动执行。
6. 不要改写 git 历史。
7. 不要强制 reset、rebase、clean，除非用户明确要求。

建议汇报格式：

```bash
git status

git add <修改的文件>
git commit -m "stage1: add config entry and output manager"
git push
```

用户会手动执行提交和推送，然后在 AutoDL 上执行：

```bash
git pull
```

## 预期项目结构

当前根目录结构应逐步形成：

```text
social-packet-learning/
  AGENTS.md
  PROJECT_SPEC.md
  configs/
    main.yaml
  prompts/
    CODEX_PROMPT_STAGE1_CONFIG_AND_OUTPUT.md
  DSDM/
    DSDM.py
    argument.py
    data.py
    pre_train_model.py
    train.py
    test.py
    models/
    misc/
  run_from_config.py
  config_adapter.py
  output_manager.py
  selection_methods.py
  evaluate_packet.py
  outputs/
    global_results.csv
```

## 配置入口规范

统一配置文件：

```text
configs/main.yaml
```

统一运行入口：

```bash
python run_from_config.py --config configs/main.yaml
```

dry-run 入口：

```bash
python run_from_config.py --config configs/main.yaml --dry-run
```

`dry-run` 只做：

1. 读取配置。
2. 解析 DSDM 参数。
3. 打印完整 resolved args。
4. 打印输出路径。
5. 不启动 DSDM 训练和蒸馏。

## 输出规范

所有单次实验输出都放在：

```text
outputs/{project.run_name}/
```

目录结构：

```text
config/
logs/
checkpoints/
synthetic/
packets/
visuals/
metrics/
embeddings/tsne/
```

全局结果表：

```text
outputs/global_results.csv
```

全局结果表用于汇总所有实验，不替代单实验目录中的详细 metrics。

## packet 规范

所有 packet 必须使用统一格式：

```python
{
    "images": images,
    "labels": labels,
    "class_ids": class_ids,
    "source": source,
    "dataset": dataset,
    "ipc": ipc,
    "meta": {
        "run_name": run_name,
        "method": method
    }
}
```

第一阶段允许的 `source`：

```text
dsdm
heuristic
importance
```

packet 中禁止出现：

```text
model_state_dict
teacher_logits
teacher_probs
soft_targets
gradients
optimizer_state
```

## 三种 packet 来源

### DSDM

由 DSDM 蒸馏生成 synthetic images。

输出：

```text
outputs/{run_name}/packets/dsdm_packet.pt
```

### Heuristic

每类随机选择 `ipc` 张真实训练样本。

输出：

```text
outputs/{run_name}/packets/heuristic_packet.pt
```

### Importance

使用 DSDM pretrained guide model pool，对训练集样本打分，每类选择低置信度样本。

默认打分方式：

```text
confidence = mean(max_softmax_probability(model_i(image)))
```

每类选择 confidence 最低的 `ipc` 张。

输出：

```text
outputs/{run_name}/packets/importance_packet.pt
```

## 评估规范

三种 packet 必须用同一个评估协议。

评估原则：

1. 同一个 fresh model。
2. 同一训练轮数。
3. 同一数据增强。
4. 同一测试集。
5. 尽量复用 DSDM 原验证逻辑。
6. packet 只替代训练数据，不改变验证协议。

评估入口：

```bash
python evaluate_packet.py --config configs/main.yaml --packet-source dsdm
python evaluate_packet.py --config configs/main.yaml --packet-source heuristic
python evaluate_packet.py --config configs/main.yaml --packet-source importance
```

## 汇报要求

每次完成任务后，必须用中文汇报：

1. 新增了哪些文件。
2. 修改了哪些文件。
3. 是否保留 DSDM 原始运行方式。
4. dry-run 如何运行。
5. smoke test 是否通过。
6. 当前 `git status`。
7. 建议用户手动执行的 git 命令。
8. 不要自行 push。