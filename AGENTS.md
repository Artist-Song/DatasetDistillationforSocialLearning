# AGENTS.md

## 项目定位

本项目名为 `social-packet-learning`。

当前项目以 DSDM 源码为起点，研究基于 packet 的社会化学习。长期目标是让不同 agent 之间通过紧凑数据包进行知识传递，而不是传输模型参数、梯度、logits 或 soft target。

当前代码核心仍然来自：

```text
DSDM/
```

第二阶段目标是在第一阶段 DSDM 工程适配的基础上，增加完整社会化学习流程：

```text
专家类学习
→ DSDM packet 蒸馏
→ packet_hub 通信注册
→ 接收端二轮学习
→ 结果汇总
```

## 当前阶段目标

当前阶段是第二阶段：异构 agent 社会化 packet 学习。

只做以下事情：

1. 构建 CIFAR-10 上的 5-agent 场景。
2. 每个 agent 拥有 2 个 expert classes。
3. agent 之间模型异构，使用 DSDM 模型库中的 `convnet`、`resnet`、`resnet_ap`。
4. 每个 agent 先训练自己的 expert model pool。
5. 每个 agent 用自己的模型和自己的 expert classes 生成 DSDM packet。
6. 所有 packet 注册到全局 `packet_hub/`。
7. 每个 receiver agent 读取全部 packet。
8. 使用 `packet_consumer.py` 将 packet 转换成 receiver 可直接训练的数据。
9. receiver 从自己的 expert model 出发进行二轮学习。
10. receiver loss 使用分类损失和专家类保持损失：
    ```text
    L = L_cls + 0.05 * L_FR
    ```
11. 评估 global accuracy、expert accuracy、new-class accuracy 和 forgetting。
12. 结果写入 `social_results.csv`。

当前阶段不要实现：

1. 多轮通信。
2. 顺序 packet 学习。
3. 动态路由。
4. attention-based packet fusion。
5. t-SNE 可视化。
6. 联邦学习参数聚合。
7. 模型参数通信。
8. logits / soft target / gradient 通信。
9. 复杂消融实验。

## 代码修改原则

1. 尽量复用第一阶段已有代码。
2. 不要重写 DSDM 主体算法。
3. 不要大面积重构 DSDM 源码。
4. DSDM 的蒸馏、decode、验证逻辑尽量复用原实现。
5. 新增社会化学习逻辑应放在外层 wrapper 和独立模块中。
6. agent 子任务必须保留 CIFAR-10 全局标签。
7. agent 模型输出维度始终是 10。
8. 每个 agent 只加载自己的 expert classes 数据。
9. receiver 训练时使用 self packet + external packets。
10. 通信量只统计 external packets 的 raw image 数量。
11. DSDM 的 factor 只属于 DSDM 方法内部展开，不计入通信量。

## 中文要求

本项目所有工程说明、Codex prompt、代码注释、汇报说明都使用中文。

## 注释要求

每个新增或修改的函数、类都必须有简短中文注释。

要求：

1. 类注释说明这个类负责什么。
2. 函数注释说明这个函数输入什么、输出什么、主要做什么。
3. 注释要短，不要写长篇解释。
4. 不要求每一行代码都有注释，只要求函数和类级别有中文说明。
5. 如果修改已有函数，也要补充或保留简短中文注释，不能破坏原逻辑。

示例：

```python
class SocialTrainer:
    """负责 receiver agent 的二轮社会化学习。"""
    ...

def consume_packet_for_training(args, packet_path):
    """读取 packet，并转换为接收端可直接训练的数据。"""
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
git commit -m "stage2: add social packet learning pipeline"
git push
```

用户会手动执行提交和推送，然后在 AutoDL 上执行：

```bash
git pull
```

## 第二阶段核心设定

### 数据集

```text
CIFAR-10
```

### agent 数量

```text
5 个 agent
```

### 每个 agent 的 expert classes

```python
agent_class_split = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}
```

所有标签必须保留 CIFAR-10 全局标签，不允许重映射。

### agent 模型异构

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet",
    4: "resnet_ap",
}
```

具体模型字符串必须和 DSDM 源码中的模型构造参数一致。

### 输出维度

所有 agent 的模型输出维度固定为：

```text
10
```

即使某个 agent 只训练两个类，也不能把分类头改成 2 维。

## 通信量规则

通信量只统计 receiver 从其他 agent 收到的 raw packet image 数量。

对于当前设定：

```text
external_comm_images = (num_agents - 1) * classes_per_agent * ipc
```

即：

```text
external_comm_images = 4 * 2 * ipc = 8 * ipc
```

如果 `ipc=10`：

```text
external_comm_images = 80
```

注意：

1. self packet 不计入通信量。
2. factor 不计入通信量。
3. decode 后的 effective image 数量不计入通信量。
4. 通信量字段只记录 `external_comm_images`。

## DSDM factor 规则

DSDM 的 `factor=2` 是 DSDM 方法内部的数据展开方式。

对于 DSDM packet：

```text
raw synthetic image
→ DSDM decode
→ factor * factor 张训练图
```

当 `factor=2` 时：

```text
1 张 raw synthetic image → 4 张训练图
```

但是通信量仍然只按 raw synthetic image 计算。

## packet_consumer 规则

新增或维护统一文件：

```text
packet_consumer.py
```

它负责把 packet 内部存储格式转换成 receiver 可直接训练的数据。

统一接口建议为：

```python
def consume_packet_for_training(args, packet_path):
    """读取 packet，并转换为接收端可直接训练的数据。"""
    ...
```

处理逻辑：

```text
source == "dsdm":
    使用 DSDM 原 decode 逻辑展开 factorized synthetic data

source == "heuristic":
    直接返回 images / labels

source == "importance":
    直接返回 images / labels
```

第二阶段主实验只要求 DSDM packet，但接口可以兼容 Heuristic / Importance。

## receiver loss 规则

receiver 二轮学习使用：

```text
L = L_cls + 0.05 * L_FR
```

其中：

```text
L_cls = CE(f_new(x_all), y_all)
```

对所有 packet 样本计算。

```text
L_FR = MSE(
  z_old(x_expert)[:, C_expert],
  z_new(x_expert)[:, C_expert]
)
```

只对 receiver 自己 expert classes 的样本计算。

要求：

1. FR 使用 MSE。
2. FR 权重固定为 0.05。
3. FR 只作用于 receiver 自己的 expert 类样本。
4. FR 只约束 expert class logits。
5. 新类样本不参与 FR。
6. 不做动态调权。
7. 不做 L1/MSE 消融。
8. 不做 sequential packet learning。

## receiver 训练数据规则

每个 receiver agent 训练时使用：

```text
自己的 self packet
+
其他所有 agent 的 external packets
```

所有 packet 先经过 `packet_consumer.py`，再 concat 成训练数据。

训练采样使用：

```text
class-balanced sampler
```

第一版不做顺序 packet 学习，统一采用 joint packet training。

## 评估指标

每个 receiver 训练前后都需要评估：

```text
acc_global_before
acc_expert_before
acc_global_after
acc_expert_after
acc_new_after
forgetting
```

定义：

```text
acc_global：CIFAR-10 全 10 类 test set accuracy
acc_expert：receiver 自己 expert classes test set accuracy
acc_new：非 receiver expert classes test set accuracy
forgetting = acc_expert_before - acc_expert_after
```

`forgetting` 保留 signed value，不强制截断为非负。

## 预期项目结构

第二阶段后，项目根目录应逐步形成：

```text
social-packet-learning/
  AGENTS.md
  PROJECT_SPEC.md
  configs/
    main.yaml
  prompts/
    CODEX_PROMPT_STAGE2_SOCIAL_PIPELINE.md
  DSDM/
  run_from_config.py
  config_adapter.py
  output_manager.py
  packet_consumer.py
  run_social_pipeline.py
  agent_data.py
  agent_trainer.py
  social_trainer.py
  social_metrics.py
  social_output_manager.py
```

如果已有 `output_manager.py` 足够，可以将 `social_output_manager.py` 的功能合并进去，但要保持逻辑清晰。

## 输出目录规范

第二阶段主输出目录：

```text
outputs/{run_name}/
```

推荐结构：

```text
outputs/{run_name}/
  config/
    main.yaml
    resolved_args.json

  agents/
    agent_0/
      checkpoints/
      packets/
      synthetic/
      visuals/
      metrics/
    agent_1/
    agent_2/
    agent_3/
    agent_4/

  packet_hub/
    agent_0_dsdm_packet.pt
    agent_1_dsdm_packet.pt
    agent_2_dsdm_packet.pt
    agent_3_dsdm_packet.pt
    agent_4_dsdm_packet.pt
    packet_manifest.csv

  social_learning/
    receiver_agent_0/
      checkpoints/
      metrics/
      logs/
    receiver_agent_1/
    receiver_agent_2/
    receiver_agent_3/
    receiver_agent_4/

  metrics/
    packet_results.csv
    social_results.csv
```

## packet_manifest.csv

固定路径：

```text
outputs/{run_name}/packet_hub/packet_manifest.csv
```

字段：

```csv
sender_agent,sender_model,classes,method,ipc,packet_path
```

示例：

```csv
0,convnet,"0,1",DSDM,10,outputs/run/packet_hub/agent_0_dsdm_packet.pt
1,convnet,"2,3",DSDM,10,outputs/run/packet_hub/agent_1_dsdm_packet.pt
2,resnet,"4,5",DSDM,10,outputs/run/packet_hub/agent_2_dsdm_packet.pt
3,resnet,"6,7",DSDM,10,outputs/run/packet_hub/agent_3_dsdm_packet.pt
4,resnet_ap,"8,9",DSDM,10,outputs/run/packet_hub/agent_4_dsdm_packet.pt
```

## social_results.csv

固定路径：

```text
outputs/{run_name}/metrics/social_results.csv
```

字段：

```csv
run_name,receiver_agent,receiver_model,expert_classes,method,ipc,external_comm_images,acc_global_before,acc_expert_before,acc_global_after,acc_expert_after,acc_new_after,forgetting,loss_cls,loss_fr,time
```

## 主运行入口

第二阶段总入口：

```bash
python run_social_pipeline.py --config configs/main.yaml --stage all
```

支持分阶段运行：

```bash
python run_social_pipeline.py --config configs/main.yaml --stage train_experts
python run_social_pipeline.py --config configs/main.yaml --stage distill_packets
python run_social_pipeline.py --config configs/main.yaml --stage build_communication
python run_social_pipeline.py --config configs/main.yaml --stage train_receivers
python run_social_pipeline.py --config configs/main.yaml --stage all
```

建议支持：

```bash
--dry-run
--resume
--overwrite
--only-agent 0
--only-receiver 0
```

## 汇报要求

每次完成任务后，必须用中文汇报：

1. 新增了哪些文件。
2. 修改了哪些文件。
3. 是否保留第一阶段和 DSDM 原始运行方式。
4. dry-run 如何运行。
5. smoke test 是否通过。
6. 当前 `git status`。
7. 建议用户手动执行的 git 命令。
8. 不要自行 push。