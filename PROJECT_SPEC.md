# PROJECT_SPEC.md

## 项目名称

`social-packet-learning`

## 研究目标

本项目研究基于 packet 的社会化学习。

长期目标是让不同 agent 之间通过紧凑数据包进行知识传递，而不是传输模型参数、梯度、logits、概率或 soft target。

当前项目以 DSDM 源码为基础。第一阶段已经完成 DSDM 的配置入口、统一输出、packet 保存和 packet 评估。第二阶段在此基础上实现完整的异构 agent 社会化学习流程。

## 当前阶段

当前阶段为第二阶段：

```text
异构 agent 社会化 packet 学习
```

核心链路：

```text
专家类学习
→ DSDM 蒸馏
→ packet_hub 通信注册
→ receiver 二轮学习
→ 社会化学习结果评估
```

## 第二阶段目标

第二阶段实现以下功能：

1. 构建 CIFAR-10 上的 5-agent 场景。
2. 每个 agent 拥有 2 个 expert classes。
3. agent 使用异构模型。
4. 每个 agent 只用自己的 expert classes 训练 expert model pool。
5. 每个 agent 用自己的模型和数据生成 DSDM packet。
6. 所有 packet 注册到 `packet_hub/`。
7. 每个 receiver agent 读取全部 packet。
8. receiver 通过 `packet_consumer.py` 获得可直接训练的数据。
9. receiver 从自己的 expert model 出发二轮学习。
10. receiver loss 使用：
    ```text
    L = L_cls + 0.05 * L_FR
    ```
11. 评估每个 receiver 的 global accuracy、expert accuracy、new-class accuracy 和 forgetting。
12. 保存 `packet_manifest.csv` 和 `social_results.csv`。

第二阶段不实现：

1. 多轮通信。
2. sequential packet learning。
3. 动态 packet 选择。
4. agent 间参数聚合。
5. logits / soft target / gradient 传输。
6. t-SNE 可视化。
7. 消融实验。
8. 大规模数据集。

## 数据集设定

数据集：

```text
CIFAR-10
```

类别数：

```text
10
```

所有阶段都保留 CIFAR-10 全局标签。

不允许把 agent 内部标签重映射成局部标签。

例如 agent 2 负责：

```text
[4, 5]
```

其训练标签仍然是：

```text
4, 5
```

不是：

```text
0, 1
```

## agent 划分

agent 数量：

```text
5
```

每个 agent 负责 2 个 expert classes：

```python
agent_class_split = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}
```

## agent 异构模型设定

agent 模型来自 DSDM 模型库。

第一版使用：

```text
convnet
resnet
resnet_ap
```

模型分配：

```python
agent_model_split = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet",
    4: "resnet_ap",
}
```

如果 DSDM 源码中模型字符串不同，以源码实际支持的 `net_type` 字符串为准。

## 模型输出维度

所有 agent 的模型输出维度固定为：

```text
10
```

即使某个 agent 只训练两个 expert classes，也不能把分类头改成 2 维。

原因：

```text
receiver 二轮学习时需要学习全局 CIFAR-10 类别。
```

## active_class_ids 与 num_classes

必须区分：

```text
num_classes = 10
active_class_ids = 当前 agent 的 expert classes
```

例如 agent 3：

```text
num_classes = 10
active_class_ids = [6, 7]
```

DSDM 中凡是遍历当前要蒸馏的类别，应使用：

```python
args.active_class_ids
```

模型构造和分类头输出维度仍使用：

```python
args.num_classes
```

## 第一阶段：专家类学习

每个 agent 只用自己的 expert classes 训练本地 expert model pool。

例如：

```text
agent 0:
  model = convnet
  train classes = [0, 1]
  output dim = 10

agent 2:
  model = resnet
  train classes = [4, 5]
  output dim = 10
```

每个 agent 训练：

```text
pretrained_model_number 个 guide models
```

默认沿用 DSDM：

```text
pretrained_model_number = 10
```


其中最后一个 guide model 作为该 agent 的：

expert_model.pt

如果：

pretrained_model_number = 10

则：

expert_model.pt = guide_model_9.pt

第一版不做 best model 选择，直接使用最后一个 guide model，原因是最后一个模型大概率训练最充分、效果最好。


## 第二阶段：DSDM packet 蒸馏

每个 agent 用自己的 expert model pool 和自己的 expert classes 生成 DSDM packet。

例如：

```text
agent 0: convnet + classes [0,1] → agent_0_dsdm_packet.pt
agent 2: resnet + classes [4,5] → agent_2_dsdm_packet.pt
agent 4: resnet_ap + classes [8,9] → agent_4_dsdm_packet.pt
```

每个 agent 的 DSDM packet 只包含自己的两个 expert classes。

packet 标签保留全局 CIFAR-10 标签。

## DSDM factor 设定

DSDM 第一版使用：

```text
factor = 2
```

`factor=2` 表示 DSDM 内部将 raw synthetic images decode 成 `2 * 2 = 4` 倍有效训练图。

重要规则：

```text
factor 不计入通信量
```

通信量只按 raw synthetic images 统计。

## packet 格式

DSDM packet 推荐格式：

```python
{
    "images": images,
    "labels": labels,
    "class_ids": class_ids,
    "source": "dsdm",
    "dataset": "cifar10",
    "ipc": ipc,
    "meta": {
        "run_name": run_name,
        "method": "DSDM",
        "sender_agent": agent_id,
        "sender_model": model_name,
        "factor": factor,
        "decode_type": decode_type
    }
}
```

packet 中禁止保存：

```text
model_state_dict
teacher_logits
teacher_probs
soft_targets
gradients
optimizer_state
```

## 第三阶段：packet_hub 通信注册

通信不做真实网络传输，第一版用 `packet_hub/` 表示所有 agent 可见的数据包中心。

路径：

```text
outputs/{run_name}/packet_hub/
```

内容：

```text
agent_0_dsdm_packet.pt
agent_1_dsdm_packet.pt
agent_2_dsdm_packet.pt
agent_3_dsdm_packet.pt
agent_4_dsdm_packet.pt
packet_manifest.csv
```

`packet_manifest.csv` 字段：

```csv
sender_agent,sender_model,classes,method,ipc,packet_path
```

receiver 训练时读取 `packet_manifest.csv` 来定位所有 packet。

## 通信量定义

通信量只记录 receiver 从其他 agent 收到的 raw packet image 数量。

公式：

```text
external_comm_images = (num_agents - 1) * classes_per_agent * ipc
```

当前设定下：

```text
external_comm_images = 4 * 2 * ipc = 8 * ipc
```

如果 `ipc=10`：

```text
external_comm_images = 80
```

不记录：

```text
self packet
factor
effective_ipc
decode 后训练样本数
```

## packet_consumer

新增或维护：

```text
packet_consumer.py
```

职责：

```text
把 packet 内部存储格式转换为 receiver 可直接训练的数据。
```

统一接口：

```python
def consume_packet_for_training(args, packet_path):
    """读取 packet，并转换为接收端可直接训练的数据。"""
    ...
```

处理规则：

```text
DSDM:
  执行 DSDM 原始 decode 逻辑

Heuristic:
  直接返回 images / labels

Importance:
  直接返回 images / labels
```

第二阶段主实验只要求 DSDM packet。

## 第四阶段：receiver 二轮学习

每个 agent 都作为一次 receiver。

receiver 使用：

```text
自己的 self packet
+
其他 4 个 agent 的 external packets
```

训练流程：

```text
读取 packet_manifest.csv
↓
读取全部 5 个 packet
↓
每个 packet 经过 packet_consumer.py
↓
concat 成 receiver 训练集
↓
class-balanced sampler
↓
从 expert_model.pt 初始化 receiver model
↓
使用 CE + 0.05 FR 训练
↓
保存 after_social.pt
```

## receiver loss

总损失：

```text
L = L_cls + 0.05 * L_FR
```

分类损失：

```text
L_cls = CE(f_new(x_all), y_all)
```

对全部 packet 样本计算。

专家类保持损失：

```text
L_FR = MSE(
  z_old(x_expert)[:, C_expert],
  z_new(x_expert)[:, C_expert]
)
```

其中：

```text
z_old：receiver 二轮学习前的冻结 expert model
z_new：正在训练的 receiver model
C_expert：receiver 自己的 expert classes
x_expert：batch 中属于 receiver expert classes 的样本
```

约束规则：

1. FR 使用 MSE。
2. FR 权重固定 0.05。
3. FR 只对 receiver 自己 expert 类样本计算。
4. FR 只约束 expert class logits。
5. 新类样本不参与 FR。
6. 不做动态调权。
7. 不做 FR loss 消融。

## 采样方式

receiver 训练使用：

```text
class-balanced sampler
```

第一版不做 sequential packet learning。

所有 packet 合并后 joint training。

## 评估指标

每个 receiver 训练前后评估：

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
acc_global：CIFAR-10 全部 test set accuracy
acc_expert：receiver 自己 expert classes 的 test set accuracy
acc_new：非 receiver expert classes 的 test set accuracy
forgetting = acc_expert_before - acc_expert_after
```

`forgetting` 保留 signed value。

## 输出目录

主输出目录：

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
        guide_model_0.pt
        guide_model_1.pt
        ...
        guide_model_9.pt
        expert_model.pt
      packets/
        dsdm_packet.pt
      synthetic/
        data_best.pt
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
        before_social.pt
        after_social.pt
      metrics/
        social_metrics.json
      logs/
        train.log

    receiver_agent_1/
    receiver_agent_2/
    receiver_agent_3/
    receiver_agent_4/

  metrics/
    packet_results.csv
    social_results.csv
```

## social_results.csv

路径：

```text
outputs/{run_name}/metrics/social_results.csv
```

字段：

```csv
run_name,receiver_agent,receiver_model,expert_classes,method,ipc,external_comm_images,acc_global_before,acc_expert_before,acc_global_after,acc_expert_after,acc_new_after,forgetting,loss_cls,loss_fr,time
```

## 主运行脚本

新增：

```text
run_social_pipeline.py
```

支持：

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

## AutoDL 工作流

本地 Codex 改完代码后，用户手动执行：

```bash
git add .
git commit -m "stage2: add social packet learning pipeline"
git push
```

AutoDL 上执行：

```bash
cd /root/autodl-tmp/social-packet-learning
git pull
conda activate <环境名>
python run_social_pipeline.py --config configs/main.yaml --stage all
```

如果中断，可以分阶段继续运行。

## 第二阶段完成标准

以下命令可用：

```bash
python run_social_pipeline.py --config configs/main.yaml --dry-run
```

```bash
python run_social_pipeline.py --config configs/main.yaml --stage train_experts --only-agent 0
```

```bash
python run_social_pipeline.py --config configs/main.yaml --stage distill_packets --only-agent 0
```

```bash
python run_social_pipeline.py --config configs/main.yaml --stage build_communication
```

```bash
python run_social_pipeline.py --config configs/main.yaml --stage train_receivers --only-receiver 0
```

完整实验命令：

```bash
python run_social_pipeline.py --config configs/main.yaml --stage all
```

## 代码风格要求

1. 所有新增说明和注释使用中文。
2. 每个新增或修改的函数、类都必须有简短中文注释。
3. 不要写过度复杂的抽象。
4. 不要大面积重构 DSDM。
5. 优先保持第一阶段已有功能可用。
6. 每次修改后先做 dry-run 或 smoke test。
7. 不要自动 git push。