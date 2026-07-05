# DeSA / Synthetic Anchors 精读笔记

文件：

```text
Synthetic Anchors - Data and Model Heterogeneities in Decentralized FL.pdf
```

题目：

```text
Overcoming Data and Model Heterogeneities in Decentralized Federated Learning via Synthetic Anchors
```

作者：

```text
Chun-Yin Huang, Kartik Srinivas, Xin Zhang, Xiaoxiao Li
```

## 一句话贡献

DeSA 在去中心化、模型异构 FL 中引入 synthetic anchors，先用合成锚点构造跨客户端共享的输入空间参考，再通过特征正则和 anchor-logit KD 让不同模型学习更通用的表示。

## 解决的问题

DeSA 处理的是 decentralized federated mutual learning：

```text
没有中心服务器；
不同 client 可以有不同模型；
不同 client 的数据分布存在 domain shift / label shift；
目标是让每个 client 的本地模型在所有 client 域上都有更好的泛化能力。
```

这和我们的方向非常接近，因为它同时触及：

1. 去中心化协作。
2. 模型异构。
3. 合成数据作为通信/对齐载体。
4. KD 作为异构模型之间的软知识传递方式。

但它的任务目标不是 socialized learning。DeSA 更强调跨域泛化；我们强调 sender expert knowledge 如何被 receiver 吸收，同时保留 receiver 自身 expert classes 的优势。

## 方法核心

DeSA 的 pipeline 可以拆成三步：

| 阶段 | 做法 | 作用 |
|---|---|---|
| Synthetic anchor generation | 每个 client 用 distribution matching / MMD 从本地数据生成 class-balanced synthetic anchors | 得到可共享的输入空间锚点 |
| Anchor sharing / aggregation | client 之间共享本地 synthetic anchors，形成 global synthetic anchors | 构造所有 client 共同可见的参考数据 |
| Local DeSA training | 用 CE、REG loss、KD loss 训练本地模型 | 同时学习本地监督、跨域表示和邻居模型知识 |

其中 synthetic anchor generation 使用随机特征提取器做 distribution matching，不需要额外预训练模型。这一点和 DSDM 不同：DSDM 依赖更强的 semantic distribution matching，并使用预训练模型族提供更语义化的特征空间。

DeSA 的最终训练目标可以概括为：

```text
L = L_CE(D_i ∪ D_syn)
  + lambda_REG * L_REG(D_i, D_syn)
  + lambda_KD  * L_KD(D_syn, neighbor logits)
```

三个 loss 的直观含义：

| Loss | 作用 |
|---|---|
| CE | 用本地真实数据和 synthetic anchors 做分类监督 |
| REG | 在 latent space 中把本地数据表示拉向 synthetic anchors 表示，增强 domain-invariant representation |
| KD | 在 shared synthetic anchors 上匹配邻居模型的平均 logits |

## 通信对象

DeSA 的通信分成两类：

| 阶段 | 通信内容 | 说明 |
|---|---|---|
| Pre-FL | synthetic anchor images | 每个 client 分享本地合成锚点，形成 global anchors |
| During-FL | global anchor logits | 训练过程中在 anchors 上交换 logits，而不是传完整模型 |

论文的通信表强调：DeSA 在训练期间只共享 virtual global anchor logits，因此比完整模型传输更轻。

这里要注意一个边界：DeSA 的 synthetic anchors 是“全局对齐基准”，不是某个 sender 的 expert knowledge packet。我们的 packet 是 sender-specific 的知识载体，重点是把每个 sender 的 expert classes 压缩后传给 receiver。

## 实验设置

DeSA 的实验主要覆盖三组 domain-shift 数据：

| 数据集组 | client 含义 | 任务特点 |
|---|---|---|
| DIGITS | MNIST、SVHN、USPS、SynthDigits、MNIST-M | 数字识别，不同视觉域 |
| OFFICE | Amazon、Caltech、DSLR、Webcam | Office 物体，不同拍摄域 |
| CIFAR10C | CIFAR-10-C 子集 | corruption/domain shift，并带有 Dirichlet label shift |

模型异构实验中，client 随机分配 ConvNet / AlexNet。对比方法包括 FedHe、FedDF、FCCL、FedProto 等异构 FL 方法；同构实验还比较 FedAvg、FedProx、MOON、Scaffold、FedGen 等。

## 和我们工作的关系

DeSA 是目前最需要认真对比的一篇，因为它和我们共享了三个关键词：

```text
decentralized
model heterogeneity
synthetic input-space carrier
```

但它和我们仍有几条关键差异：

| 维度 | DeSA | 我们 |
|---|---|---|
| 任务目标 | 去中心化 FL 下提升跨域泛化 | 社会化学习下吸收他人 expert classes 并保持自身 expert 优势 |
| 合成数据角色 | global synthetic anchors，作为所有 client 的共同对齐基准 | sender-specific distilled knowledge packets，作为 agent-to-agent 知识载体 |
| 通信过程 | pre-FL 共享 anchors，训练中反复交换 anchor logits | 第一层目标是传 image packet；logits 是可选 packet annotation |
| 数据设定 | domain shift / label shift FL | CIFAR-100 class-disjoint socialized learning |
| 蒸馏方法 | distribution matching + random feature extractor | DSDM semantic distribution matching，当前默认 factor=2 |

## 对实验设计的启发

DeSA 可以提供两个层面的参考：

1. Related Work：它证明 synthetic input-space object 可以帮助去中心化模型异构学习，是我们动机最接近的相关工作。
2. Adapted baseline：如果实现 DeSA-like baseline，需要明确它的 global anchor / repeated logits 通信口径，不能直接和一次性 sender packet 通信量混为一谈。

更稳妥的实验定位是：

```text
DeSA-style 方法可作为“synthetic anchors for heterogeneous decentralized FL”的代表；
我们的贡献是把 synthetic/distilled data 从 global alignment anchor 改造成 sender expert knowledge packet，
并放入 socialized learning 的 expert/new retention-absorption 目标中评估。
```

## 写作建议

Introduction / Related Work 中可以这样使用 DeSA：

```text
Synthetic-anchor methods show that input-space synthetic objects can mediate knowledge transfer among heterogeneous decentralized clients. However, these anchors are usually designed as global alignment references for federated mutual learning. In contrast, our distilled packets are sender-specific carriers of expert knowledge for socialized learning, where each receiver must absorb complementary classes while retaining its own expertise.
```

## 需要复查的细节

1. 如果后续实现 DeSA-like baseline，需要重新核对其通信量表，把 pre-FL synthetic anchors 和 during-FL logits 都计入。
2. DeSA 使用 ConvNet / AlexNet 异构；我们当前是 ConvNet / ResNet / ResNet_AP，异构跨度和训练 recipe 不完全一致。
3. DeSA 的 label space 默认是所有 client 共享的 K 类；我们的 class-disjoint 设置里每个 sender 初始只掌握 25 个 expert classes，但输出维度仍为 100。
