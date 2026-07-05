# 工作记录

---

## 2026-07-05

### 任务：AlexNet / VGG 作为社会化学习 Agent Backbone 适配

**目标**

为论文主实验（CIFAR-100, 4agent, 异构模式=conv-3 / conv-4 / VGG / AlexNet, IPC=10 & IPC=50）完成网络适配，使 AlexNet 和 VGG 可以作为 DSDM agent backbone 参与社会化学习 pipeline。

**实验设定（已确认）**

| agent | 架构 | width | classes | f_idx | last_feature shape |
|---|---|---|---|---|---|
| 0 | ConvNet-3 | w1.0（DSDM默认） | 0–24 | 2 | block3 output |
| 1 | ConvNet-4 | w1.5 | 25–49 | 3 | block4 output |
| 2 | VGG11-CIFAR | — | 50–74 | 10 | [B, 512] |
| 3 | AlexNetCIFAR | — | 75–99 | 7 | [B, 512] |

f_idx 统一指向 logits 前一层（last_feature），这是 DSDM 蒸馏特征层选取的核心约定。

---

### 代码修改

#### 1. `DSDM/train.py`

- 新增 import：`models.alexnet_cifar as AN`、`models.vgg_cifar as VN`
- `define_model` 函数加入 alexnet/vgg 分支：
  - `alexnet`：调用 `AN.alexnet_cifar(nclass, nch=args.nch)`
  - `vgg`：调用 `VN.vgg_cifar(nclass, nch=args.nch)`
- 注意：AlexNet/VGG 内部写死 BatchNorm，不通过 `norm_type` 动态切换

#### 2. `config_adapter.py` — `_apply_model_rules`

- 新增 f_idx 硬编码规则：
  - `alexnet` → `f_idx = "7"`（logits前一层 [B,512]）
  - `vgg` → `f_idx = "10"`（logits前一层 [B,512]）
- 新增 modeltag 分支：
  - `alexnet`/`vgg` 不依赖 depth，`modeltag` 直接使用架构名，避免错误地追加无意义的 depth 后缀

#### 3. `agent_data.py` — `_refresh_model_metadata` + `build_agent_args`

- `_refresh_model_metadata` 同步加入 alexnet/vgg 的 f_idx 和 modeltag 规则，与 config_adapter 保持一致
- `build_agent_args` 新增 per-model 蒸馏参数覆盖机制：
  - 读取 `model_pool.models.<model>.distillation.lr_img`
  - 读取 `model_pool.models.<model>.distillation.niter`
  - 读取 `model_pool.models.<model>.distillation.pretrain_dir`
  - 三者均为可选字段，存在则覆盖全局 distillation 设置，不存在则沿用全局默认

---

### 新建配置文件

#### `configs/main_cifar100_hetero4arch_ipc10.yaml`

- `run_name: cifar100_4agent_25cls_hetero4arch_ipc10`
- IPC = 10，factor = 2，decode_type = single，init = mix
- 每个模型在 model_pool 中配有独立 `distillation.lr_img`：
  - convnet3w1: `lr_img=0.1`
  - convnet4w15: `lr_img=0.1`
  - vgg: `lr_img=0.02`（来自 formal sweep 最优）
  - alexnet: `lr_img=0.005`（来自 formal sweep 最优）
- VGG/AlexNet 配有 teacher bank 路径（`pretrain_dir`）
- receiver 超参初始值：`epochs=100, lambda_fr=0.20, lambda_kd=0.5`

#### `configs/main_cifar100_hetero4arch_ipc50.yaml`

- `run_name: cifar100_4agent_25cls_hetero4arch_ipc50`
- IPC = 50，其余结构与 IPC=10 config 相同
- VGG/AlexNet 的 IPC=50 lr_img 暂沿用 IPC=10 最优值，后续需独立调参
- receiver 超参初始值：`epochs=250, lambda_fr=0.05, lambda_kd=0.5`

---

### 验证结果

**py_compile smoke test**

```
ALL OK（run_social_pipeline.py、agent_data.py、DSDM/train.py 等全部通过）
```

**per-agent 配置 dry-run 验证**

```
agent    net_type   depth  width  f_idx    modeltag           lr_img
------------------------------------------------------------------------
0        convnet    3      1.0    2        conv3in            0.1
1        convnet    4      1.5    3        conv4in_w1.5       0.1
2        vgg        3      1.0    10       vgg                0.02
3        alexnet    3      1.0    7        alexnet            0.005
```

f_idx 全部正确，per-model lr_img 覆盖生效。

---

### 注意事项与后续待确认

1. **VGG/AlexNet IPC=50 的 lr_img**：当前沿用 IPC=10 formal sweep 最优值，IPC=50 下最优 lr_img 尚未验证，正式实验前建议先做小规模 sweep。
2. **Teacher bank 路径**：config 中已配置 VGG/AlexNet 的 `pretrain_dir`，正式运行前需确认路径下有 10 个 `cifar100_model_*.pth` 文件，且为 `factor=2 / e20` 版本。
3. **expert training recipe**：VGG/AlexNet 的 `centralized_full.recipes` 当前使用通用 CIFAR recipe（augment=true），尚未针对这两个架构做专项 recipe 调优，建议正式跑 upper-bound 前确认训练收敛。
4. **receiver 超参**：hetero4arch 的最优 receiver 超参（epochs / lambda_fr / lambda_kd）尚未在新异构设定下验证，IPC=10/50 各提供了一组合理初始值，后续需要独立调参。
5. **不要使用 torchvision ImageNet 版本**：AlexNet/VGG 的实现是 CIFAR-100 32x32 定制版，不能替换为 torchvision 标准版本。

---

### 下一步建议

1. 确认 VGG/AlexNet teacher bank 路径可访问
2. 运行 `train_experts` 阶段，验证 AlexNet/VGG expert 训练正常收敛
3. 运行 `distill_packets` 阶段（IPC=10），验证 DSDM 特征层 f_idx 蒸馏正确
4. 完成 IPC=10 主实验对比（Heuristic / DSDM / DSDM+Logits）
5. 复用 IPC=10 经验，启动 IPC=50 实验
