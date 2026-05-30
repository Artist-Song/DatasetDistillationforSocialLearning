"""
seed.py

作用：
    统一管理随机种子，保证训练、数据划分、蒸馏过程尽可能可复现。

主要部分：
    1. set_seed(seed): 设置 Python / NumPy / PyTorch 的随机种子

输入输出：
    - 输入：整数随机种子
    - 输出：无返回值，直接修改运行环境的随机状态

后续扩展建议：
    - 后面如果需要更严格的可复现控制，可以增加 cudnn 相关选项
    - 如果以后做多卡训练，也可以在这里继续补充分布式随机种子控制
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    设置随机种子，尽量保证实验可复现。

    参数：
        seed: 随机种子整数

    返回：
        无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 下面两项会让结果更稳定，但在某些情况下可能略影响速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False