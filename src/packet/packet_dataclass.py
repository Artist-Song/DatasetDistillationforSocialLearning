"""
packet_dataclass.py

作用：
    定义 social packet 的标准数据结构。social packet 是本项目中
    跨 agent 传递知识的统一载体，用来替代直接交换参数。

主要部分：
    1. SocialPacket:
       - sender_id: 发送该 packet 的 agent 标识
       - class_ids: 该 packet 涉及的类别 id
       - images: 蒸馏后的图像数据
       - hard_labels: 对应硬标签
       - soft_targets: 发送端 anchor model 产生的软标签
       - meta: 额外元信息，例如 ipc、蒸馏温度、通信开销等

输入输出：
    - 输入：由 sender 侧蒸馏与打包模块构造
    - 输出：供 social pool、receiver 训练、通信统计模块统一使用

后续扩展建议：
    - 后面如果要加入 proto / cov / 历史信息，可以继续往 meta 中扩展
    - 后面如果要做 packet 序列化存盘，可在 packet_io.py 中基于这个结构实现
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class SocialPacket:
    """
    Social packet 的标准结构定义。

    字段说明：
        sender_id:
            发送该 packet 的 agent 名称，例如 "agent_1"

        class_ids:
            该 packet 覆盖的类别 id，形状通常为 [N]

        images:
            蒸馏后的图像张量，形状通常为 [B, C, H, W]

        hard_labels:
            图像对应的硬标签，形状通常为 [B]

        soft_targets:
            发送端模型对 packet 图像产生的 soft target，形状通常为 [B, num_classes]

        meta:
            其他附加信息，例如：
            {
                "ipc": 10,
                "temperature": 2.0,
                "sender_backbone": "resnet18"
            }
    """
    sender_id: str
    class_ids: torch.Tensor
    images: torch.Tensor
    hard_labels: torch.Tensor
    soft_targets: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)