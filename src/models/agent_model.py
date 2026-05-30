"""
作用：
    定义 agent 级模型封装。该模块负责从 model_pool 中构建 backbone，
    并作为多 agent 训练时统一使用的模型接口。

主要部分：
    1. AgentModel:
       - 初始化 backbone
       - forward(x): 返回 logits
       - get_backbone(): 返回底层 backbone，便于后续蒸馏或特征提取
    2. model_name / dataset / num_classes 等基本属性保存

输入输出：
    - 输入：图像张量 x，形状通常为 [B, C, H, W]
    - 输出：分类 logits，形状通常为 [B, num_classes]
"""

import torch
import torch.nn as nn

from src.models.model_pool import build_backbone


class AgentModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        dataset: str,
        num_classes: int,
        image_size=(32, 32),
        norm_type: str = "instance",
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.num_classes = num_classes
        self.image_size = image_size
        self.norm_type = norm_type

        self.backbone = build_backbone(
            name=model_name,
            dataset=dataset,
            num_classes=num_classes,
            image_size=image_size,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_backbone(self) -> nn.Module:
        return self.backbone