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

from src.models.heads import build_head
from src.models.model_pool import build_backbone


class AgentModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        dataset: str,
        num_classes: int,
        image_size=(32, 32),
        norm_type: str = "instance",
        head_type: str = "shallow_mlp",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.num_classes = num_classes
        self.image_size = image_size
        self.norm_type = norm_type
        self.head_type = head_type
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout

        self.backbone = build_backbone(
            name=model_name,
            dataset=dataset,
            num_classes=num_classes,
            image_size=image_size,
            norm_type=norm_type,
        )
        self._replace_classifier_head(num_classes)

    def _replace_classifier_head(self, num_classes: int) -> None:
        for attr_name in ["classifier", "fc", "head"]:
            if not hasattr(self.backbone, attr_name):
                continue
            classifier = getattr(self.backbone, attr_name)
            in_dim = self._infer_classifier_in_dim(classifier)
            setattr(
                self.backbone,
                attr_name,
                build_head(
                    self.head_type,
                    in_dim,
                    num_classes,
                    hidden_dim=self.head_hidden_dim,
                    dropout=self.head_dropout,
                ),
            )
            return
        raise RuntimeError("backbone classifier not found; expected classifier, fc, or head")

    @staticmethod
    def _infer_classifier_in_dim(classifier: nn.Module) -> int:
        if hasattr(classifier, "in_dim"):
            return classifier.in_dim
        if hasattr(classifier, "in_features"):
            return classifier.in_features
        if hasattr(classifier, "weight"):
            return classifier.weight.shape[1]
        raise RuntimeError("cannot infer classifier input dimension")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_backbone(self) -> nn.Module:
        return self.backbone


def _agent_model_name(cfg: dict, agent_id: int) -> str:
    agent_models = cfg.get("agent_models", {})
    if agent_id in agent_models:
        return agent_models[agent_id]
    if str(agent_id) in agent_models:
        return agent_models[str(agent_id)]
    model_cfg = cfg.get("model", {})
    if "name" in model_cfg:
        return model_cfg["name"]
    raise KeyError(f"model type for agent_id={agent_id} not found in cfg['agent_models']")


def build_agent_model(cfg: dict, agent_id: int, device: torch.device):
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg["dataset"]
    model = AgentModel(
        model_name=_agent_model_name(cfg, agent_id),
        dataset=dataset_cfg["name"],
        num_classes=dataset_cfg["num_classes"],
        image_size=tuple(dataset_cfg.get("image_size", (32, 32))),
        norm_type=model_cfg.get("norm_type", "instance"),
        head_type=model_cfg.get("head_type", "shallow_mlp"),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
        head_dropout=model_cfg.get("head_dropout", 0.1),
    )
    return model.to(device)
