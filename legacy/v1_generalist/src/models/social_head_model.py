"""
Wrapper model with a frozen local classifier and a trainable copied social classifier.
"""

import copy
import torch
import torch.nn as nn

from src.models.agent_model import AgentModel


class SocialHeadAgent(nn.Module):
    def __init__(self, cfg, device: torch.device, feature_idx: int = None):
        super().__init__()
        self.device = device
        self.local_model = AgentModel(
            model_name=cfg["model"]["name"],
            dataset=cfg["dataset"]["name"],
            num_classes=cfg["dataset"]["num_classes"],
            image_size=tuple(cfg["dataset"]["image_size"]),
            norm_type=cfg["model"]["norm_type"],
        ).to(device)
        self.feature_idx = self._default_feature_idx() if feature_idx is None else feature_idx
        self.social_head = copy.deepcopy(self._local_classifier()).to(device)

    def _backbone(self):
        return self.local_model.get_backbone()

    def _local_classifier(self):
        backbone = self._backbone()
        for attr_name in ["classifier", "fc", "head", "heads"]:
            if hasattr(backbone, attr_name):
                return getattr(backbone, attr_name)
        raise RuntimeError("local classifier not found on backbone")

    def _default_feature_idx(self) -> int:
        backbone = self._backbone()
        if hasattr(backbone, "classifier"):
            return 2
        if hasattr(backbone, "fc"):
            return 5
        return 2

    def _classifier_in_features(self) -> int:
        classifier = self._local_classifier()
        if hasattr(classifier, "in_features"):
            return classifier.in_features
        if hasattr(classifier, "weight"):
            return classifier.weight.shape[1]
        raise RuntimeError("cannot infer classifier input dimension")

    def load_local_state_dict(self, state_dict):
        self.local_model.load_state_dict(state_dict)

    def init_social_head_from_local(self):
        self.social_head = copy.deepcopy(self._local_classifier()).to(self.device)

    def freeze_backbone(self):
        for param in self._backbone().parameters():
            param.requires_grad_(False)

    def freeze_local_head(self):
        for param in self._local_classifier().parameters():
            param.requires_grad_(False)

    def train_social_head_only(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.social_head.parameters():
            param.requires_grad_(True)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self._backbone()
        try:
            output = backbone(x, return_features=True)
            if isinstance(output, tuple):
                features = output[-1]
            else:
                features = output
            return features.view(features.size(0), -1)
        except TypeError:
            pass

        if hasattr(backbone, "get_feature"):
            output = backbone.get_feature(x, self.feature_idx, self.feature_idx)
            features = output[0] if isinstance(output, tuple) else output
            if isinstance(features, list):
                features = features[-1]
            return features.view(features.size(0), -1)

        raise RuntimeError("backbone does not expose features for social head")

    def forward(self, x: torch.Tensor, head: str = "local") -> torch.Tensor:
        if head == "local":
            return self.local_model(x)
        if head == "social":
            return self.social_head(self.extract_features(x))
        raise ValueError(f"unknown head: {head}")
