"""
Wrapper model with a frozen local head and a trainable social head.
"""

import torch
import torch.nn as nn

from src.models.agent_model import AgentModel


class SocialHeadAgent(nn.Module):
    def __init__(self, cfg, device: torch.device, feature_idx: int = None):
        super().__init__()
        self.local_model = AgentModel(
            model_name=cfg["model"]["name"],
            dataset=cfg["dataset"]["name"],
            num_classes=cfg["dataset"]["num_classes"],
            image_size=tuple(cfg["dataset"]["image_size"]),
            norm_type=cfg["model"]["norm_type"],
        ).to(device)
        self.feature_idx = self._default_feature_idx() if feature_idx is None else feature_idx
        feature_dim = self._classifier_in_features()
        self.social_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, cfg["dataset"]["num_classes"]),
        ).to(device)

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
        classifier = self._local_classifier()
        if not hasattr(classifier, "weight"):
            return
        final_layer = self.social_head[-1]
        with torch.no_grad():
            final_layer.weight.copy_(classifier.weight)
            if final_layer.bias is not None and getattr(classifier, "bias", None) is not None:
                final_layer.bias.copy_(classifier.bias)

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

    def train_social_head_and_backbone(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self._backbone().parameters():
            param.requires_grad_(True)
        for param in self._local_classifier().parameters():
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
