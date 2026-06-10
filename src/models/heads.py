"""
Configurable classifier heads for v2 agent models.
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ShallowMLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head(head_type: str, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
    if head_type == "linear":
        return LinearHead(in_dim, num_classes)
    if head_type == "shallow_mlp":
        return ShallowMLPHead(in_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if head_type == "deep_mlp":
        return DeepMLPHead(in_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"unknown head_type: {head_type}")
