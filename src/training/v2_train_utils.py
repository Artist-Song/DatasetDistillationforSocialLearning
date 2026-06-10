"""Small shared training helpers for v2 entry points."""

from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SyntheticCIFARDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int, image_size):
        self.targets = [idx % num_classes for idx in range(num_samples)]
        height, width = image_size
        self.images = torch.rand(num_samples, 3, height, width)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def get_new_classes(num_classes: int, expert_classes: List[int]) -> List[int]:
    expert_set = set(expert_classes)
    return [class_id for class_id in range(num_classes) if class_id not in expert_set]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    if total_seen == 0:
        raise RuntimeError("no training batches were processed")

    return total_loss / total_seen, total_correct / total_seen
