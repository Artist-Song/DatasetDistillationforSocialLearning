"""
Lightweight image distillation utilities for social packets.

This is a small anchor-guided placeholder before integrating full DSDM. It
starts from real local samples and optimizes the images so the frozen sender
anchor classifies them with the intended labels.
"""

from torch.utils.data import Subset
import torch
import torch.nn as nn


def select_ipc_indices(dataset, class_ids, ipc: int):
    selected = []
    counts = {class_id: 0 for class_id in class_ids}

    for idx, target in enumerate(dataset.targets):
        if target in counts and counts[target] < ipc:
            selected.append(idx)
            counts[target] += 1
        if all(count == ipc for count in counts.values()):
            break

    missing = {class_id: ipc - count for class_id, count in counts.items() if count < ipc}
    if missing:
        raise RuntimeError(f"not enough samples for ipc={ipc}: {missing}")

    return selected


def stack_subset_samples(dataset, indices):
    images = []
    labels = []
    subset = Subset(dataset, indices)
    for image, label in subset:
        images.append(image)
        labels.append(label)

    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def build_raw_images(dataset, class_ids, ipc: int):
    indices = select_ipc_indices(dataset, class_ids, ipc)
    return stack_subset_samples(dataset, indices)


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    loss_h = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    loss_w = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def distill_images_with_anchor(
    anchor_model,
    init_images: torch.Tensor,
    hard_labels: torch.Tensor,
    steps: int,
    lr: float,
    tv_weight: float,
    device: torch.device,
):
    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad_(False)

    images = init_images.clone().to(device).requires_grad_(True)
    labels = hard_labels.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([images], lr=lr)

    last_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = anchor_model(images)
        loss = criterion(logits, labels)
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation_loss(images)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
        last_loss = float(loss.detach().cpu())

    return images.detach().cpu(), {"distill_final_loss": last_loss}