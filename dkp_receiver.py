"""Utilities for the opt-in DKP-SL receiver protocol."""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset


class CompletePaddedSampler(Sampler):
    """Visit every row, then repeat seeded rows so all batches stay full."""

    def __init__(self, size, batch_size, shuffle, generator=None):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.padded_size = math.ceil(self.size / self.batch_size) * self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.size, generator=self.generator).tolist()
        else:
            indices = list(range(self.size))
        padding = self.padded_size - self.size
        if padding:
            repeats = math.ceil(padding / self.size)
            indices.extend((indices * repeats)[:padding])
        return iter(indices)

    def __len__(self):
        return self.padded_size


def build_complete_balanced_loader(
    images,
    labels,
    *extra_tensors,
    batch_size=64,
    shuffle=True,
    generator=None,
    pad_to_full_batch=False,
):
    """Build a loader that visits every item once and is balanced over its classes."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same number of rows")
    if images.shape[0] == 0:
        raise ValueError("cannot build a DKP stream from an empty tensor")
    for tensor in extra_tensors:
        if tensor.shape[0] != images.shape[0]:
            raise ValueError("all DKP stream tensors must have the same number of rows")

    class_ids, counts = torch.unique(labels.long(), sorted=True, return_counts=True)
    if class_ids.numel() == 0 or not torch.all(counts == counts[0]):
        count_map = {int(class_id): int(count) for class_id, count in zip(class_ids, counts)}
        raise ValueError(f"DKP stream is not class-balanced: {count_map}")

    dataset = TensorDataset(images.float(), labels.long(), *extra_tensors)
    sampler = None
    if pad_to_full_batch:
        sampler = CompletePaddedSampler(
            len(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        drop_last=bool(pad_to_full_batch),
        num_workers=0,
        generator=generator if sampler is None else None,
    )


class CyclingLoader:
    """Cycle a finite loader, recreating its iterator to reshuffle each pass."""

    def __init__(self, loader):
        if len(loader) == 0:
            raise ValueError("cannot cycle an empty loader")
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def supervised_contrastive_loss(view1_features, view2_features, labels, temperature=0.07):
    """SupCon over two views, with same-image and same-label views as positives."""
    if view1_features.shape != view2_features.shape:
        raise ValueError("SupCon views must have identical feature shapes")
    if view1_features.ndim != 2:
        raise ValueError("SupCon expects flattened [batch, feature] tensors")
    if labels.ndim != 1 or labels.shape[0] != view1_features.shape[0]:
        raise ValueError("SupCon labels must have one row per input image")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("SupCon temperature must be finite and positive")

    features = F.normalize(torch.cat([view1_features, view2_features], dim=0), dim=1)
    repeated_labels = labels.long().repeat(2)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    sample_count = logits.shape[0]
    non_self = ~torch.eye(sample_count, dtype=torch.bool, device=logits.device)
    positive_mask = repeated_labels[:, None].eq(repeated_labels[None, :]) & non_self
    log_denominator = torch.logsumexp(logits.masked_fill(~non_self, float("-inf")), dim=1)
    log_probability = logits - log_denominator[:, None]
    positive_count = positive_mask.sum(dim=1)
    if (positive_count == 0).any():
        raise ValueError("every SupCon anchor must have at least one positive")
    mean_positive_log_probability = (
        log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count
    )
    return -mean_positive_log_probability.mean()


def mean_loss_totals(totals, steps):
    """Convert accumulated scalar losses to step means."""
    if int(steps) <= 0:
        raise ValueError("optimizer step count must be positive")
    return {name: float(value) / int(steps) for name, value in totals.items()}
