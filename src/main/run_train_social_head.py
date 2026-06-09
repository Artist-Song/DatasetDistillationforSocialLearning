"""
Train specialist social heads with class-balanced real and packet samples.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.datasets.cifar import build_cifar_train_dataset
from src.datasets.splits import get_partial_split
from src.distill.simple_distiller import select_n_per_class_indices, stack_subset_samples
from src.main.run_local_pretrain import resolve_device
from src.models.social_head_model import SocialHeadAgent
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


def load_packet_for_class(packet_dir: Path, class_id: int):
    packet_path = packet_dir / f"class_{class_id}_packet.pt"
    if not packet_path.exists():
        raise FileNotFoundError(f"packet not found: {packet_path}")
    return torch.load(packet_path, map_location="cpu", weights_only=False)


def build_balanced_social_dataset(train_dataset, known_classes, missing_classes, packet_dir: Path, k_real: int):
    real_indices = select_n_per_class_indices(train_dataset, known_classes, k_real)
    real_images, real_labels = stack_subset_samples(train_dataset, real_indices)

    packet_images = []
    packet_labels = []
    packet_soft_targets = []
    for class_id in missing_classes:
        packet = load_packet_for_class(packet_dir, class_id)
        class_mask = packet.hard_labels == class_id
        images = packet.images[class_mask][:k_real]
        labels = packet.hard_labels[class_mask][:k_real]
        soft_targets = packet.soft_targets[class_mask][:k_real]
        if images.size(0) < k_real:
            raise RuntimeError(f"not enough packet samples for class {class_id}: {images.size(0)} < {k_real}")
        packet_images.append(images)
        packet_labels.append(labels)
        packet_soft_targets.append(soft_targets)

    missing_images = torch.cat(packet_images, dim=0)
    missing_labels = torch.cat(packet_labels, dim=0)
    missing_soft_targets = torch.cat(packet_soft_targets, dim=0)

    images = torch.cat([real_images, missing_images], dim=0)
    labels = torch.cat([real_labels, missing_labels], dim=0)
    soft_targets = torch.zeros(images.size(0), missing_soft_targets.size(1), dtype=missing_soft_targets.dtype)
    is_packet = torch.zeros(images.size(0), dtype=torch.bool)
    soft_targets[real_images.size(0) :] = missing_soft_targets
    is_packet[real_images.size(0) :] = True
    return TensorDataset(images, labels, soft_targets, is_packet)


def kd_loss(logits, soft_targets, temperature: float):
    log_probs = nn.functional.log_softmax(logits / temperature, dim=1)
    return nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature * temperature)


def train_one_agent(agent_id, split, cfg, train_dataset, device, packet_dir: Path, save_dir: Path):
    social_cfg = cfg.get("social_head", cfg.get("social", {}))
    k_real = social_cfg.get("k_per_class", cfg.get("packet", {}).get("ipc", 10))
    dataset = build_balanced_social_dataset(train_dataset, split.known, split.missing, packet_dir, k_real)
    loader = DataLoader(
        dataset,
        batch_size=social_cfg.get("batch_size", cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    local_ckpt_path = Path(cfg["output"]["root"]) / "checkpoints" / "specialists" / f"agent_{agent_id}_specialist.pt"
    if not local_ckpt_path.exists():
        raise FileNotFoundError(f"specialist checkpoint not found: {local_ckpt_path}")
    local_ckpt = torch.load(local_ckpt_path, map_location=device)

    model = SocialHeadAgent(cfg, device=device, feature_idx=social_cfg.get("feature_idx"))
    model.load_local_state_dict(local_ckpt["model_state_dict"])
    model.init_social_head_from_local()
    model.freeze_backbone()
    model.freeze_local_head()
    model.train_social_head_only()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=social_cfg.get("lr", cfg["train"]["lr"]),
        momentum=0.9,
        weight_decay=social_cfg.get("weight_decay", 5e-4),
    )
    ce = nn.CrossEntropyLoss()
    temperature = cfg.get("packet", {}).get("temperature", 2.0)
    lambda_kd = social_cfg.get("lambda_kd", 1.0)
    epochs = social_cfg.get("epochs", cfg["train"]["epochs"])

    print(f"\n=== train social head agent_{agent_id} ===")
    print(f"known_classes: {split.known}")
    print(f"missing_classes: {split.missing}")
    print(f"k_per_class: {k_real}")
    print(f"samples: {len(dataset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        progress = tqdm(loader, desc=f"agent_{agent_id} social_head epoch {epoch + 1}/{epochs}")
        for images, labels, soft_targets, is_packet in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            is_packet = is_packet.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, head="social")
            loss = ce(logits, labels)
            if is_packet.any() and lambda_kd > 0:
                loss = loss + lambda_kd * kd_loss(logits[is_packet], soft_targets[is_packet], temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
            progress.set_postfix(loss=f"{total_loss / total_seen:.4f}")

    save_path = save_dir / f"agent_{agent_id}_social_head.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "known_classes": split.known,
            "missing_classes": split.missing,
            "model_state_dict": model.state_dict(),
            "local_model_state_dict": model.local_model.state_dict(),
            "cfg": cfg,
        },
        save_path,
    )
    print(f"saved: {save_path}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    splits = get_partial_split(cfg["split"]["name"])
    selected_agent_ids = parse_agent_ids(args.agent_ids, len(splits))
    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    packet_source = cfg.get("packet", {}).get("source", "global_raw")
    packet_dir = Path(cfg["output"]["root"]) / "packets" / "generalist" / packet_source
    save_dir = Path(cfg["output"]["root"]) / "checkpoints" / "social_head"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_train_social_head ===")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    print(f"packet_dir: {packet_dir}")

    for agent_id in selected_agent_ids:
        train_one_agent(agent_id, splits[agent_id], cfg, train_dataset, device, packet_dir, save_dir)


if __name__ == "__main__":
    main()
