"""
Train specialist social heads with class-balanced real and packet samples.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets.cifar import build_cifar_train_dataset
from src.datasets.splits import class_indices, get_partial_split
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


def resolve_packet_source(source: str) -> str:
    if source == "global_raw":
        return "global_raw_packet"
    if source == "global_raw_packet":
        return source
    raise NotImplementedError("social head training currently supports only global_raw_packet")


def resolve_packet_dir(cfg):
    source = cfg.get("packet", {}).get("source", "global_raw_packet")
    canonical_source = resolve_packet_source(source)
    packet_root = Path(cfg["output"]["root"]) / "packets" / "generalist"
    packet_dir = packet_root / canonical_source
    if not packet_dir.exists() and source != canonical_source:
        alias_dir = packet_root / source
        if alias_dir.exists():
            return alias_dir, canonical_source
    return packet_dir, canonical_source


class ClassBalancedStepSampler:
    def __init__(self, train_dataset, known_classes, missing_classes, packet_dir: Path, samples_per_class: int):
        self.train_dataset = train_dataset
        self.known_classes = list(known_classes)
        self.missing_classes = list(missing_classes)
        self.samples_per_class = samples_per_class
        self.known_indices = class_indices(train_dataset, self.known_classes)
        self.packet_by_class = {}
        for class_id in self.missing_classes:
            packet = load_packet_for_class(packet_dir, class_id)
            class_mask = packet.hard_labels == class_id
            images = packet.images[class_mask]
            labels = packet.hard_labels[class_mask]
            soft_targets = packet.soft_targets[class_mask]
            if images.size(0) == 0:
                raise RuntimeError(f"packet has no samples for class {class_id}")
            self.packet_by_class[class_id] = (images, labels, soft_targets)

    def _sample_known_class(self, class_id: int):
        indices = self.known_indices[class_id]
        if len(indices) < self.samples_per_class:
            raise RuntimeError(f"not enough known real samples for class {class_id}")
        perm = torch.randperm(len(indices))[: self.samples_per_class].tolist()
        images = []
        labels = []
        for pos in perm:
            image, label = self.train_dataset[indices[pos]]
            images.append(image)
            labels.append(label)
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

    def _sample_packet_class(self, class_id: int):
        images, labels, soft_targets = self.packet_by_class[class_id]
        selected = torch.randint(0, images.size(0), (self.samples_per_class,))
        return images[selected], labels[selected], soft_targets[selected]

    def sample(self):
        known_images = []
        known_labels = []
        for class_id in self.known_classes:
            images, labels = self._sample_known_class(class_id)
            known_images.append(images)
            known_labels.append(labels)

        packet_images = []
        packet_labels = []
        packet_soft_targets = []
        for class_id in self.missing_classes:
            images, labels, soft_targets = self._sample_packet_class(class_id)
            packet_images.append(images)
            packet_labels.append(labels)
            packet_soft_targets.append(soft_targets)

        return {
            "known_images": torch.cat(known_images, dim=0),
            "known_labels": torch.cat(known_labels, dim=0),
            "packet_images": torch.cat(packet_images, dim=0),
            "packet_labels": torch.cat(packet_labels, dim=0),
            "packet_soft_targets": torch.cat(packet_soft_targets, dim=0),
        }


def kd_loss(logits, soft_targets, temperature: float):
    log_probs = nn.functional.log_softmax(logits / temperature, dim=1)
    return nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature * temperature)


def train_one_agent(agent_id, split, cfg, train_dataset, device, packet_dir: Path, save_dir: Path):
    social_cfg = cfg.get("social_head", cfg.get("social", {}))
    samples_per_class = social_cfg.get("samples_per_class", social_cfg.get("k_per_class", 4))
    steps_per_epoch = social_cfg.get("steps_per_epoch", 100)
    sampler = ClassBalancedStepSampler(train_dataset, split.known, split.missing, packet_dir, samples_per_class)

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
    lambda_packet_ce = social_cfg.get("lambda_packet_ce", 1.0)
    lambda_packet_kd = social_cfg.get("lambda_packet_kd", social_cfg.get("lambda_kd", 1.0))
    lambda_known_ce = social_cfg.get("lambda_known_ce", 1.0)
    lambda_retain = social_cfg.get("lambda_retain", 1.0)
    epochs = social_cfg.get("epochs", cfg["train"]["epochs"])

    print(f"\n=== train social head agent_{agent_id} ===")
    print(f"known_classes: {split.known}")
    print(f"missing_classes: {split.missing}")
    print(f"samples_per_class: {samples_per_class}")
    print(f"steps_per_epoch: {steps_per_epoch}")

    for epoch in range(epochs):
        model.train()
        model.local_model.eval()
        total_loss = 0.0
        progress = tqdm(range(steps_per_epoch), desc=f"agent_{agent_id} social_head epoch {epoch + 1}/{epochs}")
        for _ in progress:
            batch = sampler.sample()
            known_images = batch["known_images"].to(device, non_blocking=True)
            known_labels = batch["known_labels"].to(device, non_blocking=True)
            packet_images = batch["packet_images"].to(device, non_blocking=True)
            packet_labels = batch["packet_labels"].to(device, non_blocking=True)
            packet_soft_targets = batch["packet_soft_targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            known_logits = model(known_images, head="social")
            packet_logits = model(packet_images, head="social")
            loss_known_ce = ce(known_logits, known_labels)
            loss_packet_ce = ce(packet_logits, packet_labels)
            loss_packet_kd = kd_loss(packet_logits, packet_soft_targets, temperature)
            with torch.no_grad():
                local_teacher_logits = model(known_images, head="local")
            loss_retain = nn.functional.mse_loss(known_logits, local_teacher_logits)
            loss = (
                lambda_known_ce * loss_known_ce
                + lambda_packet_ce * loss_packet_ce
                + lambda_packet_kd * loss_packet_kd
                + lambda_retain * loss_retain
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{total_loss / (progress.n + 1):.4f}",
                known_ce=f"{loss_known_ce.item():.4f}",
                packet_ce=f"{loss_packet_ce.item():.4f}",
                packet_kd=f"{loss_packet_kd.item():.4f}",
                retain=f"{loss_retain.item():.4f}",
            )

    save_path = save_dir / f"agent_{agent_id}_social_head.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "known_classes": split.known,
            "missing_classes": split.missing,
            "stage": "social_head",
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
    packet_dir, packet_source = resolve_packet_dir(cfg)
    save_dir = Path(cfg["output"]["root"]) / "checkpoints" / "social_head"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_train_social_head ===")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    print(f"packet_dir: {packet_dir}")
    print(f"packet_source: {packet_source}")

    for agent_id in selected_agent_ids:
        train_one_agent(agent_id, splits[agent_id], cfg, train_dataset, device, packet_dir, save_dir)


if __name__ == "__main__":
    main()
