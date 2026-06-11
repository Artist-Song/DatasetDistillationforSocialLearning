"""Packet-only probe for v2 hard-label image packets.

This script trains a fresh model only on generated packets and evaluates it on
the full CIFAR test split. It does not load expert/socialized checkpoints and
does not perform agent-to-agent social learning.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.cifar import build_cifar_test_dataset
from src.models.agent_model import build_agent_model
from src.packet.packet_dataclass import SocialPacket
from src.training.v2_train_utils import SyntheticCIFARDataset
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_paths import (
    get_v2_packet_dir,
    get_v2_packet_probe_checkpoint_dir,
    get_v2_packet_probe_metrics_dir,
)
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_runtime import resolve_device


FORBIDDEN_PACKET_FIELDS = {
    "soft_targets",
    "teacher_logits",
    "teacher_probs",
    "teacher_probabilities",
    "logits",
    "probs",
    "probabilities",
    "gradients",
    "model_state_dict",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fresh packet-only probe model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], required=True)
    parser.add_argument("--probe-model", type=str, choices=["conv", "resnet", "resnet_ap"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--smoke-synthetic-samples", type=int, default=None)
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_no_forbidden_packet_payload(packet: SocialPacket, packet_path: Path) -> None:
    packet_fields = set(vars(packet).keys())
    forbidden_fields = sorted(packet_fields.intersection(FORBIDDEN_PACKET_FIELDS))
    forbidden_meta = sorted(set((packet.meta or {}).keys()).intersection(FORBIDDEN_PACKET_FIELDS))
    if forbidden_fields or forbidden_meta:
        raise RuntimeError(
            f"packet contains forbidden soft/model payload at {packet_path}: "
            f"fields={forbidden_fields}, meta={forbidden_meta}"
        )


def load_packets(cfg: dict, packet_source: str) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    split_cfg = cfg["split"]
    num_agents = split_cfg["num_agents"]
    packet_dir = get_v2_packet_dir(cfg, packet_source)
    images_list = []
    labels_list = []
    packet_summaries = []

    for agent_id in range(num_agents):
        packet_path = packet_dir / f"agent_{agent_id}_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"missing packet for agent_{agent_id}: {packet_path}")
        packet: SocialPacket = torch_load(packet_path, map_location="cpu")
        assert_no_forbidden_packet_payload(packet, packet_path)

        images = packet.images.detach().float().cpu()
        labels = packet.hard_labels.detach().long().cpu()
        if images.size(0) != labels.size(0):
            raise RuntimeError(f"packet image/label count mismatch at {packet_path}")
        images_list.append(images)
        labels_list.append(labels)

        summary = {
            "agent_id": agent_id,
            "path": str(packet_path),
            "sender_id": int(packet.sender_id),
            "class_ids": [int(class_id) for class_id in list(packet.class_ids)],
            "num_images": int(images.size(0)),
            "meta": dict(packet.meta or {}),
        }
        if packet_source == "strict_dsdm":
            meta = summary["meta"]
            print(
                f"strict_dsdm packet agent_{agent_id}: "
                f"ipc={meta.get('ipc')} guide_count={meta.get('guide_count')} "
                f"distill_steps={meta.get('distill_steps')} idx_from={meta.get('idx_from')} "
                f"idx_to={meta.get('idx_to')} metric={meta.get('metric')}"
            )
        packet_summaries.append(summary)

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return images, labels, packet_summaries


def validate_label_coverage(labels: torch.Tensor, num_classes: int) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels.tolist())
    class_counts = {str(class_id): int(counts.get(class_id, 0)) for class_id in range(num_classes)}
    missing_classes = [class_id for class_id in range(num_classes) if class_counts[str(class_id)] == 0]
    if missing_classes:
        raise RuntimeError(f"packet labels do not cover all classes; missing: {missing_classes}")
    return class_counts


def make_probe_model_cfg(cfg: dict, probe_model: str) -> dict:
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["agent_models"] = {0: probe_model}
    probe_cfg.setdefault("model", {})
    return probe_cfg


def build_test_dataset(cfg: dict, args):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if args.smoke_synthetic_samples is not None:
        return SyntheticCIFARDataset(args.smoke_synthetic_samples, dataset_cfg["num_classes"], image_size)
    return build_cifar_test_dataset(
        name=dataset_cfg["name"],
        root=dataset_cfg["root"],
        image_size=image_size,
        download=not args.no_download,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in loader:
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
        raise RuntimeError("no packet probe training batches were processed")
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes: int):
    model.eval()
    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct = preds == labels
        total_correct += correct.sum().item()
        total_seen += labels.size(0)
        for class_id in range(num_classes):
            mask = labels == class_id
            class_total[class_id] += mask.sum().cpu()
            class_correct[class_id] += correct[mask].sum().cpu()

    per_class_accuracy = {}
    for class_id in range(num_classes):
        if class_total[class_id].item() == 0:
            per_class_accuracy[str(class_id)] = 0.0
        else:
            per_class_accuracy[str(class_id)] = class_correct[class_id].item() / class_total[class_id].item()
    overall_accuracy = total_correct / total_seen if total_seen else 0.0
    average_class_accuracy = sum(per_class_accuracy.values()) / num_classes
    return overall_accuracy, average_class_accuracy, per_class_accuracy


@torch.no_grad()
def train_accuracy_on_packets(model, images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    overall, _avg_class, _per_class = evaluate_loader(model, loader, device, int(labels.max().item()) + 1)
    return overall


def metrics_path(cfg: dict, packet_source: str, probe_model: str) -> Path:
    return get_v2_packet_probe_metrics_dir(cfg) / f"{packet_source}_{probe_model}.json"


def checkpoint_path(cfg: dict, packet_source: str, probe_model: str) -> Path:
    return get_v2_packet_probe_checkpoint_dir(cfg) / f"{packet_source}_{probe_model}.pt"


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.dataset_root is not None:
        cfg["dataset"]["root"] = args.dataset_root
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = resolve_device(cfg.get("device", "cpu"))
    dataset_cfg = cfg["dataset"]
    num_classes = dataset_cfg["num_classes"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr if args.lr is not None else cfg["train"].get("lr", 0.01)

    print("=== run_packet_probe_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"packet_source: {args.packet_source}")
    print(f"probe_model: {args.probe_model}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"device: {device}")
    print(f"seed: {seed}")

    images, labels, packet_summaries = load_packets(cfg, args.packet_source)
    class_counts = validate_label_coverage(labels, num_classes)
    print(f"loaded packet images: {tuple(images.shape)}")
    print(f"loaded packet labels: {tuple(labels.shape)}")
    print(f"class_counts: {class_counts}")

    train_dataset = TensorDataset(images, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    probe_cfg = make_probe_model_cfg(cfg, args.probe_model)
    model = build_agent_model(probe_cfg, agent_id=0, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    final_train_loss = None
    final_train_acc = None
    with StageTimer("packet probe training"):
        epoch_bar = progress(range(args.epochs), desc="packet probe epochs", leave=False)
        for _epoch in epoch_bar:
            final_train_loss, final_train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{final_train_loss:.4f}", acc=f"{final_train_acc:.4f}")
    print(f"packet probe final train: loss={final_train_loss:.4f} acc={final_train_acc:.4f}")

    test_dataset = build_test_dataset(cfg, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    overall_accuracy, average_class_accuracy, per_class_accuracy = evaluate_loader(model, test_loader, device, num_classes)
    packet_train_accuracy = train_accuracy_on_packets(model, images, labels, batch_size, device)

    metrics = {
        "experiment": cfg["experiment"]["name"],
        "packet_source": args.packet_source,
        "probe_model": args.probe_model,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "num_packet_images": int(images.size(0)),
        "class_counts": class_counts,
        "overall_accuracy": overall_accuracy,
        "average_class_accuracy": average_class_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "train_accuracy_on_packets": packet_train_accuracy,
        "final_train_loss": float(final_train_loss),
        "final_train_accuracy_epoch": float(final_train_acc),
        "packet_summaries": packet_summaries,
    }

    out_path = metrics_path(cfg, args.packet_source, args.probe_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"overall_accuracy: {overall_accuracy:.4f}")
    print(f"average_class_accuracy: {average_class_accuracy:.4f}")
    print(f"train_accuracy_on_packets: {packet_train_accuracy:.4f}")
    print(f"saved metrics: {out_path}")

    if not args.no_save_checkpoint:
        ckpt_path = checkpoint_path(cfg, args.packet_source, args.probe_model)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "packet_source": args.packet_source,
                "probe_model": args.probe_model,
                "cfg": cfg,
                "metrics_path": str(out_path),
                "metrics": metrics,
            },
            ckpt_path,
        )
        print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
