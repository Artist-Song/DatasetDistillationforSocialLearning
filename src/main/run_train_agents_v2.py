"""Train v2 expert agents on their own direct-split classes."""

import argparse
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_paths import get_v2_agent_checkpoint_dir
from src.utils.v2_runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train v2 expert agents on direct-split expert classes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional smoke-test epoch cap.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test batch cap per epoch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true", help="Disable CIFAR download.")
    parser.add_argument(
        "--smoke-synthetic-samples",
        type=int,
        default=None,
        help="Use a tiny synthetic CIFAR-shaped dataset for local smoke tests only.",
    )
    return parser.parse_args()


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


def train_agent(agent_id: int, cfg: dict, train_dataset, class_splits, device: torch.device, args):
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    num_classes = dataset_cfg["num_classes"]
    expert_classes = list(class_splits[agent_id])
    new_classes = get_new_classes(num_classes, expert_classes)
    agent_dataset = subset_by_classes(train_dataset, expert_classes)

    loader = DataLoader(
        agent_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_agent_model(cfg, agent_id, device)
    model_name = model.model_name
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=0.9,
        weight_decay=5e-4,
    )

    epochs = train_cfg["epochs"]
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    print(f"\n=== train expert agent_{agent_id} ===")
    print(f"model_name: {model_name}")
    print(f"expert_classes: {expert_classes}")
    print(f"new_classes: {new_classes}")
    print(f"samples: {len(agent_dataset)}")
    print(f"epochs: {epochs}")

    final_loss = None
    final_acc = None
    for epoch in range(epochs):
        final_loss, final_acc = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_batches,
        )
        print(f"agent_{agent_id} epoch {epoch + 1}/{epochs}: loss={final_loss:.4f} acc={final_acc:.4f}")

    ckpt_dir = get_v2_agent_checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"agent_{agent_id}_expert.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "expert_classes": expert_classes,
            "new_classes": new_classes,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "stage": "expert_train",
            "train_summary": {
                "epochs": epochs,
                "final_loss": float(final_loss),
                "final_acc": float(final_acc),
            },
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    split_cfg = cfg["split"]
    dataset_cfg = cfg["dataset"]

    if split_cfg["mode"] != "direct":
        raise NotImplementedError("run_train_agents_v2 supports split.mode=direct only")

    selected_agent_ids = parse_agent_ids(args.agent_ids, split_cfg["num_agents"])
    class_splits = make_direct_class_splits(
        num_classes=dataset_cfg["num_classes"],
        num_agents=split_cfg["num_agents"],
        classes_per_agent=split_cfg["classes_per_agent"],
    )
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if args.smoke_synthetic_samples is None:
        train_dataset = build_cifar_train_dataset(
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            image_size=image_size,
            download=not args.no_download,
        )
    else:
        train_dataset = SyntheticCIFARDataset(
            num_samples=args.smoke_synthetic_samples,
            num_classes=dataset_cfg["num_classes"],
            image_size=image_size,
        )

    print("=== run_train_agents_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    if args.smoke_synthetic_samples is not None:
        print(f"smoke_synthetic_samples: {args.smoke_synthetic_samples}")

    for agent_id in selected_agent_ids:
        train_agent(agent_id, cfg, train_dataset, class_splits, device, args)


if __name__ == "__main__":
    main()
