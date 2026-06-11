"""Train v2 expert agents on their own direct-split classes."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.training.v2_train_utils import SyntheticCIFARDataset, get_new_classes, train_one_epoch
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.v2_progress import StageTimer, progress
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
        "--dataset-root",
        type=str,
        default=None,
        help="Override cfg.dataset.root; point this to the parent of cifar-10-batches-py.",
    )
    parser.add_argument(
        "--smoke-synthetic-samples",
        type=int,
        default=None,
        help="Use a tiny synthetic CIFAR-shaped dataset for local smoke tests only.",
    )
    return parser.parse_args()


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
    with StageTimer(f"train expert agent_{agent_id}"):
        epoch_bar = progress(range(epochs), desc=f"agent_{agent_id} expert epochs", leave=False)
        for _epoch in epoch_bar:
            final_loss, final_acc = train_one_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=args.max_batches,
            )
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{final_loss:.4f}", acc=f"{final_acc:.4f}")
    print(f"agent_{agent_id} expert final: loss={final_loss:.4f} acc={final_acc:.4f}")

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
    if args.dataset_root is not None:
        dataset_cfg["root"] = args.dataset_root

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
    print(f"dataset_root: {dataset_cfg['root']}")
    print(f"download: {not args.no_download}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    if args.smoke_synthetic_samples is not None:
        print(f"smoke_synthetic_samples: {args.smoke_synthetic_samples}")

    with StageTimer("run_train_agents_v2 total"):
        for agent_id in progress(selected_agent_ids, desc="expert agents"):
            train_agent(agent_id, cfg, train_dataset, class_splits, device, args)


if __name__ == "__main__":
    main()
