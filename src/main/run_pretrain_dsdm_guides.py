"""Pretrain strict DSDM guide checkpoints for each sender agent."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.training.v2_train_utils import SyntheticCIFARDataset, get_new_classes, train_one_epoch
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_paths import get_v2_dsdm_guide_dir
from src.utils.v2_runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain v2 strict DSDM guide checkpoint pools.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    parser.add_argument("--max-guides", type=int, default=None, help="Optional smoke-test guide cap.")
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
    parser.add_argument("--skip-existing", action="store_true", help="Skip guide checkpoints that already exist.")
    return parser.parse_args()


def make_loader(agent_dataset, batch_size: int, num_workers: int, device: torch.device, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        agent_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def train_guide(agent_id: int, guide_id: int, cfg: dict, agent_dataset, expert_classes, device: torch.device, args):
    train_cfg = cfg["train"]
    dataset_cfg = cfg["dataset"]
    guide_seed = cfg["seed"] + agent_id * 1000 + guide_id
    set_seed(guide_seed)

    guide_dir = get_v2_dsdm_guide_dir(cfg, agent_id)
    guide_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = guide_dir / f"guide_{guide_id}.pt"
    if args.skip_existing and ckpt_path.exists():
        print(f"skip existing: {ckpt_path}")
        return ckpt_path

    model = build_agent_model(cfg, agent_id, device)
    model_name = model.model_name
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg.get("lr", 0.01),
        momentum=0.9,
        weight_decay=5e-4,
    )

    epochs = cfg["dsdm"]["guide_pretrain_epochs"]
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    loader = make_loader(
        agent_dataset=agent_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=args.num_workers,
        device=device,
        seed=guide_seed,
    )

    print(f"\n=== pretrain DSDM guide agent_{agent_id} guide_{guide_id} ===")
    print(f"model_name: {model_name}")
    print(f"guide_seed: {guide_seed}")
    print(f"expert_classes: {expert_classes}")
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
        print(
            f"agent_{agent_id} guide_{guide_id} epoch {epoch + 1}/{epochs}: "
            f"loss={final_loss:.4f} acc={final_acc:.4f}"
        )

    torch.save(
        {
            "agent_id": agent_id,
            "guide_id": guide_id,
            "expert_classes": list(expert_classes),
            "new_classes": get_new_classes(dataset_cfg["num_classes"], list(expert_classes)),
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "stage": "dsdm_guide_pretrain",
            "guide_seed": guide_seed,
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


def pretrain_agent_guides(agent_id: int, cfg: dict, train_dataset, class_splits, device: torch.device, args):
    dsdm_cfg = cfg["dsdm"]
    dataset_cfg = cfg["dataset"]
    expert_classes = list(class_splits[agent_id])
    agent_dataset = subset_by_classes(train_dataset, expert_classes)

    guide_count = dsdm_cfg["guide_model_number"]
    if args.max_guides is not None:
        guide_count = min(guide_count, args.max_guides)
    if guide_count <= 0:
        raise ValueError(f"guide_count must be positive, got {guide_count}")

    print(f"\n=== agent_{agent_id} DSDM guide pool ===")
    print(f"expert_classes: {expert_classes}")
    print(f"new_classes: {get_new_classes(dataset_cfg['num_classes'], expert_classes)}")
    print(f"guide_count: {guide_count}")

    for guide_id in range(guide_count):
        train_guide(agent_id, guide_id, cfg, agent_dataset, expert_classes, device, args)


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
        raise NotImplementedError("run_pretrain_dsdm_guides supports split.mode=direct only")

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

    print("=== run_pretrain_dsdm_guides ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"device: {device}")
    print(f"dataset_root: {dataset_cfg['root']}")
    print(f"download: {not args.no_download}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    print(f"guide_model_number: {cfg['dsdm']['guide_model_number']}")
    print(f"guide_pretrain_epochs: {cfg['dsdm']['guide_pretrain_epochs']}")
    if args.smoke_synthetic_samples is not None:
        print(f"smoke_synthetic_samples: {args.smoke_synthetic_samples}")

    for agent_id in selected_agent_ids:
        pretrain_agent_guides(agent_id, cfg, train_dataset, class_splits, device, args)


if __name__ == "__main__":
    main()
