"""Standalone DSDM reproduction check for CIFAR-10 IPC experiments.

This entry point intentionally avoids multi-agent/social-learning code paths.
It trains full-CIFAR10 ConvNet guides, distills one 10-class synthetic set, and
evaluates a fresh ConvNet only on those distilled images.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from src.datasets.cifar import build_cifar_test_dataset, build_cifar_train_dataset
from src.distill.v2_strict_dsdm import distill_images_with_strict_dsdm, freeze_guide_pool
from src.models.agent_model import build_agent_model
from src.training.v2_train_utils import train_one_epoch
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_runtime import resolve_device


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone DSDM reproduction check.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--stage", type=str, choices=["all", "pretrain_guides", "distill", "eval"], default="all")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-guide-epochs", type=int, default=None, help="Smoke-test override only.")
    parser.add_argument("--max-distill-steps", type=int, default=None, help="Smoke-test override only.")
    parser.add_argument("--eval-epochs", type=int, default=1500)
    parser.add_argument("--max-eval-epochs", type=int, default=None, help="Smoke-test override only.")
    return parser.parse_args()


def standalone_root(cfg: dict, ipc: int) -> Path:
    return Path(cfg["output"]["root"]) / f"dsdm_standalone_cifar10_ipc{ipc}"


def guide_dir(cfg: dict, ipc: int) -> Path:
    return standalone_root(cfg, ipc) / "checkpoints" / "full_guides"


def synthetic_path(cfg: dict, ipc: int) -> Path:
    return standalone_root(cfg, ipc) / "synthetic" / f"standalone_dsdm_ipc{ipc}.pt"


def visual_dir(cfg: dict, ipc: int) -> Path:
    return standalone_root(cfg, ipc) / "visuals"


def metrics_path(cfg: dict, ipc: int) -> Path:
    return standalone_root(cfg, ipc) / "metrics" / "dsdm_standalone_eval.json"


def evaluator_checkpoint_path(cfg: dict, ipc: int) -> Path:
    return standalone_root(cfg, ipc) / "checkpoints" / "evaluator_conv.pt"


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def conv_probe_cfg(cfg: dict) -> dict:
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["agent_models"] = {0: "conv"}
    probe_cfg.setdefault("model", {})
    probe_cfg["model"]["norm_type"] = "instance"
    probe_cfg["model"]["head_type"] = cfg.get("model", {}).get("head_type", "shallow_mlp")
    probe_cfg["dataset"]["num_classes"] = 10
    return probe_cfg


def build_dataset(cfg: dict, train: bool, download: bool):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if train:
        return build_cifar_train_dataset(dataset_cfg["name"], dataset_cfg["root"], image_size=image_size, download=download)
    return build_cifar_test_dataset(dataset_cfg["name"], dataset_cfg["root"], image_size=image_size, download=download)


def train_guide_model(
    guide_id: int,
    cfg: dict,
    train_dataset,
    device: torch.device,
    epochs: int,
    num_workers: int,
    seed: int,
):
    set_seed(seed + guide_id)
    model = build_agent_model(conv_probe_cfg(cfg), agent_id=0, device=device)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"].get("batch_size", 128),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    final_loss = 0.0
    final_acc = 0.0
    with StageTimer(f"standalone guide_{guide_id} pretrain"):
        epoch_bar = progress(range(epochs), desc=f"guide_{guide_id} epochs", leave=False)
        for _epoch in epoch_bar:
            final_loss, final_acc = train_one_epoch(model, loader, criterion, optimizer, device)
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{final_loss:.4f}", acc=f"{final_acc:.4f}")

    out_dir = guide_dir(cfg, cfg["standalone_ipc"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"guide_{guide_id}.pt"
    torch.save(
        {
            "stage": "dsdm_standalone_full_cifar10_guide",
            "guide_id": guide_id,
            "model_name": "conv",
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "final_loss": float(final_loss),
            "final_acc": float(final_acc),
            "cfg": cfg,
        },
        ckpt_path,
    )
    print(f"saved guide: {ckpt_path}")
    return ckpt_path


def pretrain_guides(cfg: dict, train_dataset, device: torch.device, args):
    guide_count = 10
    epochs = 20 if args.max_guide_epochs is None else min(20, args.max_guide_epochs)
    out_dir = guide_dir(cfg, args.ipc)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("guide_*.pt"))
    if len(existing) >= guide_count and args.max_guide_epochs is None:
        print(f"reuse existing standalone full guides: {out_dir}")
        return existing[:guide_count]

    guide_paths = []
    for guide_id in progress(range(guide_count), desc="standalone full guides"):
        ckpt_path = out_dir / f"guide_{guide_id}.pt"
        if ckpt_path.exists() and args.max_guide_epochs is None:
            print(f"reuse guide: {ckpt_path}")
            guide_paths.append(ckpt_path)
            continue
        guide_paths.append(
            train_guide_model(
                guide_id=guide_id,
                cfg=cfg,
                train_dataset=train_dataset,
                device=device,
                epochs=epochs,
                num_workers=args.num_workers,
                seed=args.seed if args.seed is not None else cfg["seed"],
            )
        )
    return guide_paths


def load_guide_models(cfg: dict, ipc: int, device: torch.device):
    models = []
    paths = []
    for guide_id in range(10):
        ckpt_path = guide_dir(cfg, ipc) / f"guide_{guide_id}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing standalone guide checkpoint: {ckpt_path}")
        model = build_agent_model(conv_probe_cfg(cfg), agent_id=0, device=device)
        ckpt = torch_load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        models.append(model)
        paths.append(str(ckpt_path))
    freeze_guide_pool(models)
    return models, paths


def standalone_dsdm_packet_cfg(ipc: int, max_steps: int = None) -> dict:
    steps = 10000 if max_steps is None else min(10000, max_steps)
    return {
        "ipc": ipc,
        "distill_steps": steps,
        "distill_lr": 0.1,
        "mom_img": 0.5,
        "batch_real": 256,
        "batch_syn_max": 256,
        "dsdm_init": "mix",
        "factor": 2,
        "decode_type": "single",
        "aug_type": "color_crop_cutout",
        "idx_from": 2,
        "idx_to": -1,
        "metric": "mse",
        "smooth_iter": 2000,
        "cov_weight": 50.0,
        "h_p_weight": 0.2,
        "smooth_factor": 0.99,
        "progress_desc": "standalone strict DSDM",
    }


def save_standalone_visuals(cfg: dict, ipc: int, images: torch.Tensor, labels: torch.Tensor):
    out_dir = visual_dir(cfg, ipc)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"standalone_dsdm_ipc{ipc}_grid.png"
    save_image(images.clamp(0.0, 1.0), grid_path, nrow=ipc)
    for class_id in range(10):
        class_path = out_dir / f"class_{class_id}.png"
        save_image(images[labels == class_id].clamp(0.0, 1.0), class_path, nrow=ipc)
    print(f"saved visuals: {grid_path}")


def distill_standalone(cfg: dict, train_dataset, device: torch.device, args):
    guide_models, guide_paths = load_guide_models(cfg, args.ipc, device)
    packet_cfg = standalone_dsdm_packet_cfg(args.ipc, max_steps=args.max_distill_steps)
    class_ids = list(range(10))

    with StageTimer("standalone DSDM distillation"):
        images, labels, meta = distill_images_with_strict_dsdm(
            guide_models=guide_models,
            train_dataset=train_dataset,
            class_ids=class_ids,
            packet_cfg=packet_cfg,
            device=device,
        )

    out_path = synthetic_path(cfg, args.ipc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "hard_labels": labels,
        "class_ids": class_ids,
        "guide_checkpoint_paths": guide_paths,
        "meta": {
            **meta,
            "ipc": args.ipc,
            "guide_count": len(guide_models),
            "standalone_reproduction": True,
        },
    }
    torch.save(payload, out_path)
    save_standalone_visuals(cfg, args.ipc, images, labels)
    print(f"saved standalone synthetic: {out_path}")
    return out_path


class AugmentedTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def evaluator_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def evaluator_test_transform():
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def build_normalized_test_dataset(cfg: dict, download: bool):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    base = build_cifar_test_dataset(dataset_cfg["name"], dataset_cfg["root"], image_size=image_size, download=download)
    base.transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return base


@torch.no_grad()
def evaluate(model, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(10, dtype=torch.long)
    class_total = torch.zeros(10, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        batch_correct = preds == labels
        correct += batch_correct.sum().item()
        total += labels.size(0)
        for class_id in range(10):
            mask = labels == class_id
            class_total[class_id] += mask.sum().cpu()
            class_correct[class_id] += batch_correct[mask].sum().cpu()
    per_class = {
        str(class_id): (
            class_correct[class_id].item() / class_total[class_id].item() if class_total[class_id].item() else 0.0
        )
        for class_id in range(10)
    }
    return correct / total if total else 0.0, per_class


@torch.no_grad()
def synthetic_train_accuracy(model, images: torch.Tensor, labels: torch.Tensor, device: torch.device):
    dataset = AugmentedTensorDataset(images, labels, transform=evaluator_test_transform())
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    acc, _per_class = evaluate(model, loader, device)
    return acc


def eval_standalone(cfg: dict, test_dataset, device: torch.device, args):
    syn_path = synthetic_path(cfg, args.ipc)
    if not syn_path.exists():
        raise FileNotFoundError(f"missing standalone synthetic set: {syn_path}")
    payload = torch_load(syn_path, map_location="cpu")
    images = payload["images"].float()
    labels = payload["hard_labels"].long()
    meta = payload.get("meta", {})
    if images.size(0) != args.ipc * 10:
        raise RuntimeError(f"expected {args.ipc * 10} synthetic images, got {images.size(0)}")

    epochs = args.eval_epochs if args.max_eval_epochs is None else min(args.eval_epochs, args.max_eval_epochs)
    train_dataset = AugmentedTensorDataset(images, labels, transform=evaluator_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_agent_model(conv_probe_cfg(cfg), agent_id=0, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2 * epochs // 3, 5 * epochs // 6],
        gamma=0.2,
    )

    best_accuracy = 0.0
    last_accuracy = 0.0
    best_state = None
    with StageTimer("standalone DSDM evaluator"):
        epoch_bar = progress(range(epochs), desc="standalone evaluator epochs", leave=False)
        for _epoch in epoch_bar:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            last_accuracy, per_class_accuracy = evaluate(model, test_loader, device)
            if last_accuracy > best_accuracy:
                best_accuracy = last_accuracy
                best_state = copy.deepcopy(model.state_dict())
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(train=f"{train_acc:.4f}", test=f"{last_accuracy:.4f}", best=f"{best_accuracy:.4f}")

    train_acc_syn = synthetic_train_accuracy(model, images, labels, device)
    metrics = {
        "ipc": args.ipc,
        "guide_count": meta.get("guide_count", 10),
        "distill_steps": meta.get("distill_steps", 10000),
        "distill_lr": meta.get("distill_lr", 0.1),
        "factor": meta.get("factor", 2),
        "init": meta.get("dsdm_init", "mix"),
        "idx_from": meta.get("idx_from", 2),
        "idx_to": meta.get("idx_to", -1),
        "metric": meta.get("metric", "mse"),
        "evaluator_epochs": epochs,
        "evaluator_best_accuracy": best_accuracy,
        "evaluator_last_accuracy": last_accuracy,
        "train_accuracy_on_synthetic": train_acc_syn,
        "per_class_accuracy": per_class_accuracy,
    }
    out_path = metrics_path(cfg, args.ipc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ckpt_path = evaluator_checkpoint_path(cfg, args.ipc)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_state,
            "metrics": metrics,
            "cfg": cfg,
        },
        ckpt_path,
    )
    print(f"evaluator_best_accuracy: {best_accuracy:.4f}")
    print(f"evaluator_last_accuracy: {last_accuracy:.4f}")
    print(f"train_accuracy_on_synthetic: {train_acc_syn:.4f}")
    print(f"saved metrics: {out_path}")
    return metrics


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["standalone_ipc"] = args.ipc
    if args.dataset_root is not None:
        cfg["dataset"]["root"] = args.dataset_root
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = resolve_device(cfg.get("device", "cpu"))

    print("=== run_dsdm_standalone_v2 ===")
    print(f"config: {args.config}")
    print(f"output_root: {standalone_root(cfg, args.ipc)}")
    print(f"stage: {args.stage}")
    print(f"ipc: {args.ipc}")
    print("DSDM config: factor=2 init=mix distill_lr=0.1 idx_from=2 idx_to=-1 metric=mse")
    print(f"device: {device}")

    train_dataset = None
    test_dataset = None
    if args.stage in {"all", "pretrain_guides", "distill"}:
        train_dataset = build_dataset(cfg, train=True, download=not args.no_download)
    if args.stage in {"all", "eval"}:
        test_dataset = build_normalized_test_dataset(cfg, download=not args.no_download)

    if args.stage in {"all", "pretrain_guides"}:
        pretrain_guides(cfg, train_dataset, device, args)
    if args.stage in {"all", "distill"}:
        if train_dataset is None:
            train_dataset = build_dataset(cfg, train=True, download=not args.no_download)
        distill_standalone(cfg, train_dataset, device, args)
    if args.stage in {"all", "eval"}:
        if test_dataset is None:
            test_dataset = build_normalized_test_dataset(cfg, download=not args.no_download)
        eval_standalone(cfg, test_dataset, device, args)


if __name__ == "__main__":
    main()
