#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DSDM.models.alexnet_cifar import alexnet_cifar
from DSDM.models.convnet import ConvNet
from DSDM.models.resnet import ResNet
from output_manager import atomic_torch_save
from tiny_imagenet_data import TinyImageNetDataset, build_tiny_imagenet_transforms


METRIC_FIELDS = [
    "epoch", "lr", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5",
    "clean_val_loss", "clean_val_top1", "clean_val_top5",
    "val_class_min", "val_class_q25", "epoch_seconds", "images_per_second", "peak_gpu_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-ImageNet centralized backbone validation")
    parser.add_argument(
        "--model",
        required=True,
        choices=["convnet4", "resnet18", "alexnet", "mobilenetv2"],
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--integrity-report",
        default="outputs/tinyimagenet_data_validation_20260718/data_integrity.json",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name == "convnet4":
        return ConvNet(
            200,
            net_norm="instance",
            net_depth=4,
            net_width=192,
            channel=3,
            im_size=(64, 64),
        )
    if name == "resnet18":
        return ResNet("tinyimagenet", 18, 200, norm_type="batch", size=64, nch=3)
    if name == "alexnet":
        return alexnet_cifar(200, nch=3)
    if name == "mobilenetv2":
        return mobilenet_v2(weights=None, num_classes=200)
    raise ValueError(f"Unsupported model: {name}")


def learning_rate(args: argparse.Namespace, epoch: int) -> float:
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        return args.lr * float(epoch + 1) / float(args.warmup_epochs)
    remaining = max(1, args.epochs - args.warmup_epochs)
    progress = float(epoch - args.warmup_epochs) / float(remaining)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def topk_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    prediction = logits.topk(5, dim=1).indices
    correct = prediction.eq(targets.view(-1, 1))
    return int(correct[:, :1].sum().item()), int(correct.any(dim=1).sum().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
) -> dict:
    model.train()
    loss_sum = 0.0
    top1_sum = 0
    count = 0
    start = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_count = int(targets.numel())
        top1, _ = topk_counts(logits.detach(), targets)
        loss_sum += float(loss.detach().item()) * batch_count
        top1_sum += top1
        count += batch_count
    seconds = time.monotonic() - start
    return {
        "loss": loss_sum / max(1, count),
        "top1": 100.0 * top1_sum / max(1, count),
        "count": count,
        "seconds": seconds,
        "images_per_second": count / max(seconds, 1e-9),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
) -> dict:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0
    top5_sum = 0
    count = 0
    class_correct = torch.zeros(200, dtype=torch.long)
    class_total = torch.zeros(200, dtype=torch.long)
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_count = int(targets.numel())
        top1, top5 = topk_counts(logits, targets)
        predictions = logits.argmax(dim=1)
        for class_id in targets.unique():
            mask = targets == class_id
            cid = int(class_id.item())
            class_total[cid] += int(mask.sum().item())
            class_correct[cid] += int((predictions[mask] == targets[mask]).sum().item())
        loss_sum += float(loss.item()) * batch_count
        top1_sum += top1
        top5_sum += top5
        count += batch_count
    valid = class_total > 0
    per_class = 100.0 * class_correct[valid].float() / class_total[valid].float()
    return {
        "loss": loss_sum / max(1, count),
        "top1": 100.0 * top1_sum / max(1, count),
        "top5": 100.0 * top5_sum / max(1, count),
        "class_min": float(per_class.min().item()),
        "class_q25": float(torch.quantile(per_class, 0.25).item()),
        "per_class": per_class.tolist(),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tiny-ImageNet backbone validation")

    integrity_path = ROOT / args.integrity_report
    if not integrity_path.is_file():
        raise FileNotFoundError(f"Tiny-ImageNet integrity report is missing: {integrity_path}")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("passed"):
        raise RuntimeError(f"Tiny-ImageNet integrity report did not pass: {integrity_path}")

    seed_everything(args.seed)
    device = torch.device("cuda")
    run_dir = ROOT / args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = TinyImageNetDataset(
        ROOT / args.data_root,
        "train",
        transform=build_tiny_imagenet_transforms(train=True),
    )
    val_dataset = TinyImageNetDataset(
        ROOT / args.data_root,
        "val",
        transform=build_tiny_imagenet_transforms(train=False),
    )
    clean_val_dataset = TinyImageNetDataset(
        ROOT / args.data_root,
        "val",
        transform=build_tiny_imagenet_transforms(train=False),
    )
    leaked_val_paths = {str(Path(path).resolve()) for path in integrity["exact_duplicates"]["leaked_validation_paths"]}
    clean_val_dataset.samples = [
        (path, target)
        for path, target in clean_val_dataset.samples
        if str(path.resolve()) not in leaked_val_paths
    ]
    clean_val_dataset.targets = [target for _, target in clean_val_dataset.samples]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    clean_val_loader = DataLoader(
        clean_val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = build_model(args.model).to(device, memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    amp_enabled = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    metrics_path = run_dir / "metrics.csv"
    last_path = run_dir / "checkpoint_last.pt"
    best_path = run_dir / "checkpoint_best.pt"
    start_epoch = 0
    best_top1 = -1.0
    rows: list[dict] = []
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_top1 = float(checkpoint["best_top1"])
        rows = list(checkpoint.get("metrics", []))

    config = vars(args).copy()
    config.update(
        {
            "parameters": parameter_count,
            "num_classes": 200,
            "image_size": 64,
            "train_images": len(train_dataset),
            "val_images": len(val_dataset),
            "clean_val_images": len(clean_val_dataset),
            "excluded_conflicting_val_duplicates": len(val_dataset) - len(clean_val_dataset),
            "global_label_mapping": "wnids.txt order",
            "device": torch.cuda.get_device_name(0),
        }
    )
    atomic_write_json(run_dir / "config.json", config)
    print(json.dumps({"event": "start", **config}, ensure_ascii=False), flush=True)

    total_start = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(args, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.reset_peak_memory_stats()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled, args.max_train_batches
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, amp_enabled, args.max_val_batches
        )
        clean_val_metrics = evaluate(
            model, clean_val_loader, criterion, device, amp_enabled, args.max_val_batches
        )
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "clean_val_loss": clean_val_metrics["loss"],
            "clean_val_top1": clean_val_metrics["top1"],
            "clean_val_top5": clean_val_metrics["top5"],
            "val_class_min": val_metrics["class_min"],
            "val_class_q25": val_metrics["class_q25"],
            "epoch_seconds": train_metrics["seconds"],
            "images_per_second": train_metrics["images_per_second"],
            "peak_gpu_mb": peak_gpu_mb,
        }
        rows.append(row)
        write_metrics(metrics_path, rows)
        is_best = val_metrics["top1"] > best_top1
        best_top1 = max(best_top1, val_metrics["top1"])
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "metrics": rows,
            "config": config,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(checkpoint, best_path)
            atomic_write_json(run_dir / "best_per_class.json", {"epoch": epoch + 1, "accuracy": val_metrics["per_class"]})
        print(json.dumps({"event": "epoch", "model": args.model, **row, "best_top1": best_top1}), flush=True)

    summary = {
        "status": "complete",
        "run_name": args.run_name,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "parameters": parameter_count,
        "best_val_top1": best_top1,
        "last": rows[-1],
        "total_seconds": time.monotonic() - total_start,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
