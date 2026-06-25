import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from agent_data import get_num_classes, get_run_dir, get_test_dataset, get_train_dataset
from config_adapter import build_dsdm_args_from_config, load_config


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def parse_cli():
    """解析 centralized full data 训练参数。"""
    parser = argparse.ArgumentParser(description="完整真实数据 centralized upper bound")
    parser.add_argument("--config", default="configs/main_cifar100.yaml", help="主配置文件路径")
    parser.add_argument("--model", required=True, help="模型名，可为 convnet/resnet 或配置中的 convnet3w05 等实验模型名")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--dry-run", action="store_true", help="只打印 centralized 训练配置")
    return parser.parse_args()


def _centralized_train_cfg(cfg, model_name):
    """读取 centralized 默认配置，并合并指定模型的 recipe 覆盖。"""
    base = dict(cfg.get("centralized_full", {}))
    recipes = base.pop("recipes", {}) or base.pop("models", {}) or {}
    model_cfg = recipes.get(model_name, {})
    base.update(model_cfg)
    return base


def _prepare_args(cli):
    """读取配置并覆盖 centralized 训练参数。"""
    cfg = load_config(cli.config)
    args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    model_cfg = cfg.get("model_pool", {}).get("models", {}).get(cli.model, {})
    train_cfg = _centralized_train_cfg(cfg, cli.model)
    args.model_name = cli.model
    args.net_type = str(model_cfg.get("family", cli.model))
    args.depth = int(model_cfg.get("depth", args.depth))
    args.width = float(model_cfg.get("width", args.width))
    args.norm_type = str(model_cfg.get("norm_type", args.norm_type))
    args.num_classes = get_num_classes(args)
    args.nclass = args.num_classes
    args.epochs = int(cli.epochs if cli.epochs is not None else train_cfg.get("epochs", args.epochs))
    args.lr = float(cli.lr if cli.lr is not None else train_cfg.get("lr", args.lr))
    args.batch_size = int(cli.batch_size if cli.batch_size is not None else train_cfg.get("batch_size", args.batch_size))
    args.momentum = float(train_cfg.get("momentum", args.momentum))
    args.weight_decay = float(train_cfg.get("weight_decay", args.weight_decay))
    args.train_augment = bool(train_cfg.get("augment", False))
    args.use_dsdm_train = bool(train_cfg.get("use_dsdm_train", False))
    args.scheduler = str(train_cfg.get("scheduler", "none"))
    args.scheduler_gamma = float(train_cfg.get("scheduler_gamma", 0.1))
    args.scheduler_milestones = [int(v) for v in train_cfg.get("scheduler_milestones", [])]
    args.recipe_name = str(train_cfg.get("recipe_name", "legacy"))
    _refresh_model_metadata(args)
    return args




def _refresh_model_metadata(args):
    """根据实际模型配置刷新 DSDM 模型标签。"""
    if args.net_type == "convnet":
        args.f_idx = str(args.depth - 1)
    args.datatag = f"{args.dataset}"
    if args.net_type == "resnet_ap":
        args.modeltag = f"resnet{args.depth}ap"
    elif args.net_type == "convnet":
        args.modeltag = f"conv{args.depth}"
    else:
        args.modeltag = f"{args.net_type}{args.depth}"
    if args.norm_type == "instance":
        args.modeltag += "in"
    if args.width != 1.0:
        args.modeltag += f"_w{args.width}"

def _build_model(args, device):
    """构建 centralized full data 模型。"""
    from train import define_model

    return define_model(args, get_num_classes(args)).to(device)


def _build_scheduler(args, optimizer):
    """按配置构建 centralized 训练学习率调度器。"""
    if args.scheduler == "multistep":
        milestones = args.scheduler_milestones or [2 * int(args.epochs) // 3, 5 * int(args.epochs) // 6]
        args.scheduler_milestones = milestones
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.scheduler_gamma)
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    return None


def _build_dsdm_aug(args):
    """构建 DSDM 原训练使用的 DiffAug 增强。"""
    if not bool(getattr(args, "use_dsdm_train", False)) or not bool(getattr(args, "dsa", False)):
        return None
    from misc.augment import DiffAug

    return DiffAug(strategy=args.dsa_strategy, batch=False)


def _accuracy(model, loader, device):
    """计算完整测试集 top-1 accuracy。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(1, total)


def _train_one_epoch(args, model, train_loader, criterion, optimizer, device, aug):
    """执行一个 epoch，支持 DSDM 原 cutmix 和 DiffAug。"""
    from misc.utils import rand_bbox, random_indices

    model.train()
    last_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        if aug is not None:
            with torch.no_grad():
                images = aug(images)
        optimizer.zero_grad()
        if bool(getattr(args, "use_dsdm_train", False)) and args.mixup == "cut" and np.random.rand(1) < args.mix_p:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(labels, nclass=args.nclass)
            labels_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            logits = model(images)
            loss = criterion(logits, labels) * ratio + criterion(logits, labels_b) * (1.0 - ratio)
        else:
            loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def _output_paths(args):
    """返回 centralized 输出目录和文件路径，recipe 结果独立保存。"""
    suffix = "" if args.recipe_name in {"", "legacy"} else f"_{args.recipe_name}"
    model_dir_name = f"{args.model_name}{suffix}"
    base_dir = get_run_dir(args) / "centralized_full" / model_dir_name
    summary_name = "centralized_results.csv" if not suffix else f"centralized_results_{args.recipe_name}.csv"
    summary_path = get_run_dir(args) / "centralized_full" / summary_name
    return base_dir, base_dir / "metrics.json", base_dir / "checkpoint.pt", summary_path


def _append_result(path, metrics):
    """追加 centralized full data 汇总结果。"""
    fields = [
        "run_name", "dataset", "model", "recipe_name", "depth", "width", "norm_type",
        "epochs", "lr", "batch_size", "momentum", "weight_decay", "augment",
        "use_dsdm_train", "mixup", "mix_p", "dsa", "dsa_strategy", "scheduler",
        "scheduler_milestones", "scheduler_gamma", "num_classes", "train_size", "test_size",
        "train_loss", "acc_global", "checkpoint_path", "time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})


def train_centralized_full(args):
    """使用完整真实训练集训练 centralized upper bound。"""
    if int(getattr(args, "seed", -1)) >= 0:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(args.seed))
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_set = get_train_dataset(args, normalize=True, augment=args.train_augment)
    test_set = get_test_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = _build_model(args, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()
    aug = _build_dsdm_aug(args)
    last_loss = 0.0
    for epoch in range(int(args.epochs)):
        last_loss = _train_one_epoch(args, model, train_loader, criterion, optimizer, device, aug)
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[centralized_full] epoch={epoch + 1}/{args.epochs} loss={last_loss:.4f} lr={cur_lr:.6f}")
    acc = _accuracy(model, test_loader, device)
    base_dir, metrics_path, checkpoint_path, summary_path = _output_paths(args)
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    metrics = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "model": args.model_name,
        "recipe_name": args.recipe_name,
        "depth": int(args.depth),
        "width": float(args.width),
        "norm_type": args.norm_type,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "augment": bool(args.train_augment),
        "use_dsdm_train": bool(args.use_dsdm_train),
        "mixup": args.mixup,
        "mix_p": float(args.mix_p),
        "dsa": bool(args.dsa),
        "dsa_strategy": args.dsa_strategy,
        "scheduler": args.scheduler,
        "scheduler_milestones": ",".join(str(v) for v in args.scheduler_milestones),
        "scheduler_gamma": float(args.scheduler_gamma),
        "num_classes": int(get_num_classes(args)),
        "train_size": len(train_set),
        "test_size": len(test_set),
        "train_loss": last_loss,
        "acc_global": acc,
        "checkpoint_path": str(checkpoint_path),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
    _append_result(summary_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    """运行 centralized full data upper bound 入口。"""
    cli = parse_cli()
    args = _prepare_args(cli)
    if cli.dry_run:
        print(json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True))
        return
    train_centralized_full(args)


if __name__ == "__main__":
    main()
