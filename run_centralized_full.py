import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

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
    parser.add_argument("--model", required=True, choices=["convnet", "resnet", "resnet_ap"], help="模型类型")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    return parser.parse_args()


def _prepare_args(cli):
    """读取配置并覆盖 centralized 训练参数。"""
    cfg = load_config(cli.config)
    args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    args.net_type = cli.model
    if args.net_type in {"resnet", "resnet_ap"}:
        args.depth = 10
    args.num_classes = get_num_classes(args)
    args.nclass = args.num_classes
    args.epochs = int(cli.epochs if cli.epochs is not None else args.epochs)
    args.lr = float(cli.lr if cli.lr is not None else args.lr)
    args.batch_size = int(cli.batch_size if cli.batch_size is not None else args.batch_size)
    return args


def _build_model(args, device):
    """构建 centralized full data 模型。"""
    from train import define_model

    return define_model(args, get_num_classes(args)).to(device)


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


def _output_paths(args):
    """返回 centralized 输出目录和文件路径。"""
    base_dir = get_run_dir(args) / "centralized_full" / args.net_type
    summary_path = get_run_dir(args) / "centralized_full" / "centralized_results.csv"
    return base_dir, base_dir / "metrics.json", base_dir / "checkpoint.pt", summary_path


def _append_result(path, metrics):
    """追加 centralized full data 汇总结果。"""
    fields = [
        "run_name",
        "dataset",
        "model",
        "epochs",
        "lr",
        "batch_size",
        "num_classes",
        "train_size",
        "test_size",
        "train_loss",
        "acc_global",
        "checkpoint_path",
        "time",
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
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_set = get_train_dataset(args, normalize=True)
    test_set = get_test_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = _build_model(args, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    for epoch in range(int(args.epochs)):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        print(f"[centralized_full] epoch={epoch + 1}/{args.epochs} loss={last_loss:.4f}")
    acc = _accuracy(model, test_loader, device)
    base_dir, metrics_path, checkpoint_path, summary_path = _output_paths(args)
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    metrics = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "model": args.net_type,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
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
    train_centralized_full(args)


if __name__ == "__main__":
    main()
