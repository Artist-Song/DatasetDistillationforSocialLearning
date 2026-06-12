import argparse
import os
import sys
from pathlib import Path

import torch

from config_adapter import build_dsdm_args_from_config, load_config
from output_manager import append_global_result, get_packet_path, prepare_output_dirs, save_metrics
from packet_consumer import consume_packet_for_training


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def parse_cli():
    """解析 packet 评估入口命令行参数。"""
    parser = argparse.ArgumentParser(description="评估统一格式 packet")
    parser.add_argument("--config", default="configs/main.yaml", help="主配置文件路径")
    parser.add_argument(
        "--packet-source",
        required=True,
        choices=["dsdm", "heuristic", "importance"],
        help="待评估的 packet 来源",
    )
    return parser.parse_args()


def _build_val_loader(args):
    """复用 DSDM 数据加载逻辑构建验证集 loader。"""
    from DSDM import load_resized_data
    from data import MultiEpochsDataLoader

    _, val_loader = load_resized_data(args)
    if isinstance(val_loader, MultiEpochsDataLoader):
        return val_loader
    return val_loader


def _build_packet_loader(args, images, labels):
    """把 packet 图片和标签转换为 DSDM test_data 可用的 loader。"""
    from data import MultiEpochsDataLoader, TensorDataset
    from data import transform_cifar, transform_fashion, transform_mnist, transform_svhn

    transform_map = {
        "cifar10": transform_cifar,
        "cifar100": transform_cifar,
        "svhn": transform_svhn,
        "mnist": transform_mnist,
        "fashion": transform_fashion,
    }
    if args.dataset not in transform_map:
        raise ValueError(f"暂不支持该数据集的 packet 评估: {args.dataset}")
    train_transform, _ = transform_map[args.dataset](augment=args.augment, from_tensor=True)
    dataset = TensorDataset(images, labels, train_transform)
    return MultiEpochsDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers if args.augment else 0,
        persistent_workers=args.augment and args.workers > 0,
    )


def evaluate_packet(args, packet_source):
    """读取 packet 并复用 DSDM 验证协议完成评估。"""
    from test import test_data

    packet_path = get_packet_path(args, packet_source)
    if not packet_path.exists():
        raise FileNotFoundError(f"packet 不存在: {packet_path}")
    packet = torch.load(packet_path, map_location="cpu")
    consumed = consume_packet_for_training(args, packet_path)
    train_loader = _build_packet_loader(args, consumed["images"], consumed["labels"])
    val_loader = _build_val_loader(args)
    best_acc = float(test_data(args, train_loader, val_loader, repeat=args.repeat, test_resnet=False))
    metrics = {
        "source": packet_source,
        "best_acc": best_acc,
        "last_acc": best_acc,
        "train_acc": 0.0,
        "class_ids": packet.get("class_ids", []),
        "ipc": packet.get("ipc", args.ipc),
        "num_images": int(consumed["num_images"]),
        "decoded_for_eval": bool(consumed["decoded_for_training"]),
    }
    save_metrics(args, f"packet_eval_{packet_source}", metrics)
    append_global_result(
        args,
        {
            "stage": "packet_eval",
            "method": packet.get("meta", {}).get("method", packet_source),
            "source": packet_source,
            "best_acc": best_acc,
            "last_acc": best_acc,
            "train_acc": 0.0,
            "packet_path": str(packet_path),
        },
    )
    return metrics


def main():
    """读取配置并启动指定 packet 的评估。"""
    cli = parse_cli()
    cfg = load_config(cli.config)
    args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    prepare_output_dirs(args)
    metrics = evaluate_packet(args, cli.packet_source)
    print(metrics)


if __name__ == "__main__":
    main()
