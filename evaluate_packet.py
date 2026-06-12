import argparse
import os
import sys
from pathlib import Path

import torch

from config_adapter import build_dsdm_args_from_config, load_config
from output_manager import append_global_result, get_packet_path, prepare_output_dirs, save_metrics


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


def _decode_dsdm_packet(args, packet):
    """按 DSDM multi-formation 规则解码 DSDM packet。"""
    if packet.get("source") != "dsdm" or args.factor <= 1:
        return packet

    from test import decode_fn

    images = packet["images"]
    labels = packet["labels"]
    data_dec = []
    target_dec = []
    class_ids = packet.get("class_ids", sorted({int(x) for x in labels.tolist()}))
    for class_id in class_ids:
        mask = labels == int(class_id)
        data_c = images[mask].detach()
        target_c = labels[mask].detach()
        data_c, target_c = decode_fn(
            data_c,
            target_c,
            args.factor,
            args.decode_type,
            bound=args.batch_syn_max,
        )
        data_dec.append(data_c)
        target_dec.append(target_c)

    decoded = dict(packet)
    decoded["images"] = torch.cat(data_dec).cpu()
    decoded["labels"] = torch.cat(target_dec).cpu()
    decoded["meta"] = dict(packet.get("meta", {}))
    decoded["meta"]["decoded_for_eval"] = True
    decoded["meta"]["raw_num_images"] = int(images.shape[0])
    decoded["meta"]["eval_num_images"] = int(decoded["images"].shape[0])
    return decoded


def _build_packet_loader(args, packet):
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
    dataset = TensorDataset(packet["images"], packet["labels"], train_transform)
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
    packet = _decode_dsdm_packet(args, packet)
    train_loader = _build_packet_loader(args, packet)
    val_loader = _build_val_loader(args)
    best_acc = float(test_data(args, train_loader, val_loader, repeat=args.repeat, test_resnet=False))
    metrics = {
        "source": packet_source,
        "best_acc": best_acc,
        "last_acc": best_acc,
        "train_acc": 0.0,
        "class_ids": packet.get("class_ids", []),
        "ipc": packet.get("ipc", args.ipc),
        "num_images": int(packet["images"].shape[0]),
        "decoded_for_eval": bool(packet.get("meta", {}).get("decoded_for_eval", False)),
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
