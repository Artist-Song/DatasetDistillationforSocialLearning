"""
跨架构 packet 评估脚本

功能：对指定的 dsdm_packet.pt 文件，使用不同 receiver 架构
从头训练并评估，输出跨架构迁移准确率矩阵。

用法：
  python scripts/run_cross_arch_packet_eval.py \
    --packet-paths <packet1.pt> <packet2.pt> ... \
    --packet-labels <label1> <label2> ... \
    --class-start 0 --class-end 24 \
    --dataset cifar100 \
    --out-csv outputs/experiment_registry/cross_arch_eval.csv \
    --gpu 0
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── 架构定义 ──────────────────────────────────────────────────────────────────

ARCH_CONFIGS = {
    "convnet3w1": dict(net_type="convnet", depth=3, width=1.0, norm_type="instance"),
    "convnet4w15": dict(net_type="convnet", depth=4, width=1.5, norm_type="instance"),
    "vgg":         dict(net_type="vgg",     depth=None, width=None, norm_type="batch"),
    "alexnet":     dict(net_type="alexnet", depth=None, width=None, norm_type="batch"),
    "resnet10":    dict(net_type="resnet",  depth=10, width=None, norm_type="batch"),
    "resnet18":    dict(net_type="resnet",  depth=18, width=None, norm_type="batch"),
}

ARCH_DISPLAY = {
    "convnet3w1": "ConvNet-3",
    "convnet4w15": "ConvNet-4",
    "vgg":         "VGG11",
    "alexnet":     "AlexNet",
    "resnet10":    "ResNet-10",
    "resnet18":    "ResNet-18",
}


def build_model(arch_key, num_classes, device):
    """根据 arch_key 构建模型并移至 device。"""
    cfg = ARCH_CONFIGS[arch_key]
    nch = 3
    if cfg["net_type"] == "convnet":
        from models.convnet import ConvNet
        width = int(cfg["width"] * 128)
        # ConvNet 的 net_norm 参数接受字符串，不接受类对象
        norm_str = cfg["norm_type"]  # "instance" or "batch"
        model = ConvNet(
            channel=nch, num_classes=num_classes,
            net_width=width, net_depth=cfg["depth"],
            net_act="relu", net_norm=norm_str, net_pooling="avgpooling",
        )
    elif cfg["net_type"] == "vgg":
        from models.vgg_cifar import vgg_cifar
        model = vgg_cifar(num_classes, nch=nch)
    elif cfg["net_type"] == "alexnet":
        from models.alexnet_cifar import alexnet_cifar
        model = alexnet_cifar(num_classes, nch=nch)
    elif cfg["net_type"] == "resnet":
        from models.resnet import ResNet
        model = ResNet(
            dataset="cifar100", depth=cfg["depth"],
            num_classes=num_classes, norm_type="batch", nch=nch,
        )
    else:
        raise ValueError(f"未知 net_type: {cfg['net_type']}")
    return model.to(device)


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_packet_tensors(packet_path):
    """加载 packet 文件，返回 (images, labels)。"""
    pkt = torch.load(packet_path, map_location="cpu", weights_only=False)
    images = pkt.get("images")
    if images is None:
        images = pkt.get("data")
    labels = pkt.get("labels")
    if images is None or labels is None:
        raise ValueError(f"packet 缺少 images/labels: {packet_path}")
    # 如果是 factor>1 的 raw，直接解码
    if "decode_type" in pkt and pkt["decode_type"] == "single" and images.shape[-1] != 32:
        # 尝试 reshape
        factor = pkt.get("factor", 1)
        if factor > 1:
            B, C, H, W = images.shape
            images = images.view(B * factor * factor, C, H // factor, W // factor)
            labels = labels.repeat_interleave(factor * factor)
    return images.float(), labels.long()


def get_cifar100_test_loader(class_ids, batch_size=256):
    """加载 CIFAR-100 测试集，过滤指定 class_ids。"""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409],
                             [0.2673, 0.2564, 0.2762]),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=str(ROOT / "data"), train=False, download=True, transform=transform
    )
    # 过滤并重映射标签
    class_set = set(class_ids)
    class_remap = {c: i for i, c in enumerate(sorted(class_ids))}
    indices = [i for i, (_, c) in enumerate(dataset) if c in class_set]
    subset_imgs = torch.stack([dataset[i][0] for i in indices])
    subset_labels = torch.tensor([class_remap[dataset[i][1]] for i in indices])
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader(TensorDataset(subset_imgs, subset_labels),
                      batch_size=batch_size, shuffle=False, num_workers=4)


# ─── 训练与评估 ────────────────────────────────────────────────────────────────

def train_and_eval(model, train_images, train_labels, val_loader,
                   epochs=300, lr=0.01, device="cuda"):
    """在 packet 图像上训练，返回测试集最佳准确率。"""
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.transforms as transforms

    # 简单数据增强
    def augment_batch(imgs):
        b, c, h, w = imgs.shape
        pad = 4
        padded = torch.zeros(b, c, h + 2 * pad, w + 2 * pad)
        padded[:, :, pad:pad+h, pad:pad+w] = imgs
        # random crop
        tops = torch.randint(0, 2 * pad, (b,))
        lefts = torch.randint(0, 2 * pad, (b,))
        crops = torch.stack([padded[i, :, t:t+h, l:l+w]
                             for i, (t, l) in enumerate(zip(tops, lefts))])
        # random hflip
        flip_mask = torch.rand(b) > 0.5
        crops[flip_mask] = crops[flip_mask].flip(-1)
        return crops

    # 归一化
    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std

    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True, drop_last=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 2/3), int(epochs * 5/6)], gamma=0.2
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs = augment_batch(imgs).to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 每50 epoch 评估一次
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs).argmax(1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc = acc
            print(f"  epoch {epoch+1}/{epochs} acc={acc:.2f}% best={best_acc:.2f}%", flush=True)

    return best_acc


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--packet-paths", nargs="+", required=True)
    p.add_argument("--packet-labels", nargs="+", required=True)
    p.add_argument("--class-ids", nargs="+", type=int, required=True,
                   help="packet 对应的全局 CIFAR-100 class ids")
    p.add_argument("--archs", nargs="+", default=list(ARCH_CONFIGS.keys()),
                   help="要评估的 receiver 架构列表")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out-csv", default="outputs/experiment_registry/cross_arch_eval.csv")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    assert len(args.packet_paths) == len(args.packet_labels), \
        "--packet-paths 和 --packet-labels 数量不匹配"

    num_classes = len(args.class_ids)
    val_loader = get_cifar100_test_loader(args.class_ids)
    print(f"class_ids={args.class_ids}, num_classes={num_classes}")

    if args.dry_run:
        print("[dry-run] 配置检查通过，退出。")
        for pl, pp in zip(args.packet_labels, args.packet_paths):
            print(f"  packet: {pl} @ {pp}")
        for a in args.archs:
            print(f"  receiver: {a}")
        return

    results = []
    for p_label, p_path in zip(args.packet_labels, args.packet_paths):
        print(f"\n{'='*60}")
        print(f"Packet: {p_label}  ({p_path})")
        print(f"{'='*60}")
        imgs, lbls = load_packet_tensors(p_path)
        # 重映射 labels 到本地 0-based
        class_remap = {c: i for i, c in enumerate(sorted(args.class_ids))}
        lbls_local = torch.tensor([class_remap.get(int(l), int(l)) for l in lbls.tolist()])

        for arch_key in args.archs:
            print(f"\n  Receiver: {ARCH_DISPLAY[arch_key]}")
            t0 = time.time()
            model = build_model(arch_key, num_classes, device)
            best_acc = train_and_eval(
                model, imgs.clone(), lbls_local.clone(), val_loader,
                epochs=args.epochs, device=device
            )
            elapsed = time.time() - t0
            row = {
                "packet_label": p_label,
                "receiver_arch": arch_key,
                "class_ids": str(sorted(args.class_ids)),
                "num_classes": num_classes,
                "best_acc": round(best_acc, 2),
                "epochs": args.epochs,
                "time_sec": round(elapsed, 1),
            }
            results.append(row)
            print(f"  → best_acc={best_acc:.2f}%  elapsed={elapsed:.1f}s")

    # 写 CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if is_new:
            w.writeheader()
        w.writerows(results)
    print(f"\n结果已写入: {out}")

    # 打印矩阵
    print("\n=== 跨架构评估矩阵 ===")
    archs = args.archs
    print(f"{'Packet':<25}", end="")
    for a in archs:
        print(f" {ARCH_DISPLAY[a]:>10}", end="")
    print()
    print("-" * (25 + 11 * len(archs)))
    for p_label in args.packet_labels:
        print(f"{p_label:<25}", end="")
        for arch_key in archs:
            row = next((r for r in results
                        if r["packet_label"] == p_label and r["receiver_arch"] == arch_key), None)
            val = f"{row['best_acc']:.1f}" if row else "N/A"
            print(f" {val:>10}", end="")
        print()


if __name__ == "__main__":
    main()
