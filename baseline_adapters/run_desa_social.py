from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DeSA-style synthetic-anchor baseline on social CIFAR-100.")
    parser.add_argument("--metadata", required=True, help="split_metadata.json")
    parser.add_argument("--desa-repo", default="external_baselines/repos/DESA", help="DeSA 仓库路径")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="external_baselines/outputs/desa_social/results.csv")
    parser.add_argument("--anchor-output", default="external_baselines/outputs/desa_social/anchors.pt")
    parser.add_argument("--reuse-anchors", action="store_true", help="复用已有 anchors.pt，只重跑 receiver social training")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--anchor-iters", type=int, default=1000)
    parser.add_argument("--social-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-model", type=float, default=0.01)
    parser.add_argument("--lr-img", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="只构建数据和模型，不训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    _install_desa_path(Path(args.desa_repo).resolve())

    from utils import DiffAugment, ParamDiffAug, get_network  # noqa: WPS433

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    train_dataset, test_dataset = _build_cifar100(args.data_dir)
    model_names = [_map_model_name(metadata["model_split"][str(i)]) for i in range(metadata["num_agents"])]
    classes = {int(k): [int(c) for c in v["classes"]] for k, v in metadata["agents"].items()}
    train_indices = {int(k): [int(i) for i in v["train_indices"]] for k, v in metadata["agents"].items()}

    print(f"[desa-social] agents={metadata['num_agents']} ipc={args.ipc} device={device}")
    for agent_id, model_name in enumerate(model_names):
        print(f"[agent {agent_id}] model={model_name} classes={classes[agent_id][0]}..{classes[agent_id][-1]}")
    models = [get_network(name, 3, 100, (32, 32)).to(device) for name in model_names]
    if args.smoke:
        print("[smoke] model/data construction passed")
        return

    pretrained = []
    before_rows = {}
    for agent_id, model in enumerate(models):
        loader = DataLoader(Subset(train_dataset, train_indices[agent_id]), batch_size=args.batch_size, shuffle=True)
        print(f"[pretrain] agent={agent_id} samples={len(train_indices[agent_id])}")
        _train_classifier(model, loader, device, epochs=args.pretrain_epochs, lr=args.lr_model)
        expert_model = copy.deepcopy(model).eval()
        pretrained.append(expert_model)
        before_rows[agent_id] = _evaluate_receiver(
            agent_id,
            expert_model,
            test_dataset,
            targets=np.asarray(test_dataset.targets),
            all_indices=np.arange(len(test_dataset.targets)),
            expert_classes=classes[agent_id],
            device=device,
        )

    anchor_path = Path(args.anchor_output)
    if args.reuse_anchors and anchor_path.exists():
        anchors = _load_anchors(anchor_path, device)
        print(f"[anchors] reused {anchor_path}")
    else:
        anchors = {}
        for agent_id in range(metadata["num_agents"]):
            print(f"[anchors] agent={agent_id} images={len(classes[agent_id]) * args.ipc}")
            anchors[agent_id] = _generate_active_anchors(
                get_network=get_network,
                diff_augment=DiffAugment,
                aug_param_cls=ParamDiffAug,
                dataset=train_dataset,
                indices=train_indices[agent_id],
                active_classes=classes[agent_id],
                ipc=args.ipc,
                device=device,
                iters=args.anchor_iters,
                lr_img=args.lr_img,
            )
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in anchors.items()}, anchor_path)
        print(f"[anchors] wrote {anchor_path}")

    rows = []
    targets = np.asarray(test_dataset.targets)
    all_test_indices = np.arange(len(targets))
    for receiver_id, base_model in enumerate(pretrained):
        external_images = []
        external_labels = []
        for sender_id, (images, labels) in anchors.items():
            if sender_id == receiver_id:
                continue
            external_images.append(images.detach().cpu())
            external_labels.append(labels.detach().cpu())
        synthetic_dataset = TensorDataset(torch.cat(external_images), torch.cat(external_labels))
        real_dataset = TensorLabelSubset(train_dataset, train_indices[receiver_id])
        social_dataset = ConcatDataset([real_dataset, synthetic_dataset])
        social_loader = DataLoader(social_dataset, batch_size=args.batch_size, shuffle=True)
        social_model = copy.deepcopy(base_model).to(device)
        print(f"[social-train] receiver={receiver_id} samples={len(social_dataset)}")
        _train_classifier(social_model, social_loader, device, epochs=args.social_epochs, lr=args.lr_model)
        eval_row = _evaluate_receiver(receiver_id, social_model, test_dataset, targets, all_test_indices, classes[receiver_id], device)
        before = before_rows[receiver_id]
        acc_global_after = eval_row["acc_global"]
        acc_expert_after = eval_row["acc_expert"]
        acc_new_after = eval_row["acc_new"]
        row = {
            "receiver": receiver_id,
            "method": "DESA",
            "ipc": args.ipc,
            "acc_global_before": before["acc_global"],
            "acc_expert_before": before["acc_expert"],
            "acc_global_after": acc_global_after,
            "acc_expert_after": acc_expert_after,
            "acc_new_after": acc_new_after,
            "forgetting": before["acc_expert"] - acc_expert_after,
            "external_comm_images": (metadata["num_agents"] - 1) * len(classes[receiver_id]) * args.ipc,
            "external_comm_logit_bytes": 0,
            "expert_classes": eval_row["expert_classes"],
        }
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[desa-social] wrote {output_path}")


class TensorLabelSubset(Dataset):
    """把 torchvision 样本标签统一转成 long tensor，便于和 TensorDataset 混合。"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, label = self.dataset[self.indices[item]]
        return image, torch.tensor(int(label), dtype=torch.long)


def _load_anchors(path: Path, device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """读取已生成的 DeSA anchors，并规范 key 与 label 类型。"""
    loaded = torch.load(path, map_location="cpu")
    anchors = {}
    for key, value in loaded.items():
        images, labels = value
        anchors[int(key)] = (images.to(device).float(), labels.to(device).long())
    return anchors


def _generate_active_anchors(
    get_network,
    diff_augment,
    aug_param_cls,
    dataset,
    indices,
    active_classes,
    ipc,
    device,
    iters,
    lr_img,
):
    """只为 sender 的 expert classes 生成 DeSA-style synthetic anchors。"""
    real_images, real_labels = _materialize_subset(dataset, indices, device)
    by_class = {cls: torch.nonzero(real_labels == int(cls), as_tuple=False).view(-1) for cls in active_classes}
    image_syn = torch.randn(len(active_classes) * ipc, 3, 32, 32, device=device, requires_grad=True)
    label_syn = torch.tensor([cls for cls in active_classes for _ in range(ipc)], dtype=torch.long, device=device)
    optimizer = torch.optim.SGD([image_syn], lr=lr_img, momentum=0.5)
    for iter_id in range(iters):
        net = get_network("ConvNet", 3, 100, (32, 32)).to(device).train()
        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed
        loss = torch.tensor(0.0, device=device)
        seed = int(time.time() * 1000) % 100000
        dsa_param = aug_param_cls()
        for offset, cls in enumerate(active_classes):
            cls_idx = by_class[cls]
            chosen = cls_idx[torch.randint(0, len(cls_idx), (min(256, len(cls_idx)),), device=device)]
            img_real = real_images[chosen]
            img_syn = image_syn[offset * ipc : (offset + 1) * ipc]
            img_real = diff_augment(img_real, "color_crop_cutout_flip_scale_rotate", seed=seed, param=dsa_param)
            img_syn = diff_augment(img_syn, "color_crop_cutout_flip_scale_rotate", seed=seed, param=dsa_param)
            loss = loss + torch.sum((embed(img_real).detach().mean(dim=0) - embed(img_syn).mean(dim=0)) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_id % 100 == 0:
            print(f"[anchors] iter={iter_id:04d} loss={loss.item() / len(active_classes):.4f}")
    return image_syn.detach(), label_syn.detach()


def _train_classifier(model, loader, device, epochs: int, lr: float) -> None:
    """训练 DeSA 官方网络，网络 forward 返回 feature/logits。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            _, logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"[train] epoch={epoch + 1}/{epochs} acc={correct / max(total, 1):.4f}")


def _evaluate_receiver(receiver_id, model, dataset, targets, all_indices, expert_classes, device) -> dict:
    """输出本文 social learning 指标。"""
    expert_mask = np.isin(targets, expert_classes)
    expert_indices = all_indices[expert_mask]
    new_indices = all_indices[~expert_mask]
    return {
        "receiver": receiver_id,
        "acc_global": _eval_accuracy(model, dataset, all_indices, device),
        "acc_expert": _eval_accuracy(model, dataset, expert_indices, device),
        "acc_new": _eval_accuracy(model, dataset, new_indices, device),
        "expert_classes": f"{min(expert_classes)}-{max(expert_classes)}",
    }


def _eval_accuracy(model, dataset, indices, device) -> float:
    """计算给定索引集合上的 top-1 accuracy。"""
    model.eval()
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            _, logits = model(images)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def _materialize_subset(dataset, indices, device) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个 agent 的真实数据加载到 GPU 方便 feature matching。"""
    loader = DataLoader(Subset(dataset, indices), batch_size=512, shuffle=False)
    images_all = []
    labels_all = []
    for images, labels in loader:
        images_all.append(images)
        labels_all.append(labels)
    return torch.cat(images_all).to(device).float(), torch.cat(labels_all).to(device).long()


def _build_cifar100(data_dir: str):
    """构造 DeSA runner 使用的 CIFAR-100 数据集。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return (
        datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=False),
        datasets.CIFAR100(data_dir, train=False, transform=test_transform, download=False),
    )


def _map_model_name(project_model: str) -> str:
    """把本项目模型名映射到 DeSA 官方网络名。"""
    name = project_model.lower()
    if "alexnet" in name:
        return "AlexNet"
    if "vgg" in name:
        return "VGG11"
    if "resnet" in name:
        return "ResNet18BN"
    return "ConvNet"


def _install_desa_path(repo: Path) -> None:
    """注入 DeSA 官方源码路径。"""
    path = str(repo.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
