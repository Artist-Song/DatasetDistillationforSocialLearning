from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedProto on the fixed socialized CIFAR-100 split.")
    parser.add_argument("--metadata", required=True, help="split_metadata.json")
    parser.add_argument("--fedproto-repo", default="external_baselines/repos/FedProto", help="FedProto 仓库路径")
    parser.add_argument("--data-dir", default="./data", help="CIFAR-100 数据目录")
    parser.add_argument("--output", default="external_baselines/outputs/fedproto_social/results.csv")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ld", type=float, default=1.0, help="FedProto prototype loss weight")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imagenet-init", action="store_true", help="沿用官方 ImageNet 初始化")
    parser.add_argument("--smoke", action="store_true", help="只构建数据和模型，不训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fedproto_repo = Path(args.fedproto_repo).resolve()
    _install_fedproto_paths(fedproto_repo)

    from resnet import resnet18  # noqa: WPS433
    from update import LocalUpdate  # noqa: WPS433
    from utils import agg_func, proto_aggregation, trans_cifar100_train, trans_cifar100_val  # noqa: WPS433

    _set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=trans_cifar100_train, download=False)
    test_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=trans_cifar100_val, download=False)

    user_groups = {int(k): np.asarray(v["train_indices"], dtype=np.int64) for k, v in metadata["agents"].items()}
    classes_list = {int(k): np.asarray(v["classes"], dtype=np.int64) for k, v in metadata["agents"].items()}

    fed_args = _build_fedproto_args(args, metadata, device)
    models = [_build_model(resnet18, fed_args, user_id, device, args.imagenet_init) for user_id in sorted(user_groups)]
    print(f"[fedproto] users={fed_args.num_users} rounds={fed_args.rounds} device={fed_args.device}")
    for user_id in sorted(classes_list):
        print(f"[user {user_id}] classes={classes_list[user_id][0]}..{classes_list[user_id][-1]} train={len(user_groups[user_id])}")
    if args.smoke:
        print("[smoke] model/data construction passed")
        return

    global_protos = []
    for round_id in range(fed_args.rounds):
        local_protos = {}
        local_losses = []
        print(f"\n| FedProto round {round_id + 1}/{fed_args.rounds} |")
        for user_id in sorted(user_groups):
            local = LocalUpdate(args=fed_args, dataset=train_dataset, idxs=user_groups[user_id])
            weights, loss, acc, protos = local.update_weights_het(
                fed_args,
                user_id,
                global_protos,
                model=copy.deepcopy(models[user_id]),
                global_round=round_id,
            )
            models[user_id].load_state_dict(weights, strict=True)
            local_protos[user_id] = agg_func(protos)
            local_losses.append(float(loss["total"]))
            print(f"[user {user_id}] loss={loss['total']:.4f} acc={acc:.4f}")
        global_protos = proto_aggregation(local_protos)
        print(f"[round {round_id + 1}] mean_loss={np.mean(local_losses):.4f} proto_classes={len(global_protos)}")

    rows = evaluate_social_metrics(fed_args, models, test_dataset, classes_list, global_protos)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fedproto] wrote {output_path}")


def evaluate_social_metrics(args, models, test_dataset, classes_list, global_protos) -> list[dict]:
    """按本文 global/new/expert 指标评估 FedProto 模型。"""
    rows = []
    targets = np.asarray(test_dataset.targets)
    all_indices = np.arange(len(targets))
    for user_id, model in enumerate(models):
        expert_classes = set(int(c) for c in classes_list[user_id])
        expert_indices = all_indices[np.isin(targets, list(expert_classes))]
        new_indices = all_indices[~np.isin(targets, list(expert_classes))]
        for mode in ("logit", "prototype"):
            rows.append(
                {
                    "user": user_id,
                    "mode": mode,
                    "acc_global": _eval_accuracy(args, model, test_dataset, all_indices, mode, global_protos),
                    "acc_expert": _eval_accuracy(args, model, test_dataset, expert_indices, mode, global_protos),
                    "acc_new": _eval_accuracy(args, model, test_dataset, new_indices, mode, global_protos),
                    "expert_classes": f"{min(expert_classes)}-{max(expert_classes)}",
                }
            )
    return rows


def _eval_accuracy(args, model, dataset, indices, mode, global_protos) -> float:
    """支持直接 logits 和 nearest global prototype 两种 FedProto 推理。"""
    model.eval()
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=128, shuffle=False)
    correct = 0
    total = 0
    proto_labels, proto_tensor = _stack_global_protos(global_protos, args.device) if mode == "prototype" else (None, None)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs, features = model(images)
            if mode == "logit":
                pred = outputs.argmax(dim=1)
            else:
                feats = features.reshape(features.size(0), -1)
                dists = torch.cdist(feats, proto_tensor)
                pred = proto_labels[dists.argmin(dim=1)]
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def _stack_global_protos(global_protos, device):
    """把官方 global_protos dict 转成可批量距离计算的张量。"""
    labels = sorted(int(k) for k in global_protos.keys())
    proto_list = [global_protos[label][0].reshape(-1).to(device) for label in labels]
    return torch.tensor(labels, dtype=torch.long, device=device), torch.stack(proto_list, dim=0)


def _build_fedproto_args(args: argparse.Namespace, metadata: dict, device: torch.device) -> SimpleNamespace:
    """构造官方 FedProto LocalUpdate 所需参数。"""
    return SimpleNamespace(
        num_users=int(metadata["num_agents"]),
        num_classes=int(metadata["num_classes"]),
        rounds=int(args.rounds),
        train_ep=int(args.local_epochs),
        local_bs=int(args.batch_size),
        lr=float(args.lr),
        momentum=0.5,
        optimizer="sgd",
        verbose=0,
        ld=float(args.ld),
        device=str(device),
        gpu=int(args.gpu),
        dataset="cifar100",
        mode="model_heter",
        stride=[1, 4],
    )


def _build_model(resnet18, fed_args, user_id: int, device: torch.device, imagenet_init: bool):
    """按官方 FedProto CIFAR 分支构建 ResNet18。"""
    if fed_args.mode == "model_heter":
        fed_args.stride = [1, 4] if user_id < 10 else [2, 2]
    model = resnet18(fed_args, pretrained=False, num_classes=fed_args.num_classes)
    if imagenet_init:
        import torch.utils.model_zoo as model_zoo  # noqa: WPS433

        model_urls = {"resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth"}
        initial_weight = model_zoo.load_url(model_urls["resnet18"])
        current_weight = model.state_dict()
        for key in initial_weight.keys():
            if key.startswith("fc.") or key.startswith("conv1") or key.startswith("bn1"):
                initial_weight[key] = current_weight[key]
        model.load_state_dict(initial_weight)
    return model.to(device).train()


def _install_fedproto_paths(repo: Path) -> None:
    """注入 FedProto 官方 lib 路径。"""
    for rel in ("lib", "lib/models"):
        path = str((repo / rel).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _set_seed(seed: int) -> None:
    """固定 Python/Numpy/Torch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
