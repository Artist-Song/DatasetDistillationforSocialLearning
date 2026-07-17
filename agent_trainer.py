import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from agent_data import get_agent_class_split, get_agent_dir, get_agent_train_dataset, get_num_classes, get_test_dataset


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def _guide_paths(args, ckpt_dir):
    """返回当前 agent 期望的 guide model 路径列表。"""
    return [ckpt_dir / f"guide_model_{idx}.pt" for idx in range(int(args.pretrained_model_number))]


def _all_guides_exist(args, ckpt_dir):
    """检查 guide model pool 是否完整，避免 resume 到半成品。"""
    return all(path.exists() for path in _guide_paths(args, ckpt_dir))


def _agent_test_subset(args, agent_id):
    """构建只包含 agent expert classes 的测试子集。"""
    dataset = get_test_dataset(args)
    class_ids = set(int(c) for c in get_agent_class_split(args)[int(agent_id)])
    targets = getattr(dataset, "targets")
    indices = [idx for idx, label in enumerate(targets) if int(label) in class_ids]
    return Subset(dataset, indices)


def _evaluate_expert_accuracy(model, loader, device):
    """评估 guide model 在本 agent expert classes 上的准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(1, total)


def _select_best_expert(args, agent_id, ckpt_dir, device):
    """从 guide pool 中选择 expert-class accuracy 最高的模型作为 expert。"""
    from train import define_model

    test_set = _agent_test_subset(args, agent_id)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = []
    best = None
    for model_idx, path in enumerate(_guide_paths(args, ckpt_dir)):
        if not path.exists():
            raise FileNotFoundError(f"缺少 guide model，不能选择 expert: {path}")
        model = define_model(args, get_num_classes(args)).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        acc = _evaluate_expert_accuracy(model, test_loader, device)
        item = {"model_idx": model_idx, "path": str(path), "expert_acc": acc}
        metrics.append(item)
        if best is None or acc > best["expert_acc"]:
            best = item
        del model
    if best is None:
        raise RuntimeError("没有可用 guide model，不能生成 expert_model.pt")
    expert_path = ckpt_dir / "expert_model.pt"
    torch.save(torch.load(best["path"], map_location="cpu"), expert_path)
    metadata = {
        "agent_id": int(agent_id),
        "selection_rule": "best_expert_accuracy",
        "selected_model_idx": int(best["model_idx"]),
        "selected_model_path": best["path"],
        "selected_expert_acc": float(best["expert_acc"]),
        "guide_metrics": metrics,
    }
    with open(ckpt_dir / "expert_selection.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(
        f"[train_experts] agent={agent_id} selected guide_model_{best['model_idx']} "
        f"as expert acc={best['expert_acc']:.2f}"
    )
    return expert_path


def train_agent_experts(args, agent_id, resume=False, overwrite=False):
    """训练单个 agent 的 guide model pool，并选择最优 guide 作为 expert_model。"""
    _ensure_dsdm_path()
    from train import define_model

    agent_dir = get_agent_dir(args, agent_id)
    ckpt_dir = agent_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expert_path = ckpt_dir / "expert_model.pt"
    metadata_path = ckpt_dir / "expert_selection.json"
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if resume and not overwrite and expert_path.exists() and metadata_path.exists() and _all_guides_exist(args, ckpt_dir):
        return expert_path
    if resume and not overwrite and _all_guides_exist(args, ckpt_dir):
        return _select_best_expert(args, agent_id, ckpt_dir, device)

    expert_batch_size = int(getattr(args, "expert_batch_size", args.batch_size))
    dataset = get_agent_train_dataset(
        args,
        agent_id,
        normalize=True,
        augment=bool(getattr(args, "expert_augment", False)),
    )
    loader = DataLoader(dataset, batch_size=expert_batch_size, shuffle=True, num_workers=args.workers)
    if int(args.pretrained_model_number) <= 0:
        raise RuntimeError("pretrained_model_number 必须大于 0")
    for model_idx in range(int(args.pretrained_model_number)):
        model = define_model(args, get_num_classes(args)).to(device)
        expert_lr = float(getattr(args, "expert_lr", args.lr))
        optimizer = optim.SGD(model.parameters(), lr=expert_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_name = str(getattr(args, "expert_scheduler", "none")).lower()
        if scheduler_name == "multistep":
            milestones = getattr(args, "expert_scheduler_milestones", None) or [
                int(args.pretrained_epochs) // 2,
                3 * int(args.pretrained_epochs) // 4,
            ]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(v) for v in milestones],
                gamma=float(getattr(args, "expert_scheduler_gamma", 0.1)),
            )
        elif scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.pretrained_epochs))
        elif scheduler_name in {"", "none"}:
            scheduler = None
        else:
            raise ValueError(f"不支持的 expert scheduler: {scheduler_name}")
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(int(args.pretrained_epochs)):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        guide_path = ckpt_dir / f"guide_model_{model_idx}.pt"
        torch.save(model.state_dict(), guide_path)
        del model
    return _select_best_expert(args, agent_id, ckpt_dir, device)


def prepare_agent_pretrained_dir(args, agent_id):
    """把完整 guide pool 映射为 DSDM 期望的预训练模型目录。"""
    ckpt_dir = get_agent_dir(args, agent_id) / "checkpoints"
    missing = [str(path) for path in _guide_paths(args, ckpt_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError("DSDM 预训练 guide model 不完整，请先运行 train_experts: " + ", ".join(missing))
    dsdm_dir = ckpt_dir / "dsdm_pretrained"
    dsdm_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model_idx, src in enumerate(_guide_paths(args, ckpt_dir)):
        dst = dsdm_dir / f"{args.dataset}_model_{model_idx}.pth"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            torch.save(torch.load(src, map_location="cpu"), dst)
        manifest.append({"model_idx": model_idx, "src": str(src), "dst": str(dst)})
    with open(dsdm_dir / "pretrained_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"agent_id": int(agent_id), "models": manifest}, f, indent=2, ensure_ascii=False)
    return dsdm_dir


def describe_agent(agent_id, args_or_cfg=None):
    """返回 agent 的固定类别说明。"""
    return f"agent_{agent_id}: classes={get_agent_class_split(args_or_cfg)[int(agent_id)]}"
