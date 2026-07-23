import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from agent_data import get_agent_class_split, get_agent_dir, get_agent_train_dataset, get_num_classes, get_test_dataset
from output_manager import atomic_torch_save, atomic_write_json


def _sha256_file(path):
    """Return a stable hash for checkpoint provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def _guide_pool_dir(args, ckpt_dir, epoch=None):
    """Return the selected or requested guide-pool directory."""
    snapshot_epochs = [int(v) for v in getattr(args, "guide_snapshot_epochs", [args.pretrained_epochs])]
    use_epoch_dirs = bool(getattr(args, "separate_expert", False)) or len(snapshot_epochs) > 1
    if not use_epoch_dirs:
        return Path(ckpt_dir)
    guide_epoch = int(epoch if epoch is not None else getattr(args, "guide_epoch", args.pretrained_epochs))
    return Path(ckpt_dir) / "guide_pools" / f"e{guide_epoch:04d}"


def _guide_paths(args, ckpt_dir, epoch=None):
    """Return all model paths for one guide-pool maturity."""
    pool_dir = _guide_pool_dir(args, ckpt_dir, epoch=epoch)
    model_count = int(getattr(args, "guide_model_number", args.pretrained_model_number))
    return [pool_dir / f"guide_model_{idx}.pt" for idx in range(model_count)]


def _all_guides_exist(args, ckpt_dir):
    """检查 guide model pool 是否完整，避免 resume 到半成品。"""
    return all(path.exists() for path in _guide_paths(args, ckpt_dir))


def _all_guide_snapshots_exist(args, ckpt_dir):
    """Check every requested guide maturity so a resumed sweep is complete."""
    epochs = [int(v) for v in getattr(args, "guide_snapshot_epochs", [args.pretrained_epochs])]
    return all(path.exists() for epoch in epochs for path in _guide_paths(args, ckpt_dir, epoch=epoch))


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


def _stratified_split_indices(dataset, validation_fraction, seed):
    """Build a deterministic per-class train/validation split."""
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction 必须位于 (0, 1): {fraction}")
    labels = [int(v) for v in getattr(dataset, "targets")]
    rng = np.random.RandomState(int(seed))
    train_indices = []
    validation_indices = []
    for class_id in sorted(set(labels)):
        class_indices = np.asarray([idx for idx, label in enumerate(labels) if label == class_id], dtype=np.int64)
        rng.shuffle(class_indices)
        validation_size = max(1, int(round(len(class_indices) * fraction)))
        validation_indices.extend(class_indices[:validation_size].tolist())
        train_indices.extend(class_indices[validation_size:].tolist())
    return sorted(train_indices), sorted(validation_indices)


def _build_scheduler(optimizer, name, epochs, milestones, gamma):
    """Build a training scheduler shared by guide and expert recipes."""
    scheduler_name = str(name).lower()
    if scheduler_name == "multistep":
        resolved = [int(v) for v in milestones] or [int(epochs) // 2, 3 * int(epochs) // 4]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=resolved, gamma=float(gamma))
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    if scheduler_name in {"", "none"}:
        return None
    raise ValueError(f"不支持的 scheduler: {name}")


def _build_dsdm_batch_augmentation(args, enabled):
    """Build the project's validated DSDM-style expert augmentation."""
    if not enabled or not bool(getattr(args, "dsa", False)):
        return None
    try:
        from misc.augment import DiffAug
    except ModuleNotFoundError:
        from DSDM.misc.augment import DiffAug

    return DiffAug(strategy=args.dsa_strategy, batch=False)


def _train_epoch(model, loader, criterion, optimizer, device, args=None, use_dsdm_train=False, batch_aug=None):
    """Train one epoch and return its sample-weighted mean loss."""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if batch_aug is not None:
            with torch.no_grad():
                images = batch_aug(images)
        optimizer.zero_grad()
        if (
            use_dsdm_train
            and getattr(args, "mixup", "") == "cut"
            and np.random.rand(1) < float(getattr(args, "mix_p", 0.0))
        ):
            try:
                from misc.utils import rand_bbox
            except ModuleNotFoundError:
                from DSDM.misc.utils import rand_bbox

            lam = np.random.beta(float(args.beta), float(args.beta))
            rand_index = torch.randperm(labels.numel(), device=device)
            labels_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1.0 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
            )
            logits = model(images)
            loss = criterion(logits, labels) * ratio + criterion(logits, labels_b) * (1.0 - ratio)
        else:
            loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().item()) * labels.numel()
        sample_count += labels.numel()
    return loss_sum / max(1, sample_count)


def _select_best_expert(args, agent_id, ckpt_dir, device):
    """Legacy compatibility: select one guide on the official test set."""
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
        "role": "legacy_guide_selected_expert",
        "selection_rule": "best_official_test_expert_accuracy",
        "test_used_for_selection": True,
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


def _train_guide_pool(args, agent_id, ckpt_dir, device, overwrite=False):
    """Train guide trajectories once and persist requested epoch snapshots."""
    from train import define_model

    if not overwrite and _all_guide_snapshots_exist(args, ckpt_dir):
        return _guide_paths(args, ckpt_dir)

    base_seed = int(args.seed) + 100_000 * int(agent_id)
    guide_batch_size = int(getattr(args, "guide_batch_size", args.batch_size))
    dataset = get_agent_train_dataset(
        args,
        agent_id,
        normalize=True,
        augment=bool(getattr(args, "guide_augment", False)),
    )
    max_epochs = int(getattr(args, "guide_max_epochs", args.pretrained_epochs))
    snapshot_epochs = sorted({int(v) for v in getattr(args, "guide_snapshot_epochs", [max_epochs])})
    if not snapshot_epochs or snapshot_epochs[0] <= 0 or snapshot_epochs[-1] > max_epochs:
        raise ValueError(f"guide snapshot epochs 非法: snapshots={snapshot_epochs} max={max_epochs}")
    model_count = int(getattr(args, "guide_model_number", args.pretrained_model_number))
    if model_count <= 0:
        raise RuntimeError("guide_model_number 必须大于 0")

    criterion = nn.CrossEntropyLoss()
    for model_idx in range(model_count):
        model_seed = base_seed + model_idx
        random.seed(model_seed)
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        loader_generator = torch.Generator().manual_seed(model_seed)
        loader = DataLoader(
            dataset,
            batch_size=guide_batch_size,
            shuffle=True,
            num_workers=args.workers,
            generator=loader_generator,
            persistent_workers=int(args.workers) > 0,
        )
        model = define_model(args, get_num_classes(args)).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(getattr(args, "guide_lr", args.lr)),
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = _build_scheduler(
            optimizer,
            getattr(args, "guide_scheduler", "none"),
            max_epochs,
            getattr(args, "guide_scheduler_milestones", []),
            getattr(args, "guide_scheduler_gamma", 0.1),
        )
        started = time.monotonic()
        print(
            f"[train_guides] agent={agent_id} guide={model_idx} max_epochs={max_epochs} "
            f"snapshots={snapshot_epochs} seed={model_seed}",
            flush=True,
        )
        report_every = max(1, max_epochs // 10)
        for epoch in range(1, max_epochs + 1):
            mean_loss = _train_epoch(model, loader, criterion, optimizer, device)
            if scheduler is not None:
                scheduler.step()
            if epoch in snapshot_epochs:
                atomic_torch_save(model.state_dict(), _guide_paths(args, ckpt_dir, epoch=epoch)[model_idx])
            if epoch % report_every == 0 or epoch in snapshot_epochs or epoch == max_epochs:
                print(
                    f"[train_guides] agent={agent_id} guide={model_idx} epoch={epoch}/{max_epochs} "
                    f"loss={mean_loss:.4f} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        del model

    for epoch in snapshot_epochs:
        pool_dir = _guide_pool_dir(args, ckpt_dir, epoch=epoch)
        atomic_write_json(
            {
                "agent_id": int(agent_id),
                "role": "dsdm_guide_pool",
                "epoch": int(epoch),
                "trajectory_max_epochs": max_epochs,
                "model_count": model_count,
                "model_paths": [str(path) for path in _guide_paths(args, ckpt_dir, epoch=epoch)],
                "model_artifacts": [
                    {"path": str(path), "sha256": _sha256_file(path)}
                    for path in _guide_paths(args, ckpt_dir, epoch=epoch)
                ],
                "optimizer": "sgd",
                "lr": float(getattr(args, "guide_lr", args.lr)),
                "batch_size": guide_batch_size,
                "augment": bool(getattr(args, "guide_augment", False)),
                "scheduler": str(getattr(args, "guide_scheduler", "none")),
                "test_used_for_selection": False,
            },
            pool_dir / "guide_pool_manifest.json",
        )
    return _guide_paths(args, ckpt_dir)


def _train_independent_expert(args, agent_id, ckpt_dir, device, overwrite=False):
    """Train a converged expert selected on local validation, never official test."""
    from train import define_model

    expert_path = ckpt_dir / "expert_model.pt"
    manifest_path = ckpt_dir / "expert_manifest.json"
    if not overwrite and expert_path.exists() and manifest_path.exists():
        return expert_path

    base_seed = int(args.seed) + 100_000 * int(agent_id) + 50_000
    batch_size = int(getattr(args, "expert_batch_size", args.batch_size))
    max_epochs = int(getattr(args, "expert_epochs", args.pretrained_epochs))
    validation_fraction = float(getattr(args, "expert_validation_fraction", 0.1))
    eval_interval = max(1, int(getattr(args, "expert_eval_interval", 5)))
    train_dataset = get_agent_train_dataset(args, agent_id, normalize=True, augment=bool(args.expert_augment))
    validation_dataset = get_agent_train_dataset(args, agent_id, normalize=True, augment=False)
    train_indices, validation_indices = _stratified_split_indices(train_dataset, validation_fraction, base_seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(base_seed),
        persistent_workers=int(args.workers) > 0,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)
    model = define_model(args, get_num_classes(args)).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(args.expert_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        args.expert_scheduler,
        max_epochs,
        args.expert_scheduler_milestones,
        args.expert_scheduler_gamma,
    )
    criterion = nn.CrossEntropyLoss()
    use_dsdm_train = bool(getattr(args, "expert_use_dsdm_train", False))
    batch_aug = _build_dsdm_batch_augmentation(args, use_dsdm_train)
    validation_best_path = ckpt_dir / "expert_validation_best.pt"
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args=args,
            use_dsdm_train=use_dsdm_train,
            batch_aug=batch_aug,
        )
        if scheduler is not None:
            scheduler.step()
        if epoch % eval_interval == 0 or epoch == max_epochs:
            validation_accuracy = _evaluate_expert_accuracy(model, validation_loader, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_accuracy": validation_accuracy})
            print(
                f"[train_expert] agent={agent_id} epoch={epoch}/{max_epochs} "
                f"loss={train_loss:.4f} val={validation_accuracy:.2f}",
                flush=True,
            )
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                atomic_torch_save(model.state_dict(), validation_best_path)
    del model
    if best_epoch <= 0:
        raise RuntimeError("独立 expert 没有产生 validation checkpoint")

    retrained_full = bool(getattr(args, "expert_retrain_full", True))
    if retrained_full:
        full_dataset = get_agent_train_dataset(args, agent_id, normalize=True, augment=bool(args.expert_augment))
        full_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(base_seed + 1),
            persistent_workers=int(args.workers) > 0,
        )
        random.seed(base_seed + 1)
        np.random.seed(base_seed + 1)
        torch.manual_seed(base_seed + 1)
        torch.cuda.manual_seed_all(base_seed + 1)
        model = define_model(args, get_num_classes(args)).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(args.expert_lr),
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = _build_scheduler(
            optimizer,
            args.expert_scheduler,
            max_epochs,
            args.expert_scheduler_milestones,
            args.expert_scheduler_gamma,
        )
        for _epoch in range(1, best_epoch + 1):
            _train_epoch(
                model,
                full_loader,
                criterion,
                optimizer,
                device,
                args=args,
                use_dsdm_train=use_dsdm_train,
                batch_aug=batch_aug,
            )
            if scheduler is not None:
                scheduler.step()
        atomic_torch_save(model.state_dict(), expert_path)
    else:
        atomic_torch_save(torch.load(validation_best_path, map_location="cpu"), expert_path)
        model = define_model(args, get_num_classes(args)).to(device)
        model.load_state_dict(torch.load(expert_path, map_location=device))

    test_loader = DataLoader(_agent_test_subset(args, agent_id), batch_size=batch_size, shuffle=False, num_workers=0)
    official_test_accuracy = _evaluate_expert_accuracy(model, test_loader, device)
    manifest = {
        "agent_id": int(agent_id),
        "role": "fully_converged_agent_expert_and_logit_teacher",
        "selection_rule": "best_local_validation_accuracy",
        "test_used_for_selection": False,
        "max_epochs": max_epochs,
        "selected_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_accuracy),
        "official_test_accuracy_report_only": float(official_test_accuracy),
        "validation_fraction": validation_fraction,
        "retrained_on_full_local_train": retrained_full,
        "optimizer": "sgd",
        "lr": float(args.expert_lr),
        "batch_size": batch_size,
        "augment": bool(args.expert_augment),
        "use_dsdm_train": use_dsdm_train,
        "scheduler": str(args.expert_scheduler),
        "history": history,
        "expert_path": str(expert_path),
    }
    atomic_write_json(manifest, manifest_path)
    atomic_write_json(manifest, ckpt_dir / "expert_selection.json")
    validation_best_path.unlink(missing_ok=True)
    del model
    return expert_path


def train_agent_experts(args, agent_id, resume=False, overwrite=False):
    """Train the DSDM guide pool and a separate converged expert when configured."""
    _ensure_dsdm_path()

    agent_dir = get_agent_dir(args, agent_id)
    ckpt_dir = agent_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expert_path = ckpt_dir / "expert_model.pt"
    metadata_path = ckpt_dir / "expert_selection.json"
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if resume and not overwrite and expert_path.exists() and metadata_path.exists() and _all_guide_snapshots_exist(args, ckpt_dir):
        return expert_path
    _train_guide_pool(args, agent_id, ckpt_dir, device, overwrite=overwrite)
    if bool(getattr(args, "separate_expert", False)):
        return _train_independent_expert(args, agent_id, ckpt_dir, device, overwrite=overwrite)
    return _select_best_expert(args, agent_id, ckpt_dir, device)


def prepare_agent_pretrained_dir(args, agent_id):
    """把完整 guide pool 映射为 DSDM 期望的预训练模型目录。"""
    ckpt_dir = get_agent_dir(args, agent_id) / "checkpoints"
    source_root = getattr(args, "guide_source_root", None)
    source_ckpt_dir = ckpt_dir
    if source_root:
        source_ckpt_dir = Path(source_root) / "agents" / f"agent_{int(agent_id)}" / "checkpoints"
    source_paths = _guide_paths(args, source_ckpt_dir)
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("DSDM 预训练 guide model 不完整，请先运行 train_experts: " + ", ".join(missing))
    snapshot_epochs = [int(v) for v in getattr(args, "guide_snapshot_epochs", [args.pretrained_epochs])]
    dsdm_dir = ckpt_dir / "dsdm_pretrained"
    if bool(getattr(args, "separate_expert", False)) or len(snapshot_epochs) > 1:
        dsdm_dir = dsdm_dir / f"e{int(getattr(args, 'guide_epoch', args.pretrained_epochs)):04d}"
    dsdm_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model_idx, src in enumerate(source_paths):
        dst = dsdm_dir / f"{args.dataset}_model_{model_idx}.pth"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            atomic_torch_save(torch.load(src, map_location="cpu"), dst)
        manifest.append(
            {
                "model_idx": model_idx,
                "src": str(src),
                "src_sha256": _sha256_file(src),
                "dst": str(dst),
                "dst_sha256": _sha256_file(dst),
            }
        )
    with open(dsdm_dir / "pretrained_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent_id": int(agent_id),
                "role": "dsdm_guide_pool",
                "guide_epoch": int(getattr(args, "guide_epoch", args.pretrained_epochs)),
                "models": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return dsdm_dir


def describe_agent(agent_id, args_or_cfg=None):
    """返回 agent 的固定类别说明。"""
    return f"agent_{agent_id}: classes={get_agent_class_split(args_or_cfg)[int(agent_id)]}"
