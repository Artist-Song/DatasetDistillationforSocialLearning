from __future__ import annotations

import copy
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agent_data import get_agent_class_split, get_agent_dir, get_num_classes
from packet_consumer import consume_packet_for_training


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def _refresh_model_metadata(args):
    """刷新 generalist 模型标签，保持与 DSDM define_model 参数一致。"""
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
    if float(args.width) != 1.0:
        args.modeltag += f"_w{args.width}"


def build_generalist_args(cfg, base_args):
    """根据 config 构造 packet generalist 的模型与训练参数。"""
    args = copy.deepcopy(base_args)
    generalist_cfg = cfg.get("generalist", {})
    model_name = str(generalist_cfg.get("model", cfg.get("model_pool", {}).get("default", args.model_name)))
    model_cfg = cfg.get("model_pool", {}).get("models", {}).get(model_name, {})
    args.model_name = model_name
    args.net_type = str(model_cfg.get("family", args.net_type))
    args.depth = int(model_cfg.get("depth", args.depth))
    args.width = float(model_cfg.get("width", args.width))
    args.norm_type = str(model_cfg.get("norm_type", args.norm_type))
    args.num_classes = int(generalist_cfg.get("output_dim", get_num_classes(args)))
    args.nclass = args.num_classes
    args.epochs = int(generalist_cfg.get("epochs", cfg.get("social_learning", {}).get("receiver", {}).get("epochs", 500)))
    args.lr = float(generalist_cfg.get("lr", cfg.get("social_learning", {}).get("receiver", {}).get("lr", args.lr)))
    args.batch_size = int(generalist_cfg.get("batch_size", args.batch_size))
    args.lambda_skd = float(generalist_cfg.get("lambda_skd", cfg.get("logits", {}).get("lambda_kd", 0.5)))
    args.kd_temperature = float(generalist_cfg.get("temperature", cfg.get("logits", {}).get("temperature", 2.0)))
    _refresh_model_metadata(args)
    return args


def get_generalist_dir(args):
    """返回 packet generalist 输出目录。"""
    return Path(args.output_root) / args.run_name / "packet_generalist"


def get_generalist_checkpoint_path(args):
    """返回 packet generalist checkpoint 路径。"""
    return get_generalist_dir(args) / "checkpoints" / "generalist_model.pt"


def _reuse_roots(cfg, args, enabled_key):
    """返回 source/target run 根目录。"""
    reuse_cfg = cfg.get("reuse", {})
    if not bool(reuse_cfg.get(enabled_key, False)):
        return None, None, reuse_cfg
    source_run = reuse_cfg.get("source_run_name")
    if not source_run:
        raise ValueError(f"reuse.{enabled_key}=true 时必须配置 reuse.source_run_name")
    source_root = Path(args.output_root) / str(source_run)
    target_root = Path(args.output_root) / str(args.run_name)
    return source_root, target_root, reuse_cfg


def ensure_reused_packets(cfg, args, packet_method):
    """从 source_run_name 复用已完成 packets 到当前新 run。"""
    source_root, target_root, reuse_cfg = _reuse_roots(cfg, args, "reuse_packets")
    if source_root is None:
        return []
    copied = []
    for agent_id in get_agent_class_split(args):
        src = source_root / "agents" / f"agent_{agent_id}" / "packets" / f"{packet_method}_packet.pt"
        dst = target_root / "agents" / f"agent_{agent_id}" / "packets" / f"{packet_method}_packet.pt"
        if not src.exists():
            raise FileNotFoundError(f"缺少可复用 packet: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or bool(reuse_cfg.get("overwrite_packets", False)):
            shutil.copyfile(src, dst)
            copied.append(str(dst))
    return copied


def ensure_reused_experts(cfg, args):
    """从 source_run_name 复用 expert checkpoints 到当前新 run。"""
    source_root, target_root, reuse_cfg = _reuse_roots(cfg, args, "reuse_experts")
    if source_root is None:
        return []
    copied = []
    for agent_id in get_agent_class_split(args):
        src = source_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        dst = target_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        if not src.exists():
            raise FileNotFoundError(f"缺少可复用 expert checkpoint: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or bool(reuse_cfg.get("overwrite_experts", False)):
            shutil.copyfile(src, dst)
            copied.append(str(dst))
    return copied


def ensure_reused_artifacts(cfg, args, packet_method):
    """复用 packet generalist 实验所需的旧 run artifacts。"""
    copied = []
    copied.extend(ensure_reused_packets(cfg, args, packet_method))
    copied.extend(ensure_reused_experts(cfg, args))
    return copied


def _normalize_images(args, images, device):
    """按 CIFAR 均值方差归一化 packet 图片。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset], device=device).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset], device=device).view(1, -1, 1, 1)
    return (images.to(device).float() - mean) / std


def _load_generalist_training_tensors(args, packet_method):
    """读取所有 agent packet，拼接 generalist 训练张量。"""
    image_chunks = []
    label_chunks = []
    logit_chunks = []
    class_id_chunks = []
    sender_chunks = []
    for agent_id in get_agent_class_split(args):
        packet_path = get_agent_dir(args, agent_id) / "packets" / f"{packet_method}_packet.pt"
        consumed = consume_packet_for_training(args, packet_path, require_sender_logits=True)
        image_chunks.append(consumed["images"])
        label_chunks.append(consumed["labels"])
        logit_chunks.append(consumed["sender_logits"].float())
        class_id_chunks.append(consumed["sender_logit_class_ids"].long())
        sender_chunks.append(torch.full((consumed["images"].shape[0],), int(agent_id), dtype=torch.long))
    return {
        "images": torch.cat(image_chunks),
        "labels": torch.cat(label_chunks),
        "sender_logits": torch.cat(logit_chunks),
        "sender_logit_class_ids": torch.cat(class_id_chunks),
        "sender_agents": torch.cat(sender_chunks),
    }


def _build_generalist_loader(args, tensors):
    """构建 packet generalist 训练 loader。"""
    dataset = TensorDataset(
        tensors["images"].float(),
        tensors["labels"].long(),
        tensors["sender_logits"].float(),
        tensors["sender_logit_class_ids"].long(),
        tensors["sender_agents"].long(),
    )
    return DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)


def _sender_kd_loss(student_logits, teacher_logits, teacher_class_ids, temperature):
    """在 sender expert classes 上计算 packet generalist 的 sender KD。"""
    student_selected = student_logits.gather(1, teacher_class_ids)
    teacher_p = torch.softmax(teacher_logits / temperature, dim=1)
    student_log_p = torch.log_softmax(student_selected / temperature, dim=1)
    return temperature * temperature * nn.functional.kl_div(student_log_p, teacher_p, reduction="batchmean")


def train_packet_generalist(cfg, base_args, packet_method):
    """用所有 communicated packets 训练 packet-mediated generalist。"""
    from train import define_model

    ensure_reused_artifacts(cfg, base_args, packet_method)
    args = build_generalist_args(cfg, base_args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tensors = _load_generalist_training_tensors(base_args, packet_method)
    loader = _build_generalist_loader(args, tensors)
    model = define_model(args, int(args.num_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=float(args.momentum), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2 * int(args.epochs) // 3, 5 * int(args.epochs) // 6],
        gamma=0.2,
    )
    lambda_skd = float(args.lambda_skd)
    temperature = float(args.kd_temperature)
    metrics_dir = get_generalist_dir(base_args) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "generalist_train.csv"
    last_cls = 0.0
    last_kd = 0.0
    model.train()
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss_cls", "loss_skd", "loss", "time"])
        writer.writeheader()
        for epoch in range(1, int(args.epochs) + 1):
            total_cls = 0.0
            total_kd = 0.0
            total_loss = 0.0
            total_n = 0
            for images, labels, sender_logits, sender_class_ids, _sender_agents in loader:
                images = _normalize_images(args, images, device)
                labels = labels.to(device)
                sender_logits = sender_logits.to(device)
                sender_class_ids = sender_class_ids.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss_cls = criterion(logits, labels)
                loss_skd = _sender_kd_loss(logits, sender_logits, sender_class_ids, temperature)
                loss = loss_cls + lambda_skd * loss_skd
                loss.backward()
                optimizer.step()
                n = int(images.shape[0])
                total_cls += float(loss_cls.detach().cpu()) * n
                total_kd += float(loss_skd.detach().cpu()) * n
                total_loss += float(loss.detach().cpu()) * n
                total_n += n
            scheduler.step()
            last_cls = total_cls / max(1, total_n)
            last_kd = total_kd / max(1, total_n)
            if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 10) == 0:
                writer.writerow(
                    {
                        "epoch": epoch,
                        "loss_cls": last_cls,
                        "loss_skd": last_kd,
                        "loss": total_loss / max(1, total_n),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                f.flush()
                print(f"[packet_generalist] epoch={epoch}/{args.epochs} loss_cls={last_cls:.4f} loss_skd={last_kd:.4f}")
    ckpt_path = get_generalist_checkpoint_path(base_args)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": args.model_name,
            "net_type": args.net_type,
            "depth": int(args.depth),
            "width": float(args.width),
            "norm_type": args.norm_type,
            "output_dim": int(args.num_classes),
            "packet_method": packet_method,
            "loss_cls": last_cls,
            "loss_skd": last_kd,
        },
        ckpt_path,
    )
    print(f"[packet_generalist] saved: {ckpt_path}")
    return ckpt_path


def _load_generalist_model(cfg, base_args, device):
    """加载已训练的 packet generalist。"""
    from train import define_model

    args = build_generalist_args(cfg, base_args)
    ckpt_path = get_generalist_checkpoint_path(base_args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少 packet generalist checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = define_model(args, int(args.num_classes)).to(device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()
    return args, model, ckpt_path


def attach_generalist_logits_to_packets(cfg, base_args, packet_method, only_agent=None):
    """用 packet generalist 为 packet effective images 标注 full-100 logits。"""
    ensure_reused_artifacts(cfg, base_args, packet_method)
    device = torch.device("cuda" if base_args.device == "cuda" and torch.cuda.is_available() else "cpu")
    g_args, model, ckpt_path = _load_generalist_model(cfg, base_args, device)
    agent_ids = [int(only_agent)] if only_agent is not None else list(get_agent_class_split(base_args).keys())
    batch_size = int(getattr(g_args, "batch_size", 128))
    for agent_id in agent_ids:
        packet_path = get_agent_dir(base_args, agent_id) / "packets" / f"{packet_method}_packet.pt"
        consumed = consume_packet_for_training(base_args, packet_path, require_sender_logits=True)
        images = consumed["images"]
        logits_chunks = []
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = _normalize_images(g_args, images[start : start + batch_size], device)
                logits_chunks.append(model(batch).detach().cpu())
        generalist_logits = torch.cat(logits_chunks, dim=0)
        packet = torch.load(packet_path, map_location="cpu")
        packet["has_generalist_logits"] = True
        packet["generalist_logits"] = generalist_logits.half()
        packet["generalist_logit_dim"] = int(generalist_logits.shape[1])
        packet["generalist_logit_num_images"] = int(generalist_logits.shape[0])
        packet["generalist_logit_dtype"] = "float16"
        packet["generalist_model"] = str(getattr(g_args, "model_name", ""))
        packet["generalist_checkpoint"] = str(ckpt_path)
        torch.save(packet, packet_path)
        print(
            f"[attach_generalist_logits] agent={agent_id} method={packet_method} "
            f"shape={tuple(packet['generalist_logits'].shape)} path={packet_path}"
        )
