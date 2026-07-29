import json
from pathlib import Path

import torch

from agent_data import get_agent_dir, get_num_classes
from output_manager import atomic_torch_save, atomic_write_json
from packet_consumer import _decode_dsdm_images
from packet_integrity import (
    decoded_digests,
    file_sha256,
    strict_packet_validation_enabled,
    validate_strict_dkp_packet,
)


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.append(str(dsdm_root))


def _normalize_images(args, images, device):
    """按 DSDM 数据集均值方差归一化图片。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset], device=device).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset], device=device).view(1, -1, 1, 1)
    return (images.to(device).float() - mean) / std


def _load_sender_model(args, expert_path, device):
    """加载 sender expert model 并返回 eval 模型。"""
    from train import define_model

    model = define_model(args, get_num_classes(args)).to(device)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    model.eval()
    return model


def _packet_training_data(args, packet):
    """Return packet images and labels aligned with receiver training."""
    if packet.get("source") == "dsdm":
        return _decode_dsdm_images(args, packet)
    if packet.get("source") in {"heuristic", "importance", "fast", "full_real"}:
        return packet["images"].detach().cpu(), packet["labels"].detach().cpu()
    raise ValueError(f"不支持的 packet source: {packet.get('source')}")


def _teacher_provenance(agent_dir, expert_path, args, output_dim):
    """Load the converged-expert manifest without embedding its full history."""
    manifest_path = agent_dir / "checkpoints" / "expert_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        legacy_path = agent_dir / "checkpoints" / "expert_selection.json"
        manifest = {}
        if legacy_path.exists():
            with open(legacy_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        manifest_path = legacy_path
    checkpoint_sha256 = file_sha256(expert_path)
    manifest_sha = manifest.get(
        "checkpoint_sha256",
        manifest.get("expert_checkpoint_sha256", manifest.get("expert_sha256")),
    )
    if manifest_sha and str(manifest_sha) != checkpoint_sha256:
        raise ValueError(f"Expert manifest checkpoint SHA-256 mismatch: {manifest_path}")
    if strict_packet_validation_enabled(args):
        if not manifest_path.is_file() or not manifest_sha:
            raise ValueError(f"Strict DKP requires an expert manifest with checkpoint SHA-256: {manifest_path}")
        if int(manifest.get("global_output_dim", -1)) != int(output_dim):
            raise ValueError(f"Expert manifest global output dimension mismatch: {manifest_path}")
        if int(manifest.get("agent_id", -1)) != int(getattr(args, "agent_id", -1)):
            raise ValueError(f"Expert manifest agent id mismatch: {manifest_path}")
        if [int(value) for value in manifest.get("active_class_ids", [])] != [
            int(value) for value in getattr(args, "active_class_ids", [])
        ]:
            raise ValueError(f"Expert manifest active class order mismatch: {manifest_path}")
    return {
        "role": manifest.get("role", "legacy_or_unproven_expert"),
        "checkpoint": str(expert_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dim": int(output_dim),
        "model_id": str(getattr(args, "model_name", getattr(args, "sender_model", ""))),
        "agent_id": int(getattr(args, "agent_id", -1)),
        "class_ids": [int(value) for value in getattr(args, "active_class_ids", [])],
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "selected_epoch": manifest.get("selected_epoch"),
        "best_validation_accuracy": manifest.get("best_validation_accuracy"),
        "official_test_accuracy_report_only": manifest.get(
            "official_test_accuracy_report_only", manifest.get("selected_expert_acc")
        ),
        "test_used_for_selection": bool(manifest.get("test_used_for_selection", True)),
    }


def attach_sender_logits_to_packet(args, agent_id, packet_method, *, resume=False, overwrite=False):
    """为指定 agent 的 packet 附加 sender expert-class logits。"""
    _ensure_dsdm_path()
    agent_dir = get_agent_dir(args, agent_id)
    packet_path = agent_dir / "packets" / f"{packet_method}_packet.pt"
    expert_root = getattr(args, "expert_source_root", None)
    expert_agent_dir = agent_dir
    if expert_root:
        expert_agent_dir = Path(expert_root) / "agents" / f"agent_{int(agent_id)}"
    expert_path = expert_agent_dir / "checkpoints" / "expert_model.pt"
    if not packet_path.exists():
        raise FileNotFoundError(f"缺少 agent packet: {packet_path}")
    if not expert_path.exists():
        raise FileNotFoundError(f"缺少 sender expert model: {expert_path}")

    packet = torch.load(packet_path, map_location="cpu", weights_only=False)
    strict = strict_packet_validation_enabled(args)
    if strict and overwrite:
        raise ValueError("Strict DKP logit attachment forbids overwrite; use a new run_name")
    sender_class_ids = [int(c) for c in packet.get("class_ids", [])]
    if not sender_class_ids:
        raise ValueError(f"packet 缺少 class_ids: {packet_path}")

    if strict and sender_class_ids != [int(value) for value in getattr(args, "active_class_ids", [])]:
        raise ValueError("Packet class_ids do not equal the configured sender class order")
    pool_source = packet.get("meta", {}).get("pool_source_packet")
    if strict and pool_source and Path(pool_source).resolve() == packet_path.resolve():
        raise ValueError("Refusing to attach logits directly to the immutable full-class source pool")

    images, labels = _packet_training_data(args, packet)
    if strict:
        validate_strict_dkp_packet(
            args,
            packet,
            images,
            labels,
            sender_agent=agent_id,
            sender_model=getattr(args, "model_name", getattr(args, "sender_model", "")),
            require_sender_logits=False,
        )
    if packet.get("has_sender_logits", False):
        if strict and resume:
            proof = validate_strict_dkp_packet(
                args,
                packet,
                images,
                labels,
                sender_agent=agent_id,
                sender_model=getattr(args, "model_name", getattr(args, "sender_model", "")),
                require_sender_logits=True,
            )
            current_sha = file_sha256(expert_path)
            if proof["checkpoint_sha256"] != current_sha:
                raise ValueError("Existing sender logits were produced by a different expert checkpoint")
            print(f"[attach_logits] reuse validated sender_logits: {packet_path}")
            return packet_path
        if strict:
            raise FileExistsError(
                f"Packet already has sender logits; use --resume after validation or a new run_name: {packet_path}"
            )
        print(f"[attach_logits] 覆盖已有 sender_logits: {packet_path}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_sender_model(args, expert_path, device)
    batch_size = int(getattr(args, "batch_size", 64))
    class_index = torch.tensor(sender_class_ids, device=device, dtype=torch.long)
    logits_chunks = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = _normalize_images(args, images[start : start + batch_size], device)
            full_logits = model(batch)
            if full_logits.ndim != 2 or int(full_logits.shape[1]) != int(get_num_classes(args)):
                raise ValueError(
                    f"Sender expert output must be [N,{get_num_classes(args)}], got {tuple(full_logits.shape)}"
                )
            logits_chunks.append(full_logits[:, class_index].detach().cpu())
    sender_logits = torch.cat(logits_chunks, dim=0)
    if not bool(torch.isfinite(sender_logits).all().item()):
        raise FloatingPointError(f"sender logits 包含 NaN/Inf: {packet_path}")
    probabilities = torch.softmax(sender_logits.float(), dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1).values
    margin = top2[:, 0] if top2.shape[1] == 1 else top2[:, 0] - top2[:, 1]
    predicted_global = class_index.detach().cpu()[sender_logits.argmax(dim=1)]
    label_agreement = float((predicted_global == labels.long()).float().mean().item())
    provenance = _teacher_provenance(expert_agent_dir, expert_path, args, get_num_classes(args))
    persisted_logits = sender_logits.half()
    integrity = decoded_digests(images, labels, persisted_logits)
    quality = {
        "agent_id": int(agent_id),
        "packet_method": str(packet_method),
        "num_images": int(sender_logits.shape[0]),
        "num_expert_classes": int(sender_logits.shape[1]),
        "finite": True,
        "packet_label_agreement": label_agreement,
        "mean_entropy": float(entropy.mean().item()),
        "mean_top1_margin": float(margin.mean().item()),
        "mean_abs_logit": float(sender_logits.abs().mean().item()),
        "max_abs_logit": float(sender_logits.abs().max().item()),
        "teacher": provenance,
        "decoded_integrity": integrity,
    }

    packet["has_sender_logits"] = True
    packet["sender_logits"] = persisted_logits
    packet["sender_logit_class_ids"] = torch.tensor(sender_class_ids, dtype=torch.long)
    packet["sender_logit_dim"] = len(sender_class_ids)
    packet["sender_logit_num_images"] = int(sender_logits.shape[0])
    packet["sender_logit_dtype"] = "float16"
    packet["sender_logit_teacher"] = provenance
    packet["sender_logit_quality"] = quality
    packet["decoded_integrity"] = integrity
    packet.setdefault("meta", {})["sender_logit_teacher"] = provenance
    if strict:
        validate_strict_dkp_packet(
            args,
            packet,
            images,
            labels,
            sender_agent=agent_id,
            sender_model=getattr(args, "model_name", getattr(args, "sender_model", "")),
            require_sender_logits=True,
        )
    atomic_torch_save(packet, packet_path)
    atomic_write_json(quality, agent_dir / "metrics" / f"logit_quality_{packet_method}.json")
    print(
        f"[attach_logits] agent={agent_id} method={packet_method} "
        f"shape={tuple(packet['sender_logits'].shape)} path={packet_path}"
    )
    return packet_path
