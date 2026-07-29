import csv
import fcntl
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from agent_data import get_agent_class_split, get_agent_dir, get_agent_model_split, get_receiver_dir, get_run_dir
from output_manager import atomic_copyfile, atomic_write_json
from packet_integrity import (
    STRICT_DKP_PROTOCOL,
    file_sha256,
    strict_packet_validation_enabled,
    validate_strict_dkp_packet,
    validate_strict_manifest_rows,
)


MANIFEST_FIELDS = ["sender_agent", "sender_model", "classes", "method", "ipc", "packet_path"]
STRICT_MANIFEST_FIELDS = MANIFEST_FIELDS + [
    "protocol",
    "complete",
    "packet_sha256",
    "pool_protocol",
    "pool_source_sha256",
    "pool_source_best_snapshot_sha256",
    "decoded_images_sha256",
    "decoded_labels_sha256",
    "sender_logits_sha256",
    "decoded_alignment_sha256",
    "expert_checkpoint_sha256",
]
SOCIAL_RESULT_FIELDS = [
    "run_name",
    "protocol",
    "dkp_variant",
    "receiver_agent",
    "receiver_model",
    "expert_classes",
    "packet_method",
    "method",
    "init_mode",
    "self_data_mode",
    "local_ce_source",
    "local_ce_real_fraction",
    "self_real_per_class",
    "self_packet_raw_images",
    "self_packet_decoded_images",
    "self_packet_path",
    "self_packet_sha256",
    "use_fr",
    "lambda_fr",
    "lambda_sc",
    "supcon_temperature",
    "use_logits",
    "communication_mode",
    "use_generalist_logits",
    "kd_mix_beta",
    "lambda_kd",
    "kd_temperature",
    "ipc",
    "self_real_images",
    "external_comm_images",
    "external_comm_logit_bytes",
    "external_comm_generalist_logit_bytes",
    "acc_global_before",
    "acc_expert_before",
    "acc_global_after",
    "acc_expert_after",
    "acc_new_after",
    "forgetting",
    "loss_cls",
    "loss",
    "loss_ce_local",
    "loss_ce_local_real",
    "loss_ce_local_packet",
    "loss_ce_external",
    "loss_fr",
    "loss_kd",
    "loss_sc",
    "loss_sender_kd",
    "loss_generalist_kd",
    "receiver_augment",
    "freeze_bn_stats",
    "optimizer_steps",
    "target_optimizer_steps",
    "scheduler_unit",
    "scheduler_step_milestones",
    "training_seconds",
    "prototype_initialized_classes",
    "classifier_type",
    "prototype_init_mode",
    "prototype_alpha",
    "prototype_beta",
    "prototype_weight_norm_min",
    "prototype_weight_norm_max",
    "checkpoint_retention",
    "after_social_checkpoint_sha256",
    "receiver_provenance_path",
    "receiver_provenance_sha256",
    "receiver_seed",
    "receiver_init_checkpoint_sha256",
    "fr_teacher_checkpoint_sha256",
    "time",
]

OLD_SOCIAL_RESULT_FIELDS = [
    "run_name",
    "receiver_agent",
    "receiver_model",
    "expert_classes",
    "method",
    "ipc",
    "external_comm_images",
    "acc_global_before",
    "acc_expert_before",
    "acc_global_after",
    "acc_expert_after",
    "acc_new_after",
    "forgetting",
    "loss_cls",
    "loss_fr",
    "time",
]


def prepare_social_output_dirs(args):
    """创建第二阶段社会化学习输出目录。"""
    run_dir = get_run_dir(args)
    for subdir in ["agents", "packet_hub", "social_learning", "metrics"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    for agent_id in get_agent_class_split(args):
        for subdir in ["checkpoints", "packets", "synthetic", "visuals", "metrics", "logs"]:
            (get_agent_dir(args, agent_id) / subdir).mkdir(parents=True, exist_ok=True)
        for subdir in ["checkpoints", "metrics", "logs"]:
            (get_receiver_dir(args, agent_id) / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_social_config(config_path, args):
    """Atomically create or exactly reuse immutable social-run config snapshots."""
    config_dir = get_run_dir(args) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config_dir / ".snapshot.lock"
    with open(lock_path, "a", encoding="ascii") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if config_path:
                source = Path(config_path)
                target = config_dir / "main.yaml"
                if target.exists():
                    if file_sha256(source) != file_sha256(target):
                        raise FileExistsError(f"Existing social config snapshot differs: {target}")
                else:
                    atomic_copyfile(source, target)
            resolved_path = config_dir / "social_resolved_args.json"
            expected_args = json.loads(json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
            if resolved_path.exists():
                try:
                    existing_args = json.loads(resolved_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(f"Existing resolved-args snapshot is invalid: {resolved_path}") from error
                if existing_args != expected_args:
                    raise FileExistsError(f"Existing resolved-args snapshot differs: {resolved_path}")
            else:
                atomic_write_json(expected_args, resolved_path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def get_packet_hub_dir(args):
    """返回 packet_hub 目录。"""
    return get_run_dir(args) / "packet_hub"


def get_method_packet_hub_dir(args, packet_method):
    """返回指定 packet 方法的 hub 目录。"""
    return get_packet_hub_dir(args) / packet_method


def get_manifest_path(args, packet_method="dsdm"):
    """返回指定 packet 方法的 manifest 路径。"""
    return get_method_packet_hub_dir(args, packet_method) / "packet_manifest.csv"


def get_social_results_path(args):
    """返回 social_results.csv 路径。"""
    return get_run_dir(args) / "metrics" / "social_results.csv"


def register_agent_packet(args, agent_id, packet_path, packet_method="dsdm"):
    """把单个 agent 的 packet 复制到 packet_hub 并返回 manifest 行。"""
    class_split = get_agent_class_split(args)
    model_split = get_agent_model_split(args)
    hub_dir = get_method_packet_hub_dir(args, packet_method)
    hub_dir.mkdir(parents=True, exist_ok=True)
    dst = hub_dir / f"agent_{int(agent_id)}_{packet_method}_packet.pt"
    strict = strict_packet_validation_enabled(args)
    if strict and dst.exists():
        if file_sha256(packet_path) != file_sha256(dst):
            raise FileExistsError(f"Refusing to overwrite an existing strict packet-hub artifact: {dst}")
    else:
        atomic_copyfile(packet_path, dst)
    row = {
        "sender_agent": int(agent_id),
        "sender_model": model_split[int(agent_id)],
        "classes": ",".join(str(c) for c in class_split[int(agent_id)]),
        "method": packet_method.upper(),
        "ipc": int(args.ipc),
        "packet_path": str(dst),
    }
    if strict:
        packet = torch.load(dst, map_location="cpu", weights_only=False)
        from packet_logits import _ensure_dsdm_path

        _ensure_dsdm_path()
        from packet_consumer import _decode_dsdm_images

        decoded_images, decoded_labels = _decode_dsdm_images(args, packet)
        proof = validate_strict_dkp_packet(
            args,
            packet,
            decoded_images,
            decoded_labels,
            sender_agent=agent_id,
            sender_model=model_split[int(agent_id)],
            require_sender_logits=bool(getattr(args, "use_sender_logits", False)),
        )
        meta = packet.get("meta", {})
        row.update(
            {
                "protocol": str(getattr(args, "communication_protocol", STRICT_DKP_PROTOCOL)),
                "complete": "true",
                "packet_sha256": file_sha256(dst),
                "pool_protocol": str(meta.get("pool_protocol", "")),
                "pool_source_sha256": str(meta.get("pool_source_sha256", "")),
                "pool_source_best_snapshot_sha256": str(meta.get("pool_source_best_snapshot_sha256", "")),
                "decoded_images_sha256": proof["decoded_images_sha256"],
                "decoded_labels_sha256": proof["decoded_labels_sha256"],
                "sender_logits_sha256": proof.get("sender_logits_sha256", ""),
                "decoded_alignment_sha256": proof.get("decoded_alignment_sha256", ""),
                "expert_checkpoint_sha256": proof.get("checkpoint_sha256") or "",
            }
        )
    return row


def write_packet_manifest(args, rows, packet_method="dsdm"):
    """写入 packet_hub 的 packet_manifest.csv。"""
    path = get_manifest_path(args, packet_method)
    path.parent.mkdir(parents=True, exist_ok=True)
    strict = strict_packet_validation_enabled(args)
    if strict:
        rows = sorted(rows, key=lambda row: int(row["sender_agent"]))
        validate_strict_manifest_rows(args, rows, packet_method)
    fields = STRICT_MANIFEST_FIELDS if strict else MANIFEST_FIELDS
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def read_packet_manifest(args, packet_method="dsdm"):
    """读取 packet_manifest.csv 并返回字典列表。"""
    path = get_manifest_path(args, packet_method)
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if strict_packet_validation_enabled(args):
        validate_strict_manifest_rows(args, rows, packet_method)
    return rows


def append_social_result(args, row):
    """Append one result through a process lock and atomic whole-file replace."""
    path = get_social_results_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {field: row.get(field, "") for field in SOCIAL_RESULT_FIELDS}
    clean["run_name"] = clean["run_name"] or args.run_name
    clean["time"] = clean["time"] or datetime.now().isoformat(timespec="seconds")
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a", encoding="ascii") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            migrate_social_results_schema(path)
            existing_rows = []
            if path.exists():
                with open(path, "r", encoding="utf-8", newline="") as source:
                    existing_rows = list(csv.DictReader(source))
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                mode="w",
                encoding="utf-8",
                newline="",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=SOCIAL_RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {field: existing.get(field, "") for field in SOCIAL_RESULT_FIELDS}
                    for existing in existing_rows
                )
                writer.writerow(clean)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return path


def migrate_social_results_schema(path):
    """把旧版或错位的 social_results.csv 迁移到当前字段顺序。"""
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == SOCIAL_RESULT_FIELDS:
            return False
        rows = list(reader)
        old_fields = reader.fieldnames or []

    migrated = []
    for row in rows:
        migrated.append(_migrate_social_result_row(row, old_fields))

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False
    ) as f:
        temp_path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=SOCIAL_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(migrated)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return True


def _migrate_social_result_row(row, old_fields):
    """迁移单行 social result，兼容旧表头和新行错位追加。"""
    extra = row.get(None)
    if extra:
        values = [row.get(field, "") for field in old_fields] + list(extra)
        restored = {field: values[index] if index < len(values) else "" for index, field in enumerate(SOCIAL_RESULT_FIELDS)}
        return restored

    clean = {field: "" for field in SOCIAL_RESULT_FIELDS}
    for field in SOCIAL_RESULT_FIELDS:
        if field in row:
            clean[field] = row.get(field, "")
    old_method = row.get("method", "")
    clean["packet_method"] = row.get("packet_method", old_method.lower() if old_method else "")
    clean["method"] = old_method
    clean["init_mode"] = row.get("init_mode", "expert")
    clean["self_data_mode"] = row.get("self_data_mode", "packet")
    clean["self_real_per_class"] = row.get("self_real_per_class", "0")
    clean["use_fr"] = row.get("use_fr", "true")
    clean["lambda_fr"] = row.get("lambda_fr", "0.05")
    clean["use_logits"] = row.get("use_logits", "false")
    clean["communication_mode"] = row.get("communication_mode", "direct")
    clean["use_generalist_logits"] = row.get("use_generalist_logits", "false")
    clean["kd_mix_beta"] = row.get("kd_mix_beta", "0.0")
    clean["lambda_kd"] = row.get("lambda_kd", "0.0")
    clean["kd_temperature"] = row.get("kd_temperature", "2.0")
    clean["self_real_images"] = row.get("self_real_images", "0")
    clean["external_comm_logit_bytes"] = row.get("external_comm_logit_bytes", "0")
    clean["external_comm_generalist_logit_bytes"] = row.get("external_comm_generalist_logit_bytes", "0")
    clean["loss_kd"] = row.get("loss_kd", "0.0")
    clean["loss_sender_kd"] = row.get("loss_sender_kd", "0.0")
    clean["loss_generalist_kd"] = row.get("loss_generalist_kd", "0.0")
    clean["receiver_augment"] = row.get("receiver_augment", "false")
    clean["freeze_bn_stats"] = row.get("freeze_bn_stats", "false")
    return clean
