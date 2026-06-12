import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from agent_data import AGENT_CLASS_SPLIT, AGENT_MODEL_SPLIT, get_agent_dir, get_receiver_dir, get_run_dir


MANIFEST_FIELDS = ["sender_agent", "sender_model", "classes", "method", "ipc", "packet_path"]
SOCIAL_RESULT_FIELDS = [
    "run_name",
    "receiver_agent",
    "receiver_model",
    "expert_classes",
    "packet_method",
    "method",
    "init_mode",
    "use_fr",
    "lambda_fr",
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
    for agent_id in AGENT_CLASS_SPLIT:
        for subdir in ["checkpoints", "packets", "synthetic", "visuals", "metrics", "logs"]:
            (get_agent_dir(args, agent_id) / subdir).mkdir(parents=True, exist_ok=True)
        for subdir in ["checkpoints", "metrics", "logs"]:
            (get_receiver_dir(args, agent_id) / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_social_config(config_path, args):
    """保存第二阶段运行使用的配置快照。"""
    config_dir = get_run_dir(args) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, config_dir / "main.yaml")
    with open(config_dir / "social_resolved_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, sort_keys=True)


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
    hub_dir = get_method_packet_hub_dir(args, packet_method)
    hub_dir.mkdir(parents=True, exist_ok=True)
    dst = hub_dir / f"agent_{int(agent_id)}_{packet_method}_packet.pt"
    shutil.copyfile(packet_path, dst)
    return {
        "sender_agent": int(agent_id),
        "sender_model": AGENT_MODEL_SPLIT[int(agent_id)],
        "classes": ",".join(str(c) for c in AGENT_CLASS_SPLIT[int(agent_id)]),
        "method": packet_method.upper(),
        "ipc": int(args.ipc),
        "packet_path": str(dst),
    }


def write_packet_manifest(args, rows, packet_method="dsdm"):
    """写入 packet_hub 的 packet_manifest.csv。"""
    path = get_manifest_path(args, packet_method)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_packet_manifest(args, packet_method="dsdm"):
    """读取 packet_manifest.csv 并返回字典列表。"""
    path = get_manifest_path(args, packet_method)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_social_result(args, row):
    """追加单个 receiver 的社会化学习结果。"""
    path = get_social_results_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    migrate_social_results_schema(path)
    exists = path.exists()
    clean = {field: row.get(field, "") for field in SOCIAL_RESULT_FIELDS}
    clean["run_name"] = clean["run_name"] or args.run_name
    clean["time"] = clean["time"] or datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOCIAL_RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(clean)
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

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOCIAL_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(migrated)
    return True


def _migrate_social_result_row(row, old_fields):
    """迁移单行 social result，兼容旧表头和新行错位追加。"""
    extra = row.get(None)
    if extra:
        values = [row.get(field, "") for field in old_fields] + list(extra)
        restored = {field: values[index] if index < len(values) else "" for index, field in enumerate(SOCIAL_RESULT_FIELDS)}
        return restored

    clean = {field: "" for field in SOCIAL_RESULT_FIELDS}
    for field in OLD_SOCIAL_RESULT_FIELDS:
        if field in row:
            clean[field] = row.get(field, "")
    old_method = row.get("method", "")
    clean["packet_method"] = row.get("packet_method", old_method.lower() if old_method else "")
    clean["method"] = old_method
    clean["init_mode"] = row.get("init_mode", "expert")
    clean["use_fr"] = row.get("use_fr", "true")
    clean["lambda_fr"] = row.get("lambda_fr", "0.05")
    return clean
