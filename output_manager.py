import csv
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import torch


PACKET_SOURCES = {"dsdm", "heuristic", "importance", "fast", "full_real"}
GLOBAL_RESULT_FIELDS = [
    "run_name",
    "stage",
    "dataset",
    "method",
    "source",
    "ipc",
    "model",
    "best_acc",
    "last_acc",
    "train_acc",
    "packet_path",
    "config_path",
    "time",
]


def atomic_torch_save(payload, path):
    """先写同目录临时文件再原子替换，避免破坏硬链接指向的历史 artifact。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def atomic_copyfile(source, target):
    """复制到同目录临时文件后原子替换目标。"""
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_write_json(payload, path):
    """Write JSON through a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def get_run_dir(args):
    """根据 output_root 和 run_name 返回本次实验目录。"""
    return Path(args.output_root) / args.run_name


def get_packet_path(args, source):
    """根据 packet 来源返回统一 packet 文件路径。"""
    if source not in PACKET_SOURCES:
        raise ValueError(f"不支持的 packet source: {source}")
    return get_run_dir(args) / "packets" / f"{source}_packet.pt"


def get_metrics_path(args, name):
    """根据指标名称返回 metrics JSON 文件路径。"""
    return get_run_dir(args) / "metrics" / f"{name}.json"


def prepare_output_dirs(args):
    """创建本次实验需要的全部输出目录。"""
    run_dir = get_run_dir(args)
    for subdir in [
        "config",
        "logs",
        "checkpoints",
        "synthetic",
        "synthetic/history",
        "packets",
        "visuals",
        "visuals/history",
        "visuals/per_class",
        "metrics",
        "embeddings/tsne",
    ]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg, args):
    """保存当前主配置快照到实验目录。"""
    path = get_run_dir(args) / "config" / "main.yaml"
    try:
        import yaml
    except ModuleNotFoundError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def save_config_file_snapshot(config_path, args):
    """复制原始配置文件到实验目录，保留用户写法。"""
    dst = get_run_dir(args) / "config" / "main.yaml"
    if config_path is not None:
        shutil.copyfile(config_path, dst)
    return dst


def save_resolved_args(args):
    """保存解析后的 DSDM 参数到 JSON。"""
    path = get_run_dir(args) / "config" / "resolved_args.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def save_best_synthetic(args, synset, best_acc, iteration):
    """保存 DSDM 当前最优 synthetic data。"""
    path = get_run_dir(args) / "synthetic" / "data_best.pt"
    payload = {
        "images": synset.data.detach().cpu(),
        "labels": synset.targets.detach().cpu(),
        "best_acc": best_acc,
        "iteration": iteration,
        "dataset": args.dataset,
        "ipc": args.ipc,
        "pcbn_enabled": bool(getattr(args, "pcbn_enabled", False)),
        "pcbn_weight": float(getattr(args, "pcbn_weight", 0.0)),
        "pcbn_layers": getattr(args, "pcbn_layers", "all"),
        "pcbn_normalize_layers": bool(getattr(args, "pcbn_normalize_layers", True)),
        "pcbn_hook_count": int(getattr(args, "pcbn_hook_count", 0)),
    }
    atomic_torch_save(payload, path)
    history_path = get_run_dir(args) / "synthetic" / "history" / f"best_iter_{int(iteration):05d}.pt"
    atomic_torch_save(payload, history_path)
    atomic_write_json(
        {
            "best_acc": float(best_acc),
            "iteration": int(iteration),
            "latest_best": str(path),
            "history_snapshot": str(history_path),
            "pcbn_enabled": payload["pcbn_enabled"],
            "pcbn_weight": payload["pcbn_weight"],
            "pcbn_layers": payload["pcbn_layers"],
            "pcbn_normalize_layers": payload["pcbn_normalize_layers"],
            "pcbn_hook_count": payload["pcbn_hook_count"],
        },
        get_run_dir(args) / "synthetic" / "best_manifest.json",
    )
    return path


def _validate_packet_payload(payload):
    """检查 packet 中是否包含不允许通信的模型级状态。"""
    banned = {
        "model_state_dict",
        "gradients",
        "optimizer_state",
        "features",
        "feature_maps",
        "activations",
    }
    overlap = banned.intersection(payload.keys())
    if overlap:
        raise ValueError(f"packet 包含禁止字段: {sorted(overlap)}")
    for name in ("images", "sender_logits", "generalist_logits"):
        tensor = payload.get(name)
        if torch.is_tensor(tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"packet 包含非有限 {name}")


def save_packet(args, images, labels, class_ids, source, method, meta=None):
    """保存统一格式的 hard-label image packet。"""
    if source not in PACKET_SOURCES:
        raise ValueError(f"不支持的 packet source: {source}")
    packet_meta = {"run_name": args.run_name, "method": method}
    if meta:
        packet_meta.update(meta)
    payload = {
        "images": images.detach().cpu(),
        "labels": labels.detach().cpu(),
        "class_ids": [int(c) for c in class_ids],
        "source": source,
        "dataset": args.dataset,
        "ipc": int(args.ipc),
        "factor": int(getattr(args, "factor", 1)) if source == "dsdm" else 1,
        "decode_type": getattr(args, "decode_type", "single") if source == "dsdm" else "none",
        "packet_format": "compact_multi_formation" if source == "dsdm" else "raw_images",
        "meta": packet_meta,
    }
    _validate_packet_payload(payload)
    path = get_packet_path(args, source)
    atomic_torch_save(payload, path)
    return path


def finalize_dsdm_packet(packet_path, numerical_summary):
    """Mark a finite DSDM packet complete only after all condensation iterations return."""
    packet_path = Path(packet_path)
    payload = torch.load(packet_path, map_location="cpu")
    _validate_packet_payload(payload)
    meta = dict(payload.get("meta", {}))
    meta.update(
        {
            "condense_complete": True,
            "completed_iterations": int(numerical_summary["completed_iterations"]),
            "grad_clip_norm": float(numerical_summary["grad_clip_norm"]),
            "grad_clip_count": int(numerical_summary["grad_clip_count"]),
            "max_grad_norm": float(numerical_summary["max_grad_norm"]),
        }
    )
    payload["meta"] = meta
    _validate_packet_payload(payload)
    atomic_torch_save(payload, packet_path)
    return packet_path


def _save_grid(path, images, args, unnormalize=False):
    """保存一组图片网格，可用于 packet 可视化。"""
    from data import save_img

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_img(str(path), images, unnormalize=unnormalize, dataname=args.dataset)
    return path


def save_init_visual(args, synset):
    """保存初始蒸馏图的可视化。"""
    return _save_grid(get_run_dir(args) / "visuals" / "init.png", synset.data, args)


def save_aug_visual(args, images):
    """保存增强后图片的可视化。"""
    return _save_grid(get_run_dir(args) / "visuals" / "aug.png", images, args, unnormalize=True)


def save_best_visuals(args, images, labels, iteration=None):
    """保存最优蒸馏图整体和按类别可视化。"""
    run_dir = get_run_dir(args)
    _save_grid(run_dir / "visuals" / "best_grid.png", images, args)
    if iteration is not None:
        _save_grid(
            run_dir / "visuals" / "history" / f"best_iter_{int(iteration):05d}.png",
            images,
            args,
        )
    for class_id in sorted({int(x) for x in labels.detach().cpu().tolist()}):
        mask = labels.detach().cpu() == class_id
        if mask.any():
            _save_grid(run_dir / "visuals" / "per_class" / f"class_{class_id}.png", images[mask], args)
    return run_dir / "visuals" / "best_grid.png"


def save_packet_visual(args, source, images):
    """保存指定 packet 来源的图片网格。"""
    return _save_grid(get_run_dir(args) / "visuals" / f"{source}_grid.png", images, args)


def save_metrics(args, name, metrics):
    """保存单次实验或 packet 评估指标。"""
    path = get_metrics_path(args, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def append_global_result(args, row):
    """向全局结果表追加一行汇总结果。"""
    path = Path(args.output_root) / "global_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    clean_row = {field: row.get(field, "") for field in GLOBAL_RESULT_FIELDS}
    clean_row["run_name"] = clean_row["run_name"] or args.run_name
    clean_row["dataset"] = clean_row["dataset"] or args.dataset
    clean_row["ipc"] = clean_row["ipc"] or args.ipc
    clean_row["model"] = clean_row["model"] or args.net_type
    clean_row["config_path"] = clean_row["config_path"] or str(get_run_dir(args) / "config" / "main.yaml")
    clean_row["time"] = clean_row["time"] or datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GLOBAL_RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(clean_row)
    return path
