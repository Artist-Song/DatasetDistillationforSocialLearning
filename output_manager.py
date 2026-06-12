import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch


PACKET_SOURCES = {"dsdm", "heuristic", "importance"}
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
        "packets",
        "visuals",
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
    }
    torch.save(payload, path)
    return path


def _validate_packet_payload(payload):
    """检查 packet 中是否包含禁止字段。"""
    banned = {
        "model_state_dict",
        "teacher_logits",
        "teacher_probs",
        "soft_targets",
        "gradients",
        "optimizer_state",
    }
    overlap = banned.intersection(payload.keys())
    if overlap:
        raise ValueError(f"packet 包含禁止字段: {sorted(overlap)}")


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
    torch.save(payload, path)
    return path


def _save_grid(path, images, args, unnormalize=False):
    """保存一组图片网格，可用于 packet 可视化。"""
    from data import save_img

    save_img(str(path), images, unnormalize=unnormalize, dataname=args.dataset)
    return path


def save_init_visual(args, synset):
    """保存初始蒸馏图的可视化。"""
    return _save_grid(get_run_dir(args) / "visuals" / "init.png", synset.data, args)


def save_aug_visual(args, images):
    """保存增强后图片的可视化。"""
    return _save_grid(get_run_dir(args) / "visuals" / "aug.png", images, args, unnormalize=True)


def save_best_visuals(args, images, labels):
    """保存最优蒸馏图整体和按类别可视化。"""
    run_dir = get_run_dir(args)
    _save_grid(run_dir / "visuals" / "best_grid.png", images, args)
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
