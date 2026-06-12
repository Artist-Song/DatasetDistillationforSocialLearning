import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from config_adapter import build_dsdm_args_from_config, load_config
from packet_consumer import consume_packet_for_training
from social_output_manager import get_manifest_path, get_run_dir, read_packet_manifest


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def parse_cli():
    """解析 packet 有效性验证命令行参数。"""
    parser = argparse.ArgumentParser(description="验证 packet 数量、标签分布和 decode 状态")
    parser.add_argument("--config", default="configs/main.yaml", help="主配置文件路径")
    parser.add_argument("--packet-method", default="dsdm", choices=["dsdm", "heuristic", "importance"], help="packet 方法")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不读取 packet")
    return parser.parse_args()


def _count_labels(labels):
    """统计标签分布并返回普通字典。"""
    counter = Counter(int(x) for x in labels.tolist())
    return {str(k): int(counter.get(k, 0)) for k in sorted(counter)}


def _metrics_paths(args, packet_method):
    """返回 packet 有效性验证的 JSON 和 CSV 输出路径。"""
    metrics_dir = get_run_dir(args) / "metrics"
    return (
        metrics_dir / f"packet_integrity_{packet_method}.json",
        metrics_dir / f"packet_integrity_{packet_method}.csv",
    )


def _expected_dsdm_summary(args):
    """返回当前 DSDM 设置下的期望数量。"""
    classes_per_agent = 2
    num_agents = 5
    raw_per_agent = classes_per_agent * int(args.ipc)
    train_per_agent = raw_per_agent * int(args.factor) ** 2
    return {
        "raw_per_agent": raw_per_agent,
        "train_per_agent": train_per_agent,
        "total_raw_images": num_agents * raw_per_agent,
        "total_train_images": num_agents * train_per_agent,
        "per_class_train_images": int(args.ipc) * int(args.factor) ** 2,
    }


def _build_warning(args, packet_method, summary):
    """根据期望数量生成 warning 列表。"""
    warnings = []
    if packet_method != "dsdm":
        return warnings
    expected = _expected_dsdm_summary(args)
    for key in ["total_raw_images", "total_train_images"]:
        if int(summary.get(key, -1)) != int(expected[key]):
            warnings.append(f"{key}={summary.get(key)}，期望 {expected[key]}")
    for class_id in range(10):
        got = int(summary.get("per_class_train_images", {}).get(str(class_id), 0))
        if got != int(expected["per_class_train_images"]):
            warnings.append(f"class {class_id} train images={got}，期望 {expected['per_class_train_images']}")
    return warnings


def validate_packets(args, packet_method):
    """读取 manifest 并统计每个 packet 与合并后的有效性指标。"""
    rows = read_packet_manifest(args, packet_method)
    packet_rows = []
    total_raw = 0
    total_train = 0
    per_class_raw = Counter()
    per_class_train = Counter()

    for row in rows:
        packet_path = row["packet_path"]
        packet = torch.load(packet_path, map_location="cpu")
        consumed = consume_packet_for_training(args, packet_path)
        raw_labels = packet["labels"].long()
        train_labels = consumed["labels"].long()
        raw_dist = _count_labels(raw_labels)
        train_dist = _count_labels(train_labels)
        per_class_raw.update({int(k): v for k, v in raw_dist.items()})
        per_class_train.update({int(k): v for k, v in train_dist.items()})
        total_raw += int(consumed["raw_images"])
        total_train += int(consumed["num_images"])
        packet_rows.append(
            {
                "sender_agent": row["sender_agent"],
                "sender_model": row["sender_model"],
                "packet_method": packet_method,
                "class_ids": ",".join(str(c) for c in packet.get("class_ids", [])),
                "raw_images": int(consumed["raw_images"]),
                "decoded_or_train_images": int(consumed["num_images"]),
                "raw_label_distribution": json.dumps(raw_dist, ensure_ascii=False),
                "train_label_distribution": json.dumps(train_dist, ensure_ascii=False),
                "decoded_for_training": bool(consumed["decoded_for_training"]),
            }
        )

    summary = {
        "packet_method": packet_method,
        "total_raw_images": int(total_raw),
        "total_train_images": int(total_train),
        "per_class_raw_images": {str(k): int(per_class_raw.get(k, 0)) for k in range(10)},
        "per_class_train_images": {str(k): int(per_class_train.get(k, 0)) for k in range(10)},
    }
    warnings = _build_warning(args, packet_method, summary)
    result = {"summary": summary, "packets": packet_rows, "warnings": warnings}
    json_path, csv_path = _metrics_paths(args, packet_method)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sender_agent",
            "sender_model",
            "packet_method",
            "class_ids",
            "raw_images",
            "decoded_or_train_images",
            "raw_label_distribution",
            "train_label_distribution",
            "decoded_for_training",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packet_rows)
    return result, json_path, csv_path


def main():
    """执行 packet 有效性验证入口。"""
    cli = parse_cli()
    cfg = load_config(cli.config)
    args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    manifest_path = get_manifest_path(args, cli.packet_method)
    json_path, csv_path = _metrics_paths(args, cli.packet_method)
    if cli.dry_run:
        print(f"packet_method: {cli.packet_method}")
        print(f"manifest: {manifest_path}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return
    result, json_path, csv_path = validate_packets(args, cli.packet_method)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
