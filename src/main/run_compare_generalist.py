"""
Compare local specialists with generalist-packet social-head specialists.
"""

import argparse
import csv
import json
from pathlib import Path

from src.datasets.splits import get_partial_split
from src.utils.config import load_yaml
from src.utils.experiment import assert_report_experiment, get_experiment_id, get_experiment_metadata, get_experiment_root, save_experiment_files, validate_reuse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


def load_report(report_dir: Path, stage: str, agent_suffix: str):
    report_path = report_dir / f"eval_{stage}_{agent_suffix}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"eval report not found: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f), report_path


def estimate_communication(cfg, num_agents: int, missing_per_agent: int):
    packet_cfg = cfg.get("packet", {})
    dataset_cfg = cfg["dataset"]
    image_h, image_w = dataset_cfg["image_size"]
    image_channels = packet_cfg.get("image_channels", 3)
    bytes_per_float = packet_cfg.get("bytes_per_float", 4)
    ipc = packet_cfg.get("ipc", 10)
    num_classes = dataset_cfg["num_classes"]
    train_samples = packet_cfg.get("stage1_train_samples", 50000 if dataset_cfg["name"].lower() == "cifar10" else 0)

    image_bytes = image_channels * image_h * image_w * bytes_per_float
    soft_target_bytes = num_classes * bytes_per_float
    per_packet_sample_bytes = image_bytes + soft_target_bytes

    stage1_bytes = train_samples * image_bytes
    stage2_samples = num_agents * missing_per_agent * ipc
    stage2_bytes = stage2_samples * per_packet_sample_bytes
    return {
        "stage1_generalist_training_bytes": stage1_bytes,
        "stage1_generalist_training_mib": stage1_bytes / (1024 * 1024),
        "stage2_missing_packet_samples": stage2_samples,
        "stage2_missing_packet_bytes": stage2_bytes,
        "stage2_missing_packet_mib": stage2_bytes / (1024 * 1024),
        "assumptions": {
            "image_channels": image_channels,
            "bytes_per_float": bytes_per_float,
            "ipc": ipc,
            "soft_targets_per_sample": num_classes,
            "stage1_train_samples": train_samples,
        },
    }


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_agents_csv(path: Path, rows):
    fieldnames = [
        "experiment_id",
        "agent_id",
        "known_accuracy",
        "missing_accuracy",
        "general_accuracy",
        "delta_known_accuracy",
        "delta_missing_accuracy",
        "delta_general_accuracy",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row["experiment_id"],
                    "agent_id": row["agent_id"],
                    "known_accuracy": row["social_head"]["known_accuracy"],
                    "missing_accuracy": row["social_head"]["missing_accuracy"],
                    "general_accuracy": row["social_head"]["general_accuracy"],
                    "delta_known_accuracy": row["delta_known_accuracy"],
                    "delta_missing_accuracy": row["delta_missing_accuracy"],
                    "delta_general_accuracy": row["delta_general_accuracy"],
                }
            )


def write_metrics_csv(path: Path, row):
    fieldnames = list(row.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    validate_reuse(cfg, args.config)
    experiment_id = get_experiment_id(cfg, args.config)
    experiment_root = get_experiment_root(cfg, args.config)
    experiment = get_experiment_metadata(cfg, args.config)
    agent_suffix = "all" if args.agent_ids == "all" else args.agent_ids.replace(",", "_").replace("-", "to")
    report_dir = experiment_root / "reports" / "generalist_packet"

    local_report, local_path = load_report(report_dir, "local", agent_suffix)
    social_report, social_path = load_report(report_dir, "social_head", agent_suffix)
    assert_report_experiment(local_report, experiment_id, local_path)
    assert_report_experiment(social_report, experiment_id, social_path)
    local_by_agent = {item["agent_id"]: item for item in local_report["agents"]}

    agents = []
    for social_item in social_report["agents"]:
        agent_id = social_item["agent_id"]
        local_item = local_by_agent[agent_id]
        row = {
            "experiment_id": experiment_id,
            "agent_id": agent_id,
            "known_classes": social_item["known_classes"],
            "missing_classes": social_item["missing_classes"],
            "local": {
                "known_accuracy": local_item["known_accuracy"],
                "missing_accuracy": local_item["missing_accuracy"],
                "general_accuracy": local_item["general_accuracy"],
            },
            "social_head": {
                "known_accuracy": social_item["known_accuracy"],
                "missing_accuracy": social_item["missing_accuracy"],
                "general_accuracy": social_item["general_accuracy"],
            },
            "delta_known_accuracy": social_item["known_accuracy"] - local_item["known_accuracy"],
            "delta_missing_accuracy": social_item["missing_accuracy"] - local_item["missing_accuracy"],
            "delta_general_accuracy": social_item["general_accuracy"] - local_item["general_accuracy"],
        }
        agents.append(row)

    summary = {
        "known_accuracy": social_report["summary"]["known_accuracy"],
        "missing_accuracy": social_report["summary"]["missing_accuracy"],
        "general_accuracy": social_report["summary"]["general_accuracy"],
        "delta_known_accuracy": social_report["summary"]["known_accuracy"] - local_report["summary"]["known_accuracy"],
        "delta_missing_accuracy": social_report["summary"]["missing_accuracy"] - local_report["summary"]["missing_accuracy"],
        "delta_general_accuracy": social_report["summary"]["general_accuracy"] - local_report["summary"]["general_accuracy"],
    }
    splits = get_partial_split(cfg["split"]["name"])
    missing_per_agent = len(next(iter(splits.values())).missing)
    communication = estimate_communication(cfg, len(social_report["selected_agent_ids"]), missing_per_agent)

    results = {
        "experiment_id": experiment_id,
        "experiment": experiment,
        "dataset": cfg["dataset"]["name"],
        "split_name": cfg["split"]["name"],
        "local_report": str(local_path),
        "social_head_report": str(social_path),
        "agents": agents,
        "summary": summary,
        "communication": communication,
    }

    print("=== run_compare_generalist ===")
    print(f"experiment_id: {experiment_id}")
    print(f"experiment_root: {experiment_root}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"stage1_generalist_training_mib: {communication['stage1_generalist_training_mib']:.4f}")
    print(f"stage2_missing_packet_mib: {communication['stage2_missing_packet_mib']:.4f}")

    report_path = report_dir / f"compare_generalist_{agent_suffix}.json"
    save_experiment_files(cfg, args.config, {"compare_report_dir": str(report_dir)})
    write_json(report_path, results)
    summary_path = report_dir / "summary.json"
    write_json(
        summary_path,
        {
            "experiment_id": experiment_id,
            "experiment": experiment,
            "summary": summary,
            "communication": communication,
            "local_report": str(local_path),
            "social_head_report": str(social_path),
            "compare_report": str(report_path),
        },
    )
    metrics_row = {
        "experiment_id": experiment_id,
        "source_experiment_id": experiment.get("source_experiment_id"),
        "config_path": experiment.get("config_path"),
        "packet_source": cfg.get("packet", {}).get("source"),
        "ipc": cfg.get("packet", {}).get("ipc"),
        "social_epochs": cfg.get("social_head", {}).get("epochs"),
        "samples_per_class": cfg.get("social_head", {}).get("samples_per_class"),
        "steps_per_epoch": cfg.get("social_head", {}).get("steps_per_epoch"),
        "lambda_packet_ce": cfg.get("social_head", {}).get("lambda_packet_ce"),
        "lambda_packet_kd": cfg.get("social_head", {}).get("lambda_packet_kd"),
        "lambda_known_ce": cfg.get("social_head", {}).get("lambda_known_ce"),
        "lambda_retain": cfg.get("social_head", {}).get("lambda_retain"),
        **summary,
        "stage1_generalist_training_mib": communication["stage1_generalist_training_mib"],
        "stage2_missing_packet_mib": communication["stage2_missing_packet_mib"],
    }
    write_agents_csv(report_dir / "agents.csv", agents)
    write_metrics_csv(report_dir / "metrics.csv", metrics_row)
    print(f"saved_report: {report_path}")
    print(f"saved_summary: {summary_path}")
    print(f"saved_agents_csv: {report_dir / 'agents.csv'}")
    print(f"saved_metrics_csv: {report_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
