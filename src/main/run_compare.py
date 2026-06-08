"""
Compare local and post-communication evaluation reports.

By default this keeps the original behavior and compares local_pretrain to
social_train. Use --target-stage packet_only_train to compare the new
packet-only fine-tuning path.
"""

import argparse
import json
from pathlib import Path

from src.utils.config import load_yaml
from src.utils.run_name import build_base_run_name, build_packet_only_run_name, build_social_run_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config path")
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='agent suffix used by run_eval, e.g. "all", "0", "0,2,4", or "0-3"',
    )
    parser.add_argument(
        "--target-stage",
        type=str,
        default="social_train",
        choices=["social_train", "packet_only_train"],
        help="target stage to compare against local_pretrain",
    )
    return parser.parse_args()


def agent_suffix(agent_ids: str) -> str:
    return "all" if agent_ids == "all" else agent_ids.replace(",", "_").replace("-", "to")


def load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_agents(report):
    return {item["agent_id"]: item for item in report["agents"]}


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    suffix = agent_suffix(args.agent_ids)
    base_run_name = build_base_run_name(cfg)
    social_run_name = build_social_run_name(cfg)
    target_run_name = social_run_name
    if args.target_stage == "packet_only_train":
        target_run_name = build_packet_only_run_name(cfg)
    report_dir = Path(cfg["output"]["root"]) / "reports" / "eval"

    local_path = report_dir / f"local_pretrain_{base_run_name}_{suffix}.json"
    target_path = report_dir / f"{args.target_stage}_{target_run_name}_{suffix}.json"

    local_report = load_report(local_path)
    target_report = load_report(target_path)
    local_agents = index_agents(local_report)
    target_agents = index_agents(target_report)

    comparison = {
        "local_report": str(local_path),
        "target_report": str(target_path),
        "target_stage": args.target_stage,
        "dataset": cfg["dataset"]["name"],
        "split_mode": cfg["split"]["mode"],
        "model": cfg["model"]["name"],
        "packet": cfg.get("packet", {}),
        "social": cfg.get("social", {}),
        "packet_only": cfg.get("packet_only", {}),
        "agents": [],
    }

    print(f"=== run_compare local_pretrain -> {args.target_stage} ===")
    print(f"local_report: {local_path}")
    print(f"target_report: {target_path}")

    for agent_id in sorted(target_agents):
        local_item = local_agents[agent_id]
        target_item = target_agents[agent_id]
        expert_delta = target_item["expert_accuracy"] - local_item["expert_accuracy"]
        general_delta = target_item["general_accuracy"] - local_item["general_accuracy"]
        comparison["agents"].append(
            {
                "agent_id": agent_id,
                "class_ids": target_item["class_ids"],
                "local_expert_accuracy": local_item["expert_accuracy"],
                "social_expert_accuracy": target_item["expert_accuracy"],
                "target_expert_accuracy": target_item["expert_accuracy"],
                "delta_expert_accuracy": expert_delta,
                "local_general_accuracy": local_item["general_accuracy"],
                "social_general_accuracy": target_item["general_accuracy"],
                "target_general_accuracy": target_item["general_accuracy"],
                "delta_general_accuracy": general_delta,
            }
        )
        print(
            f"agent_{agent_id} "
            f"expert {local_item['expert_accuracy']:.4f}->{target_item['expert_accuracy']:.4f} "
            f"delta={expert_delta:+.4f} | "
            f"general {local_item['general_accuracy']:.4f}->{target_item['general_accuracy']:.4f} "
            f"delta={general_delta:+.4f}"
        )

    local_summary = local_report["summary"]
    target_summary = target_report["summary"]
    summary = {
        "local_average_expert_accuracy": local_summary["average_expert_accuracy"],
        "social_average_expert_accuracy": target_summary["average_expert_accuracy"],
        "target_average_expert_accuracy": target_summary["average_expert_accuracy"],
        "delta_average_expert_accuracy": (
            target_summary["average_expert_accuracy"] - local_summary["average_expert_accuracy"]
        ),
        "local_average_general_accuracy": local_summary["average_general_accuracy"],
        "social_average_general_accuracy": target_summary["average_general_accuracy"],
        "target_average_general_accuracy": target_summary["average_general_accuracy"],
        "delta_average_general_accuracy": (
            target_summary["average_general_accuracy"] - local_summary["average_general_accuracy"]
        ),
    }
    comparison["summary"] = summary

    print("=== summary ===")
    print(
        "average_expert_accuracy "
        f"{summary['local_average_expert_accuracy']:.4f}->{summary['target_average_expert_accuracy']:.4f} "
        f"delta={summary['delta_average_expert_accuracy']:+.4f}"
    )
    print(
        "average_general_accuracy "
        f"{summary['local_average_general_accuracy']:.4f}->{summary['target_average_general_accuracy']:.4f} "
        f"delta={summary['delta_average_general_accuracy']:+.4f}"
    )

    compare_dir = Path(cfg["output"]["root"]) / "reports" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    compare_prefix = "compare" if args.target_stage == "social_train" else f"compare_{args.target_stage}"
    compare_path = compare_dir / f"{compare_prefix}_{target_run_name}_{suffix}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"saved_compare: {compare_path}")


if __name__ == "__main__":
    main()
