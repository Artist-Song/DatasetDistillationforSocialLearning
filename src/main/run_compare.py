"""
Compare local and social evaluation reports.

This script reads the JSON reports produced by run_eval.py and reports per-agent
and average changes from local_pretrain to social_train.
"""

import argparse
import json
from pathlib import Path

from src.utils.config import load_yaml
from src.utils.run_name import build_base_run_name, build_social_run_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='要比较的 agent 后缀，例如 "all"、"0"、"0,2,4" 或 "0-3"，需与 run_eval 一致',
    )
    return parser.parse_args()


def agent_suffix(agent_ids: str) -> str:
    return "all" if agent_ids == "all" else agent_ids.replace(",", "_").replace("-", "to")


def load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"report 不存在: {path}")
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
    report_dir = Path(cfg["output"]["root"]) / "reports" / "eval"

    local_path = report_dir / f"local_pretrain_{base_run_name}_{suffix}.json"
    social_path = report_dir / f"social_train_{social_run_name}_{suffix}.json"

    local_report = load_report(local_path)
    social_report = load_report(social_path)
    local_agents = index_agents(local_report)
    social_agents = index_agents(social_report)

    comparison = {
        "local_report": str(local_path),
        "social_report": str(social_path),
        "dataset": cfg["dataset"]["name"],
        "split_mode": cfg["split"]["mode"],
        "model": cfg["model"]["name"],
        "social": cfg.get("social", {}),
        "agents": [],
    }

    print("=== run_compare local_pretrain -> social_train ===")
    print(f"local_report: {local_path}")
    print(f"social_report: {social_path}")

    for agent_id in sorted(social_agents):
        local_item = local_agents[agent_id]
        social_item = social_agents[agent_id]
        expert_delta = social_item["expert_accuracy"] - local_item["expert_accuracy"]
        general_delta = social_item["general_accuracy"] - local_item["general_accuracy"]
        comparison["agents"].append(
            {
                "agent_id": agent_id,
                "class_ids": social_item["class_ids"],
                "local_expert_accuracy": local_item["expert_accuracy"],
                "social_expert_accuracy": social_item["expert_accuracy"],
                "delta_expert_accuracy": expert_delta,
                "local_general_accuracy": local_item["general_accuracy"],
                "social_general_accuracy": social_item["general_accuracy"],
                "delta_general_accuracy": general_delta,
            }
        )
        print(
            f"agent_{agent_id} "
            f"expert {local_item['expert_accuracy']:.4f}->{social_item['expert_accuracy']:.4f} "
            f"delta={expert_delta:+.4f} | "
            f"general {local_item['general_accuracy']:.4f}->{social_item['general_accuracy']:.4f} "
            f"delta={general_delta:+.4f}"
        )

    local_summary = local_report["summary"]
    social_summary = social_report["summary"]
    summary = {
        "local_average_expert_accuracy": local_summary["average_expert_accuracy"],
        "social_average_expert_accuracy": social_summary["average_expert_accuracy"],
        "delta_average_expert_accuracy": (
            social_summary["average_expert_accuracy"] - local_summary["average_expert_accuracy"]
        ),
        "local_average_general_accuracy": local_summary["average_general_accuracy"],
        "social_average_general_accuracy": social_summary["average_general_accuracy"],
        "delta_average_general_accuracy": (
            social_summary["average_general_accuracy"] - local_summary["average_general_accuracy"]
        ),
    }
    comparison["summary"] = summary

    print("=== summary ===")
    print(
        "average_expert_accuracy "
        f"{summary['local_average_expert_accuracy']:.4f}->{summary['social_average_expert_accuracy']:.4f} "
        f"delta={summary['delta_average_expert_accuracy']:+.4f}"
    )
    print(
        "average_general_accuracy "
        f"{summary['local_average_general_accuracy']:.4f}->{summary['social_average_general_accuracy']:.4f} "
        f"delta={summary['delta_average_general_accuracy']:+.4f}"
    )

    compare_dir = Path(cfg["output"]["root"]) / "reports" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    compare_path = compare_dir / f"compare_{social_run_name}_{suffix}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"saved_compare: {compare_path}")


if __name__ == "__main__":
    main()