"""
Run a small social-training sweep over packet KD modes and retain weights.

For each setting, this script writes a generated config and runs:
1. run_social_train
2. run_eval --checkpoint-stage social_train
3. run_compare

Local pretrain eval should be generated once before running this script.
"""

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils.config import load_yaml


def parse_csv_values(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="基础实验配置文件路径")
    parser.add_argument(
        "--packet-kd-modes",
        type=str,
        default="full,sender_subset,none",
        help="逗号分隔的 packet KD 模式",
    )
    parser.add_argument(
        "--lambda-retains",
        type=str,
        default="0.0,0.1,0.3,0.5,1.0",
        help="逗号分隔的 lambda_retain 值",
    )
    parser.add_argument(
        "--lambda-packets",
        type=str,
        default="1.0",
        help="逗号分隔的 lambda_packet 值",
    )
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='要运行的 agent，例如 "all"、"0"、"0,2,4" 或 "0-3"',
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过 social train，仅运行 eval 和 compare",
    )
    return parser.parse_args()


def write_generated_config(cfg, output_dir: Path, mode: str, lambda_retain: float, lambda_packet: float) -> Path:
    generated_cfg = copy.deepcopy(cfg)
    generated_cfg.setdefault("social", {})
    generated_cfg["social"]["packet_kd_mode"] = mode
    generated_cfg["social"]["lambda_retain"] = lambda_retain
    generated_cfg["social"]["lambda_packet"] = lambda_packet

    output_dir.mkdir(parents=True, exist_ok=True)
    retain_tag = str(lambda_retain).replace(".", "p")
    packet_tag = str(lambda_packet).replace(".", "p")
    config_path = output_dir / f"social_kd-{mode}_retain-{retain_tag}_packet-{packet_tag}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(generated_cfg, f, allow_unicode=True, sort_keys=False)
    return config_path


def run_command(command):
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    modes = parse_csv_values(args.packet_kd_modes, str)
    lambda_retains = parse_csv_values(args.lambda_retains, float)
    lambda_packets = parse_csv_values(args.lambda_packets, float)
    generated_dir = Path(cfg["output"]["root"]) / "generated_configs"

    for mode in modes:
        for lambda_retain in lambda_retains:
            for lambda_packet in lambda_packets:
                print("\n=== sweep setting ===")
                print(f"packet_kd_mode: {mode}")
                print(f"lambda_retain: {lambda_retain}")
                print(f"lambda_packet: {lambda_packet}")

                config_path = write_generated_config(
                    cfg=cfg,
                    output_dir=generated_dir,
                    mode=mode,
                    lambda_retain=lambda_retain,
                    lambda_packet=lambda_packet,
                )

                if not args.skip_train:
                    run_command(
                        [
                            sys.executable,
                            "-m",
                            "src.main.run_social_train",
                            "--config",
                            str(config_path),
                            "--agent-ids",
                            args.agent_ids,
                        ]
                    )
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.main.run_eval",
                        "--config",
                        str(config_path),
                        "--checkpoint-stage",
                        "social_train",
                        "--agent-ids",
                        args.agent_ids,
                    ]
                )
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.main.run_compare",
                        "--config",
                        str(config_path),
                        "--agent-ids",
                        args.agent_ids,
                    ]
                )


if __name__ == "__main__":
    main()