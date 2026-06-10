"""
v2 Task placeholder: build raw or strict DSDM agent packets.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from src.utils.config import load_yaml

    cfg = load_yaml(args.config)
    packet_source = args.packet_source or cfg.get("packet", {}).get("source")
    print("=== run_build_packets_v2 ===")
    print(f"config: {args.config}")
    print(f"packet_source: {packet_source}")
    print("TODO: build agent-to-agent packets without soft targets.")


if __name__ == "__main__":
    main()
