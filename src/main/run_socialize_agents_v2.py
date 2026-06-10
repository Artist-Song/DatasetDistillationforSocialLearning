"""
v2 Task placeholder: socialize agents with packet images.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    from src.utils.config import load_yaml

    cfg = load_yaml(args.config)
    packet_source = args.packet_source or cfg.get("packet", {}).get("source")
    print("=== run_socialize_agents_v2 ===")
    print(f"config: {args.config}")
    print(f"agent_ids: {args.agent_ids}")
    print(f"packet_source: {packet_source}")
    print("TODO: run head warm-up and last-block adaptation.")


if __name__ == "__main__":
    main()
