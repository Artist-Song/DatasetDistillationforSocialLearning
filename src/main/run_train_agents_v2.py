"""
v2 Task placeholder: train expert agents on their own classes.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    from src.utils.config import load_yaml

    cfg = load_yaml(args.config)
    print("=== run_train_agents_v2 ===")
    print(f"config: {args.config}")
    print(f"agent_ids: {args.agent_ids}")
    print(f"experiment: {cfg.get('experiment', {}).get('name')}")
    print("TODO: train each expert agent on its direct-split expert classes.")


if __name__ == "__main__":
    main()
