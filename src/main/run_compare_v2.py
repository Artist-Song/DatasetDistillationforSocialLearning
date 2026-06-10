"""
v2 Task placeholder: compare local, raw IPC, and strict DSDM IPC results.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    from src.utils.config import load_yaml

    cfg = load_yaml(args.config)
    print("=== run_compare_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg.get('experiment', {}).get('name')}")
    print("TODO: compare local_only, raw_ipc, and strict_dsdm_ipc.")


if __name__ == "__main__":
    main()
