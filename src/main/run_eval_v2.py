"""
v2 Task placeholder: evaluate expert or socialized agents.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-stage", type=str, choices=["expert", "socialized"], required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from src.utils.config import load_yaml

    cfg = load_yaml(args.config)
    print("=== run_eval_v2 ===")
    print(f"config: {args.config}")
    print(f"checkpoint_stage: {args.checkpoint_stage}")
    print(f"packet_source: {args.packet_source or cfg.get('packet', {}).get('source')}")
    print("TODO: report expert/new/overall accuracy per agent.")


if __name__ == "__main__":
    main()
