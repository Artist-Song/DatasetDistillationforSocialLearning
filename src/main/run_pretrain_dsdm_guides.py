"""
v2 Task placeholder: pretrain strict DSDM guide checkpoints.
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
    dsdm_cfg = cfg.get("dsdm", {})
    print("=== run_pretrain_dsdm_guides ===")
    print(f"config: {args.config}")
    print(f"agent_ids: {args.agent_ids}")
    print(f"guide_model_number: {dsdm_cfg.get('guide_model_number')}")
    print("TODO: train multiple guide checkpoints per sender agent.")


if __name__ == "__main__":
    main()
