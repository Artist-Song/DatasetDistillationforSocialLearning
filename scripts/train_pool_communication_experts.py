#!/usr/bin/env python3
"""Train local experts for a pool-reuse communication run without training DSDM guides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args, get_agent_ids  # noqa: E402
from agent_trainer import train_agent_expert_only  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from social_output_manager import prepare_social_output_dirs, save_social_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-agent", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.resume and cli.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_config(cli.config)
    base_args = build_dsdm_args_from_config(config, config_path=cli.config)
    prepare_social_output_dirs(base_args)
    save_social_config(cli.config, base_args)
    for agent_id in get_agent_ids(base_args, cli.only_agent):
        args = build_agent_args(config, cli.config, agent_id)
        path = train_agent_expert_only(
            args,
            agent_id,
            resume=cli.resume,
            overwrite=cli.overwrite,
        )
        print(f"[expert-only] agent={agent_id} model={args.model_name} path={path}", flush=True)


if __name__ == "__main__":
    main()
