#!/usr/bin/env python3
"""Build sender-local packets from reusable backbone-specific full-class pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from fullclass_pool_packets import load_pool_catalog, materialize_pool_packets  # noqa: E402
from social_output_manager import prepare_social_output_dirs, save_social_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool-catalog", required=True)
    parser.add_argument("--only-agent", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.resume and cli.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_config(cli.config)
    args = build_dsdm_args_from_config(config, config_path=cli.config)
    catalog = load_pool_catalog(cli.pool_catalog)
    if not cli.dry_run:
        prepare_social_output_dirs(args)
        save_social_config(cli.config, args)
    results = materialize_pool_packets(
        config,
        args,
        catalog,
        only_agent=cli.only_agent,
        resume=cli.resume,
        overwrite=cli.overwrite,
        dry_run=cli.dry_run,
    )
    summary = {
        "protocol": str(catalog.get("protocol", "backbone_specific_fullclass_pool_slice_v1")),
        "config": str(cli.config),
        "pool_catalog": str(cli.pool_catalog),
        "dry_run": bool(cli.dry_run),
        "agents": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
