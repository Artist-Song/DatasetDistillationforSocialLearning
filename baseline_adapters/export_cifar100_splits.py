from __future__ import annotations

import argparse
from pathlib import Path

from baseline_adapters.common import (
    build_export_manifest,
    export_fedre_npz,
    export_masc_imagefolder,
    export_split_metadata,
    load_social_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-100 socialized split for external baselines.")
    parser.add_argument("--config", required=True, help="项目主实验 yaml 配置")
    parser.add_argument("--output-root", default="external_baselines/outputs/splits", help="导出根目录")
    parser.add_argument("--ipc", type=int, default=10, help="MASC/DeSA 图像通信预算 IPC")
    parser.add_argument(
        "--target",
        choices=["all", "masc", "fedre", "metadata"],
        default="all",
        help="导出目标；metadata 供 FedProto/DeSA runner 读取",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_social_split(args.config)
    run_name = Path(args.config).stem
    output_root = Path(args.output_root) / run_name / f"ipc{args.ipc}"

    print(f"[split] dataset={split.dataset} agents={split.num_agents} seed={split.seed}")
    for agent_id, classes in split.class_split.items():
        print(f"[agent {agent_id}] model={split.model_split.get(agent_id, 'unknown')} classes={classes[0]}..{classes[-1]}")
    print(f"[output] {output_root}")

    if args.dry_run:
        print("[dry-run] no files written")
        return

    exports: dict[str, str] = {}
    if args.target in {"all", "fedre"}:
        fedre_path = export_fedre_npz(split, output_root / "fedre")
        exports["fedre_npz"] = str(fedre_path)
        print(f"[fedre] wrote {fedre_path}")

    if args.target in {"all", "masc"}:
        masc_path = export_masc_imagefolder(split, output_root / "masc", ipc=args.ipc)
        exports["masc_imagefolder"] = str(masc_path)
        print(f"[masc] wrote {masc_path}")

    if args.target in {"all", "metadata"}:
        metadata_path = export_split_metadata(split, output_root / "metadata", ipc=args.ipc)
        exports["split_metadata"] = str(metadata_path)
        print(f"[metadata] wrote {metadata_path}")

    manifest_path = build_export_manifest(split, output_root, exports)
    print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
