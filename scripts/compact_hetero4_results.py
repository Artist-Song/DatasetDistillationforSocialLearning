#!/usr/bin/env python3
"""Compact hetero4 output runs while retaining result/provenance files.

The default mode is review-only.  ``--apply`` removes model/checkpoint/packet
payloads and notebook caches from hetero4 output directories.  Metrics,
configs, manifests, logs, and image previews are intentionally retained in
their original paths so registry sources remain valid.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DEFAULT_MANIFEST = ROOT / "docs" / "operations" / "storage" / (
    f"hetero4_compaction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.tsv"
)
RUN_GLOB = "cifar100_4agent_25cls_hetero4arch*"
TUNING_PATTERN = re.compile(
    r"guardian|tune|tuning|ablation|smoke|recipe|midpacket|refine|joint",
    re.IGNORECASE,
)
PAYLOAD_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".tar",
    ".gz",
}


def files_to_remove(run: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        if ".ipynb_checkpoints" in path.parts:
            candidates.append((path, "notebook cache"))
        elif path.suffix.lower() in PAYLOAD_SUFFIXES:
            candidates.append((path, "model/checkpoint/packet payload"))
    return candidates


def physical_bytes(paths: list[Path]) -> tuple[int, int]:
    """Return logical bytes and unique allocated blocks for candidate files."""
    logical = 0
    inodes: dict[tuple[int, int], tuple[int, int]] = {}
    for path in paths:
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        logical += info.st_size
        key = (info.st_dev, info.st_ino)
        inodes[key] = (info.st_nlink, info.st_blocks * 512)
    return logical, sum(blocks for _, blocks in inodes.values())


def write_manifest(path: Path, rows: list[dict[str, object]], apply: bool) -> None:
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_name", "path", "bytes", "reason", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mode = "applied" if apply else "dry-run"
    print(f"[hetero4 cleanup] {mode} manifest: {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="delete listed payloads")
    parser.add_argument(
        "--scope",
        choices=("hetero4", "tuning", "all"),
        default="hetero4",
        help="select hetero4 runs, non-hetero4 tuning runs, or both",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    hetero4_runs = {
        path for path in OUTPUTS.glob(RUN_GLOB) if path.is_dir()
    }
    tuning_runs = {
        path
        for path in OUTPUTS.iterdir()
        if path.is_dir()
        and path not in hetero4_runs
        and TUNING_PATTERN.search(path.name)
    }
    if args.scope == "hetero4":
        runs = sorted(hetero4_runs)
    elif args.scope == "tuning":
        runs = sorted(tuning_runs)
    else:
        runs = sorted(hetero4_runs | tuning_runs)
    rows: list[dict[str, object]] = []
    files: list[Path] = []
    by_run: defaultdict[str, int] = defaultdict(int)
    for run in runs:
        for path, reason in files_to_remove(run):
            files.append(path)
            by_run[run.name] += path.stat().st_size
            rows.append(
                {
                    "run_name": run.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "reason": reason,
                    "status": "deleted" if args.apply else "planned",
                }
            )

    logical, allocated = physical_bytes(files)
    print(f"[hetero4 cleanup] runs: {len(runs)}")
    print(f"[hetero4 cleanup] files: {len(files)}")
    print(f"[hetero4 cleanup] logical bytes: {logical:,}")
    print(f"[hetero4 cleanup] unique allocated bytes: {allocated:,}")
    if by_run:
        print("[hetero4 cleanup] largest runs:")
        for name, size in sorted(by_run.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"  {size:>14,}  {name}")

    if args.apply:
        for path in files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for run in runs:
            for path in sorted(run.rglob("*"), reverse=True):
                if path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass

    write_manifest(args.manifest, rows, args.apply)


if __name__ == "__main__":
    main()
