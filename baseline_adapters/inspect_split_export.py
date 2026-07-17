from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect exported baseline split.")
    parser.add_argument("--export-root", required=True, help="external_baselines/outputs/splits/.../ipcX 目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.export_root)
    manifest_path = root / "manifest.json"
    metadata_path = root / "metadata" / "split_metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"[manifest] dataset={manifest['dataset']} agents={manifest['num_agents']} seed={manifest['seed']}")
    print(f"[exports] {manifest['exports']}")
    for agent_id, info in metadata["agents"].items():
        print(
            f"[agent {agent_id}] classes={info['classes'][0]}..{info['classes'][-1]} "
            f"train={info['train_count']} test={info['test_count']} packet={len(info['packet_indices_ipc'])}"
        )

    fedre_train = root / "fedre" / "Cifar100" / "train" / "0.npz"
    if fedre_train.exists():
        data = np.load(fedre_train, allow_pickle=True)["data"].tolist()
        print(f"[fedre client0] x={data['x'].shape} {data['x'].dtype} y={data['y'].shape}")

    masc_root = root / "masc"
    if masc_root.exists():
        png_count = sum(1 for _ in masc_root.rglob("*.png"))
        print(f"[masc] png={png_count}")


if __name__ == "__main__":
    main()
