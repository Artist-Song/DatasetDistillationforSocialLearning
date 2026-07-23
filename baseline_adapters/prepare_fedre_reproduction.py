from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100


OFFICIAL_DATASET = "Cifar100FedREPat10"
SOCIAL_DATASET = "Cifar100SocialPAT4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare isolated CIFAR-100 datasets for FedRE reproduction.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-root",
        default="external_baselines/outputs/fedre_reproduction/datasets",
    )
    parser.add_argument(
        "--fedre-dataset-root",
        default="external_baselines/repos/FedRE/HtFLlib/dataset",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    fedre_dataset_root = Path(args.fedre_dataset_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fedre_dataset_root.mkdir(parents=True, exist_ok=True)

    train = CIFAR100(args.data_dir, train=True, download=False)
    test = CIFAR100(args.data_dir, train=False, download=False)
    train_x = np.asarray(train.data)
    train_y = np.asarray(train.targets, dtype=np.int64)
    test_x = np.asarray(test.data)
    test_y = np.asarray(test.targets, dtype=np.int64)

    official_root = output_root / OFFICIAL_DATASET
    social_root = output_root / SOCIAL_DATASET
    _prepare_official_pat10(official_root, train_x, train_y, test_x, test_y, force=args.force)
    _prepare_social_pat4(social_root, train_x, train_y, test_x, test_y, force=args.force)
    _link_dataset(official_root, fedre_dataset_root / OFFICIAL_DATASET)
    _link_dataset(social_root, fedre_dataset_root / SOCIAL_DATASET)

    print(f"[fedre-data] official={official_root}")
    print(f"[fedre-data] social={social_root}")


def _prepare_official_pat10(
    root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    force: bool,
) -> None:
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        _validate_existing(root, expected_clients=10)
        return

    all_x = np.concatenate([train_x, test_x], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0)
    normalized = _normalize(all_x)
    np.random.seed(0)
    agents: dict[str, dict] = {}

    for client_id in range(10):
        classes = list(range(client_id * 10, (client_id + 1) * 10))
        indices = np.concatenate([np.flatnonzero(all_y == cls) for cls in classes])
        local_x = normalized[indices]
        local_y = all_y[indices]
        x_train, x_test, y_train, y_test = train_test_split(
            local_x,
            local_y,
            train_size=0.75,
            shuffle=True,
        )
        _save_client(root, client_id, x_train, y_train, x_test, y_test)
        agents[str(client_id)] = _agent_manifest(classes, y_train, y_test)

    _write_dataset_metadata(
        root,
        dataset_name=OFFICIAL_DATASET,
        protocol="fedre_official_pat10_merge_then_local_75_25",
        agents=agents,
        source_train_images=len(train_y),
        source_test_images=len(test_y),
        official_test_preserved=False,
    )


def _prepare_social_pat4(
    root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    force: bool,
) -> None:
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        _validate_existing(root, expected_clients=4)
        return

    normalized_train = _normalize(train_x)
    normalized_test = _normalize(test_x)
    agents: dict[str, dict] = {}
    for client_id in range(4):
        classes = list(range(client_id * 25, (client_id + 1) * 25))
        train_indices = np.flatnonzero(np.isin(train_y, classes))
        test_indices = np.flatnonzero(np.isin(test_y, classes))
        x_train, y_train = normalized_train[train_indices], train_y[train_indices]
        x_test, y_test = normalized_test[test_indices], test_y[test_indices]
        _save_client(root, client_id, x_train, y_train, x_test, y_test)
        agents[str(client_id)] = _agent_manifest(classes, y_train, y_test)

    _write_dataset_metadata(
        root,
        dataset_name=SOCIAL_DATASET,
        protocol="social_pat4_official_train_test",
        agents=agents,
        source_train_images=len(train_y),
        source_test_images=len(test_y),
        official_test_preserved=True,
    )


def _normalize(images: np.ndarray) -> np.ndarray:
    chw = np.transpose(images.astype(np.float32) / 255.0, (0, 3, 1, 2))
    return (chw - 0.5) / 0.5


def _save_client(
    root: Path,
    client_id: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> None:
    _atomic_npz(root / "train" / f"{client_id}.npz", {"x": train_x, "y": train_y})
    _atomic_npz(root / "test" / f"{client_id}.npz", {"x": test_x, "y": test_y})


def _agent_manifest(classes: list[int], train_y: np.ndarray, test_y: np.ndarray) -> dict:
    return {
        "classes": classes,
        "train_count": int(len(train_y)),
        "test_count": int(len(test_y)),
        "train_per_class": {str(cls): int(np.sum(train_y == cls)) for cls in classes},
        "test_per_class": {str(cls): int(np.sum(test_y == cls)) for cls in classes},
    }


def _write_dataset_metadata(
    root: Path,
    *,
    dataset_name: str,
    protocol: str,
    agents: dict[str, dict],
    source_train_images: int,
    source_test_images: int,
    official_test_preserved: bool,
) -> None:
    manifest = {
        "dataset": "cifar100",
        "fedre_dataset_name": dataset_name,
        "protocol": protocol,
        "normalization": "(ToTensor - 0.5) / 0.5",
        "num_clients": len(agents),
        "num_classes": 100,
        "global_labels": "0-99",
        "source_train_images": source_train_images,
        "source_test_images": source_test_images,
        "official_test_preserved": official_test_preserved,
        "agents": agents,
    }
    config = {
        "num_clients": len(agents),
        "num_classes": 100,
        "non_iid": True,
        "balance": True,
        "partition": protocol,
        "alpha": 0.1,
        "batch_size": 10,
        "Size of samples for labels in clients": {
            client_id: [[cls, count] for cls, count in info["train_per_class"].items()]
            for client_id, info in agents.items()
        },
    }
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(root / "config.json", config)
    _validate_existing(root, expected_clients=len(agents))


def _validate_existing(root: Path, *, expected_clients: int) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["num_clients"]) != expected_clients:
        raise ValueError(f"{root}: expected {expected_clients} clients")
    all_classes: list[int] = []
    for client_id in range(expected_clients):
        for split in ("train", "test"):
            path = root / split / f"{client_id}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
        info = manifest["agents"][str(client_id)]
        all_classes.extend(int(cls) for cls in info["classes"])
    if sorted(all_classes) != list(range(100)):
        raise ValueError(f"{root}: client class sets must be a disjoint cover of 0-99")


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
        np.savez_compressed(handle, data=payload)
    os.replace(temp_path, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _link_dataset(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"Refusing to replace real dataset directory: {target}")
    target.symlink_to(source, target_is_directory=True)


if __name__ == "__main__":
    main()
