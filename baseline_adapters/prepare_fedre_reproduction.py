from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split  # noqa: E402
from config_adapter import load_config  # noqa: E402


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
    parser.add_argument(
        "--project-config",
        nargs="+",
        default=None,
        help="One or more current project configs whose 5/10/20-agent class splits should be exported.",
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

    if args.project_config:
        for config_value in args.project_config:
            config_path = Path(config_value).resolve()
            project_cfg = load_config(config_path)
            dataset_name = project_dataset_name(project_cfg)
            project_root = output_root / dataset_name
            _prepare_project_split(
                project_root,
                dataset_name,
                config_path,
                project_cfg,
                train_x,
                train_y,
                test_x,
                test_y,
                force=args.force,
            )
            _link_dataset(project_root, fedre_dataset_root / dataset_name)
            print(f"[fedre-data] project={project_root} dataset={dataset_name}")
    else:
        official_root = output_root / OFFICIAL_DATASET
        social_root = output_root / SOCIAL_DATASET
        _prepare_official_pat10(official_root, train_x, train_y, test_x, test_y, force=args.force)
        _prepare_social_pat4(social_root, train_x, train_y, test_x, test_y, force=args.force)
        _link_dataset(official_root, fedre_dataset_root / OFFICIAL_DATASET)
        _link_dataset(social_root, fedre_dataset_root / SOCIAL_DATASET)

        print(f"[fedre-data] official={official_root}")
        print(f"[fedre-data] social={social_root}")


def project_dataset_name(project_cfg: dict) -> str:
    class_split = get_agent_class_split(project_cfg)
    seed = int(project_cfg.get("runtime", {}).get("seed", 0))
    canonical = json.dumps(
        {str(agent): classes for agent, classes in class_split.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    split_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return f"Cifar100ProjectA{len(class_split)}S{seed}_{split_hash}"


def _prepare_project_split(
    root: Path,
    dataset_name: str,
    config_path: Path,
    project_cfg: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    force: bool,
) -> None:
    class_split = get_agent_class_split(project_cfg)
    if len(class_split) not in {5, 10, 20}:
        raise ValueError(f"FedRE project adapter supports 5/10/20 agents, got {len(class_split)}")
    if sorted(class_split) != list(range(len(class_split))):
        raise ValueError("FedRE project agent IDs must be contiguous from zero")
    flattened = [class_id for classes in class_split.values() for class_id in classes]
    if sorted(flattened) != list(range(100)) or len(flattened) != 100:
        raise ValueError("FedRE project classes must be a disjoint global-label cover of 0-99")

    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = _validate_existing(root, expected_clients=len(class_split))
        if manifest.get("project_config_sha256") != config_sha256:
            raise ValueError(f"{root}: existing project config SHA differs")
        if manifest.get("fedre_dataset_name") != dataset_name:
            raise ValueError(f"{root}: existing FedRE dataset name differs")
        return

    normalized_train = _normalize(train_x)
    normalized_test = _normalize(test_x)
    agents: dict[str, dict] = {}
    for client_id, classes in class_split.items():
        train_indices = np.flatnonzero(np.isin(train_y, classes))
        test_indices = np.flatnonzero(np.isin(test_y, classes))
        local_train_x, local_train_y = normalized_train[train_indices], train_y[train_indices]
        local_test_x, local_test_y = normalized_test[test_indices], test_y[test_indices]
        _save_client(root, client_id, local_train_x, local_train_y, local_test_x, local_test_y)
        agents[str(client_id)] = _agent_manifest(classes, local_train_y, local_test_y)

    _write_dataset_metadata(
        root,
        dataset_name=dataset_name,
        protocol=f"project_nested_{len(class_split)}agent_class_disjoint_official_train_test",
        agents=agents,
        source_train_images=len(train_y),
        source_test_images=len(test_y),
        official_test_preserved=True,
        extra_manifest={
            "project_config": str(config_path),
            "project_config_sha256": config_sha256,
            "project_run_name": str(project_cfg["project"]["run_name"]),
            "class_assignment_seed": int(project_cfg.get("runtime", {}).get("seed", 0)),
        },
    )


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
    extra_manifest: dict | None = None,
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
    if extra_manifest:
        manifest.update(extra_manifest)
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


def _validate_existing(root: Path, *, expected_clients: int) -> dict:
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
    return manifest


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
