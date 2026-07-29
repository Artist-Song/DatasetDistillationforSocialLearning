#!/usr/bin/env python3
"""Copy immutable cosine experts into a derived DKP receiver run with SHA provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split, get_agent_model_split  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_copyfile, atomic_write_json  # noqa: E402
from social_output_manager import prepare_social_output_dirs  # noqa: E402


ARTIFACT_NAMES = ("expert_model.pt", "expert_selection.json", "expert_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _require_same_protocol(source: dict, target: dict) -> None:
    if get_agent_class_split(source) != get_agent_class_split(target):
        raise ValueError("expert reuse requires identical agent class splits")
    if get_agent_model_split(source) != get_agent_model_split(target):
        raise ValueError("expert reuse requires identical agent model assignments")
    source_models = source["model_pool"]["models"]
    target_models = target["model_pool"]["models"]
    for model_name in get_agent_model_split(source).values():
        for key in ("family", "depth", "width", "norm_type", "classifier"):
            if source_models[model_name].get(key) != target_models[model_name].get(key):
                raise ValueError(f"{model_name}: expert reuse model field differs: {key}")


def main() -> None:
    cli = _parse_args()
    source_path = Path(cli.source_config)
    target_path = Path(cli.target_config)
    source = load_config(source_path)
    target = load_config(target_path)
    _require_same_protocol(source, target)

    source_args = build_dsdm_args_from_config(source, config_path=source_path)
    target_args = build_dsdm_args_from_config(target, config_path=target_path)
    expected_source_run = target.get("expert_reuse", {}).get("source_run")
    if expected_source_run != source_args.run_name:
        raise ValueError(
            f"target expert_reuse.source_run={expected_source_run!r} does not match {source_args.run_name!r}"
        )
    target_run = prepare_social_output_dirs(target_args)
    target_config_snapshot = target_run / "config" / "main.yaml"
    if target_config_snapshot.exists():
        if _sha256(target_config_snapshot) != _sha256(target_path):
            raise FileExistsError(f"target config snapshot differs: {target_config_snapshot}")
    else:
        atomic_copyfile(target_path, target_config_snapshot)

    records = []
    class_split = get_agent_class_split(source)
    model_split = get_agent_model_split(source)
    for agent_id in sorted(class_split):
        source_dir = Path(source_args.output_root) / source_args.run_name / "agents" / f"agent_{agent_id}" / "checkpoints"
        target_dir = Path(target_args.output_root) / target_args.run_name / "agents" / f"agent_{agent_id}" / "checkpoints"
        copied = {}
        for name in ARTIFACT_NAMES:
            src = source_dir / name
            dst = target_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"missing source expert artifact: {src}")
            source_sha = _sha256(src)
            if dst.exists():
                if not cli.resume:
                    raise FileExistsError(f"target expert artifact already exists: {dst}")
                if _sha256(dst) != source_sha:
                    raise ValueError(f"resume target SHA differs from source: {dst}")
            else:
                atomic_copyfile(src, dst)
            destination_sha = _sha256(dst)
            if destination_sha != source_sha:
                raise ValueError(f"expert artifact copy SHA mismatch: {dst}")
            copied[name] = {"source": str(src), "destination": str(dst), "sha256": source_sha}
        records.append(
            {
                "agent_id": int(agent_id),
                "model": model_split[agent_id],
                "class_ids": class_split[agent_id],
                "artifacts": copied,
            }
        )

    manifest = {
        "protocol": str(target_args.communication_protocol),
        "agent_count": len(class_split),
        "classes_per_agent": len(next(iter(class_split.values()))),
        "source_run": source_args.run_name,
        "target_run": target_args.run_name,
        "source_config": str(source_path),
        "source_config_sha256": _sha256(source_path),
        "target_config": str(target_path),
        "target_config_sha256": _sha256(target_path),
        "require_checkpoint_sha_match": True,
        "agents": records,
    }
    manifest_path = target_run / "provenance" / "expert_reuse_manifest.json"
    if manifest_path.exists() and not cli.resume:
        raise FileExistsError(f"expert reuse manifest already exists: {manifest_path}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"existing expert reuse manifest differs: {manifest_path}")
    else:
        atomic_write_json(manifest, manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
