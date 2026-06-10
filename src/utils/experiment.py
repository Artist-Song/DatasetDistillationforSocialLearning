"""
Experiment id, reuse, and output path helpers for the generalist pipeline.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _stable_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["seed", "dataset", "split", "model", "method", "train", "generalist", "specialist", "packet", "social_head"]
    return {key: deepcopy(cfg.get(key)) for key in keys if key in cfg}


def _short_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def get_experiment_id(cfg: Dict[str, Any], config_path: Optional[str] = None) -> str:
    output_cfg = cfg.get("output", {})
    explicit = cfg.get("experiment_id") or cfg.get("experiment", {}).get("id") or output_cfg.get("experiment_id")
    if explicit:
        return str(explicit)

    config_stem = Path(config_path).stem if config_path else None
    base = cfg.get("method", {}).get("name") or config_stem or "generalist_packet"
    return f"{base}_seed{cfg.get('seed', 'na')}_{_short_hash(_stable_payload(cfg))}"


def get_experiment_root(cfg: Dict[str, Any], config_path: Optional[str] = None) -> Path:
    root = Path(cfg.get("output", {}).get("root", "./outputs"))
    return root / get_experiment_id(cfg, config_path=config_path)


def get_reuse_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    reuse = cfg.get("reuse") or {}
    return {
        "source_experiment_id": reuse.get("source_experiment_id"),
        "generalist": bool(reuse.get("generalist", False)),
        "specialists": bool(reuse.get("specialists", False)),
        "packets": bool(reuse.get("packets", False)),
    }


def get_source_experiment_id(cfg: Dict[str, Any]) -> Optional[str]:
    return get_reuse_cfg(cfg).get("source_experiment_id")


def get_source_experiment_root(cfg: Dict[str, Any]) -> Optional[Path]:
    source_experiment_id = get_source_experiment_id(cfg)
    if not source_experiment_id:
        return None
    root = Path(cfg.get("output", {}).get("root", "./outputs"))
    return root / source_experiment_id


def validate_reuse(cfg: Dict[str, Any], config_path: Optional[str] = None) -> None:
    reuse = get_reuse_cfg(cfg)
    source_experiment_id = reuse["source_experiment_id"]
    if not source_experiment_id:
        if reuse["generalist"] or reuse["specialists"] or reuse["packets"]:
            raise RuntimeError("reuse flags require reuse.source_experiment_id")
        return
    current_experiment_id = get_experiment_id(cfg, config_path=config_path)
    if current_experiment_id == source_experiment_id:
        raise RuntimeError(
            "reuse.source_experiment_id must differ from current experiment_id: "
            f"{current_experiment_id}"
        )


def get_stage_read_root(cfg: Dict[str, Any], stage: str, config_path: Optional[str] = None) -> Path:
    reuse = get_reuse_cfg(cfg)
    use_source = bool(reuse.get(stage, False))
    if use_source:
        source_root = get_source_experiment_root(cfg)
        if source_root is None:
            raise RuntimeError(f"reuse.{stage}=true requires reuse.source_experiment_id")
        return source_root
    return get_experiment_root(cfg, config_path=config_path)


def get_stage_expected_experiment_id(cfg: Dict[str, Any], stage: str, config_path: Optional[str] = None) -> str:
    reuse = get_reuse_cfg(cfg)
    if reuse.get(stage, False):
        source_experiment_id = reuse.get("source_experiment_id")
        if not source_experiment_id:
            raise RuntimeError(f"reuse.{stage}=true requires reuse.source_experiment_id")
        return str(source_experiment_id)
    return get_experiment_id(cfg, config_path=config_path)


def require_experiment_id(actual_id: Optional[str], expected_id: str, path: Path, cfg: Dict[str, Any], config_path: Optional[str] = None) -> None:
    if actual_id != expected_id:
        raise RuntimeError(
            f"experiment_id mismatch for {path}: expected_id={expected_id}, actual_id={actual_id}, "
            f"current_experiment_id={get_experiment_id(cfg, config_path=config_path)}, "
            f"source_experiment_id={get_source_experiment_id(cfg)}"
        )


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_experiment_files(
    cfg: Dict[str, Any],
    config_path: Optional[str],
    stage_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    experiment_root = get_experiment_root(cfg, config_path=config_path)
    experiment_root.mkdir(parents=True, exist_ok=True)
    config_out = experiment_root / "config.yaml"
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    manifest = {
        "experiment_id": get_experiment_id(cfg, config_path=config_path),
        "source_experiment_id": get_source_experiment_id(cfg),
        "reuse": get_reuse_cfg(cfg),
        "config_path": str(config_path) if config_path else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": cfg.get("dataset"),
        "split": cfg.get("split"),
        "model": cfg.get("model"),
        "method": cfg.get("method"),
        "packet": cfg.get("packet"),
        "social_head": cfg.get("social_head", cfg.get("social")),
        "stage_paths": stage_paths or {},
        "write_root": str(experiment_root),
    }
    manifest_path = experiment_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def get_experiment_metadata(cfg: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "experiment_id": get_experiment_id(cfg, config_path=config_path),
        "experiment_root": str(get_experiment_root(cfg, config_path=config_path)),
        "source_experiment_id": get_source_experiment_id(cfg),
        "source_experiment_root": str(get_source_experiment_root(cfg)) if get_source_experiment_root(cfg) else None,
        "reuse": get_reuse_cfg(cfg),
        "config_path": str(config_path) if config_path else None,
        "seed": cfg.get("seed"),
        "dataset": cfg.get("dataset"),
        "split": cfg.get("split"),
        "model": cfg.get("model"),
        "method": cfg.get("method"),
        "train": cfg.get("train"),
        "generalist": cfg.get("generalist"),
        "specialist": cfg.get("specialist"),
        "packet": cfg.get("packet"),
        "social_head": cfg.get("social_head", cfg.get("social")),
        "output": cfg.get("output"),
    }


def assert_report_experiment(report: Dict[str, Any], expected_experiment_id: str, report_path: Path) -> None:
    report_experiment_id = report.get("experiment_id") or report.get("experiment", {}).get("experiment_id")
    if report_experiment_id != expected_experiment_id:
        raise RuntimeError(
            f"experiment_id mismatch for {report_path}: "
            f"expected {expected_experiment_id}, got {report_experiment_id}"
        )


def require_packet_dir(packet_dir: Path, num_classes: int) -> None:
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet_dir not found: {packet_dir}")
    missing = [str(packet_dir / f"class_{class_id}_packet.pt") for class_id in range(num_classes) if not (packet_dir / f"class_{class_id}_packet.pt").exists()]
    if missing:
        raise FileNotFoundError("packet_dir is incomplete. Missing packets:\n" + "\n".join(missing))
