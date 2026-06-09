"""
Experiment id and output path helpers for the generalist pipeline.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


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


def get_experiment_metadata(cfg: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "experiment_id": get_experiment_id(cfg, config_path=config_path),
        "experiment_root": str(get_experiment_root(cfg, config_path=config_path)),
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
