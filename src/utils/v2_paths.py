"""Path helpers for the v2 experiment layout."""

from pathlib import Path


def get_v2_experiment_root(cfg: dict) -> Path:
    return Path(cfg["output"]["root"]) / cfg["experiment"]["name"]


def get_v2_agent_checkpoint_dir(cfg: dict) -> Path:
    return get_v2_experiment_root(cfg) / "checkpoints" / "agents"


def get_v2_dsdm_guide_dir(cfg: dict, agent_id: int) -> Path:
    return get_v2_experiment_root(cfg) / "checkpoints" / "dsdm_guides" / f"agent_{agent_id}"
