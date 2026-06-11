"""Path helpers for the v2 experiment layout."""

from pathlib import Path


def get_v2_experiment_root(cfg: dict) -> Path:
    return Path(cfg["output"]["root"]) / cfg["experiment"]["name"]


def get_v2_agent_checkpoint_dir(cfg: dict) -> Path:
    return get_v2_experiment_root(cfg) / "checkpoints" / "agents"


def get_v2_dsdm_guide_dir(cfg: dict, agent_id: int) -> Path:
    return get_v2_experiment_root(cfg) / "checkpoints" / "dsdm_guides" / f"agent_{agent_id}"


def get_v2_packet_dir(cfg: dict, packet_source: str) -> Path:
    return get_v2_experiment_root(cfg) / "packets" / packet_source


def get_v2_packet_visual_dir(cfg: dict, packet_source: str) -> Path:
    return get_v2_experiment_root(cfg) / "packet_visuals" / packet_source


def get_v2_socialized_checkpoint_dir(cfg: dict, packet_source: str, adaptation_mode: str = "last_block_anchor") -> Path:
    base_dir = get_v2_experiment_root(cfg) / "checkpoints" / "socialized" / packet_source
    if adaptation_mode == "last_block_anchor":
        return base_dir
    return base_dir / adaptation_mode


def get_v2_metrics_dir(cfg: dict) -> Path:
    return get_v2_experiment_root(cfg) / "metrics"


def get_v2_comparison_dir(cfg: dict) -> Path:
    return get_v2_experiment_root(cfg) / "comparison"
