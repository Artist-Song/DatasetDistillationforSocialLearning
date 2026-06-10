"""
Experiment run naming helpers.
"""


def build_base_run_name(cfg) -> str:
    return f"{cfg['dataset']['name']}_{cfg['split']['mode']}_{cfg['model']['name']}"


def format_float_tag(value) -> str:
    return str(value).replace(".", "p")


def build_social_run_name(cfg) -> str:
    base_name = build_base_run_name(cfg)
    social_cfg = cfg.get("social", {})
    packet_cfg = cfg.get("packet", {})
    packet_source = packet_cfg.get("source", "raw")
    packet_kd_mode = social_cfg.get("packet_kd_mode", "sender_subset")
    lambda_retain = social_cfg.get("lambda_retain", 1.0)
    lambda_packet = social_cfg.get("lambda_packet", 1.0)
    return (
        f"{base_name}"
        f"_src-{packet_source}"
        f"_kd-{packet_kd_mode}"
        f"_retain-{format_float_tag(lambda_retain)}"
        f"_packet-{format_float_tag(lambda_packet)}"
    )


def build_packet_only_run_name(cfg) -> str:
    base_name = build_base_run_name(cfg)
    packet_cfg = cfg.get("packet", {})
    social_cfg = cfg.get("social", {})
    packet_only_cfg = cfg.get("packet_only", {})
    packet_source = packet_cfg.get("source", "raw")
    packet_kd_mode = packet_only_cfg.get("packet_kd_mode", social_cfg.get("packet_kd_mode", "sender_subset"))
    lambda_kd = packet_only_cfg.get("lambda_kd", social_cfg.get("lambda_kd", 1.0))
    lambda_retain = packet_only_cfg.get("lambda_retain", 0.0)
    freeze_backbone = packet_only_cfg.get("freeze_backbone", False)
    head_tag = "head" if freeze_backbone else "full"
    return (
        f"{base_name}"
        f"_src-{packet_source}"
        f"_po-kd-{packet_kd_mode}"
        f"_po-lkd-{format_float_tag(lambda_kd)}"
        f"_po-retain-{format_float_tag(lambda_retain)}"
        f"_po-{head_tag}"
    )
