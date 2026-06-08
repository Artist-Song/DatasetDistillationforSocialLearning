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
    packet_kd_mode = social_cfg.get("packet_kd_mode", "sender_subset")
    lambda_retain = social_cfg.get("lambda_retain", 1.0)
    lambda_packet = social_cfg.get("lambda_packet", 1.0)
    return (
        f"{base_name}"
        f"_kd-{packet_kd_mode}"
        f"_retain-{format_float_tag(lambda_retain)}"
        f"_packet-{format_float_tag(lambda_packet)}"
    )