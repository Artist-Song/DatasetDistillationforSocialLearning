"""
config.py

作用：
    负责读取和管理项目中的 YAML 配置文件，为后续的训练、蒸馏、
    social packet 构建、实验运行入口提供统一的配置读取接口。

主要部分：
    1. load_yaml(path): 读取单个 yaml 文件
    2. 这里后续可以继续扩展：
       - merge_configs(...)
       - override_config(...)
       - save_config(...)

输入输出：
    - 输入：yaml 文件路径
    - 输出：Python 字典（dict）

后续扩展建议：
    - 后面接入 autoresearch 时，可以在这里增加“配置覆盖”和“实验配置落盘”功能
    - 后面如果要支持命令行参数覆盖 yaml，也建议放在这里统一处理
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """
    读取一个 YAML 配置文件，并返回字典。

    参数：
        path: 配置文件路径

    返回：
        解析后的 Python 字典
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)