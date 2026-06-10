"""Configuration loading helpers."""

from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        return literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"unsupported yaml line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        parsed_value = _parse_scalar(value)
        current[key] = parsed_value
        if isinstance(parsed_value, dict) and value.strip() == "":
            stack.append((indent, parsed_value))

    return root


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    PyYAML is used when installed. The fallback parser supports the simple
    nested key/value YAML shape used by the v2 scaffold configs.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    text = path_obj.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml(text)
