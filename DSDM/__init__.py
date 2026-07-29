"""Stable lazy exports for the repository-local DSDM implementation."""

import importlib
import sys
from pathlib import Path


__all__ = (
    "ClassMemDataLoader",
    "Synthesizer",
    "diffaug",
    "load_resized_data",
    "matchloss",
    "run_dsdm",
)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    implementation = importlib.import_module(".DSDM", __name__)
    value = getattr(implementation, name)
    globals()[name] = value
    return value
