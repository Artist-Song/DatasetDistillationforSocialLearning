"""Small progress and timing helpers for v2 experiment entry points."""

from contextlib import ContextDecorator
from time import perf_counter

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs.
    def tqdm(iterable, **_kwargs):
        return iterable


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class StageTimer(ContextDecorator):
    def __init__(self, name: str):
        self.name = name
        self.started_at = None
        self.elapsed = 0.0

    def __enter__(self):
        self.started_at = perf_counter()
        print(f"[timer] start {self.name}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.elapsed = perf_counter() - self.started_at
        status = "failed" if exc_type else "done"
        print(f"[timer] {status} {self.name}: {format_seconds(self.elapsed)}")
        return False


def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs)
