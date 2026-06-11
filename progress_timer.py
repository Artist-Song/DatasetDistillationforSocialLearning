import sys
import time


def format_seconds(seconds):
    """把秒数格式化为便于阅读的时间字符串。"""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressTimer:
    """显示单行进度条、已用时间和预计剩余时间。"""

    def __init__(self, total, name="progress", width=28, min_interval=1.0, stream=None):
        """初始化进度条计时器。"""
        self.total = max(1, int(total))
        self.name = name
        self.width = max(10, int(width))
        self.min_interval = float(min_interval)
        self.stream = stream or sys.stdout
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.current = 0

    def update(self, current=None, extra=""):
        """更新进度条，current 为空时自动前进一步。"""
        if current is None:
            self.current += 1
        else:
            self.current = int(current)
        self.current = min(max(self.current, 0), self.total)

        now = time.time()
        if self.current < self.total and now - self.last_print_time < self.min_interval:
            return
        self.last_print_time = now

        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start_time
        eta = elapsed * (1.0 - ratio) / ratio if ratio > 0 else 0
        message = (
            f"\r[{self.name}] [{bar}] "
            f"{self.current}/{self.total} {ratio * 100:6.2f}% "
            f"elapsed {format_seconds(elapsed)} eta {format_seconds(eta)}"
        )
        if extra:
            message += f" | {extra}"
        self.stream.write(message)
        self.stream.flush()
        if self.current >= self.total:
            self.stream.write("\n")
            self.stream.flush()

    def close(self, extra=""):
        """结束进度条，并强制打印最终状态。"""
        self.update(self.total, extra=extra)
