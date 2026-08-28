"""Progress reporting for runs long enough that silence is indistinguishable
from a hang.

A terminal gets a line rewritten in place; anything else -- a pipe, a log file,
nohup -- gets an occasional ordinary line, because carriage returns turn a
captured log into one unreadable smear.
"""

from __future__ import annotations

import sys
import time
from types import TracebackType
from typing import Protocol, TextIO


class Reporter(Protocol):
    """What the long-running stages call to say where they are."""

    def __call__(self, done: int, total: int, note: str = "") -> None: ...


def format_duration(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class Progress:
    """Report progress, with a rate and an estimate of what is left."""

    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        min_interval: float = 0.2,
        line_interval: float = 30.0,
    ):
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.line_interval = line_interval
        self.interactive = getattr(self.stream, "isatty", lambda: False)()
        self.started_at = time.monotonic()
        self._last_render = 0.0
        self._width = 0
        self._done = 0
        self._total = 0

    def _render(self, done: int, total: int, note: str) -> str:
        elapsed = time.monotonic() - self.started_at
        rate = done / elapsed if elapsed > 0 else 0.0
        parts = [f"{self.label}: {done:,}/{total:,}"]
        if total:
            parts.append(f"{done / total * 100:3.0f}%")
        if rate:
            parts.append(f"{rate:.1f}/s")
            remaining = total - done
            if remaining > 0:
                parts.append(f"eta {format_duration(remaining / rate)}")
        if note:
            parts.append(note)
        return "  ".join(parts)

    def __call__(self, done: int, total: int, note: str = "") -> None:
        self._done, self._total = done, total
        now = time.monotonic()
        interval = self.min_interval if self.interactive else self.line_interval
        if now - self._last_render < interval and done != total:
            return
        self._last_render = now
        line = self._render(done, total, note)
        if self.interactive:
            # Pad to erase whatever the previous, possibly longer, line left.
            self.stream.write("\r" + line.ljust(self._width))
            self._width = max(self._width, len(line))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self) -> None:
        elapsed = time.monotonic() - self.started_at
        summary = (
            f"{self.label}: {self._done:,} in {format_duration(elapsed)}"
            if self._done
            else f"{self.label}: nothing to do"
        )
        if self.interactive:
            self.stream.write("\r" + summary.ljust(self._width) + "\n")
        else:
            self.stream.write(summary + "\n")
        self.stream.flush()

    def __enter__(self) -> Progress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
