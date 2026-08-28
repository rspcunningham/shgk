from __future__ import annotations

import io

from shgk.progress import Progress, format_duration


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_format_duration_scales_with_size() -> None:
    assert format_duration(9) == "9s"
    assert format_duration(75) == "1m15s"
    assert format_duration(3 * 3600 + 4 * 60) == "3h04m"
    assert format_duration(-5) == "0s"


def test_a_terminal_is_rewritten_in_place() -> None:
    stream = FakeTTY()
    progress = Progress("packages", stream=stream, min_interval=0)
    progress(1, 10)
    progress(2, 10)
    output = stream.getvalue()
    assert output.count("\r") == 2
    assert "\n" not in output          # one line, rewritten
    assert "packages: 2/10" in output


def test_a_pipe_gets_whole_lines_and_no_carriage_returns() -> None:
    """A captured log must stay readable, so it gets lines instead of redraws."""
    stream = io.StringIO()
    progress = Progress("packages", stream=stream, line_interval=0)
    progress(1, 10)
    progress(2, 10)
    assert "\r" not in stream.getvalue()
    assert stream.getvalue().count("\n") == 2


def test_updates_are_throttled_but_completion_always_shows() -> None:
    stream = FakeTTY()
    progress = Progress("packages", stream=stream, min_interval=3600)
    progress(1, 10)   # first call renders
    progress(2, 10)   # throttled away
    progress(10, 10)  # done: always rendered, however recently we drew
    assert "packages: 2/10" not in stream.getvalue()
    assert "packages: 10/10" in stream.getvalue()


def test_report_carries_percent_rate_and_estimate() -> None:
    stream = FakeTTY()
    progress = Progress("questions", stream=stream, min_interval=0)
    progress.started_at -= 10.0       # pretend ten seconds have passed
    progress(5, 100, "translated")
    line = stream.getvalue()
    assert "questions: 5/100" in line
    assert "5%" in line
    assert "0.5/s" in line
    assert "eta 3m10s" in line
    assert "translated" in line


def test_closing_summarises_and_ends_the_line() -> None:
    stream = FakeTTY()
    with Progress("packages", stream=stream, min_interval=0) as progress:
        progress(4, 4)
    assert stream.getvalue().endswith("\n")
    assert "packages: 4 in " in stream.getvalue()


def test_closing_without_work_says_so() -> None:
    stream = io.StringIO()
    with Progress("packages", stream=stream):
        pass
    assert "nothing to do" in stream.getvalue()
