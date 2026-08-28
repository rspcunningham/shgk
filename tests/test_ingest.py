from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from shgk import db
from shgk.ingest import RESTLESS_DAYS, IndexUnavailable, discover, ingest
from shgk.sources.gotquestions import BASE_URL, PACK_URL

QUESTION = "Назовите предмет, который используется для чая в поезде."

# A package only settles once it is old enough to have collected its statistics.
RECENT = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
OLD = (datetime.now(UTC) - timedelta(days=RESTLESS_DAYS + 30)).strftime("%Y-%m-%d")


def _pack_html(pack_id: int, *, questions: int = 1, played_at: str = "2020-01-01",
                title: str = "Пакет", suffix: str = "") -> str:
    pack = {
        "id": pack_id,
        "title": title,
        "startDate": played_at,
        "questions": questions,
        "tours": [{"id": pack_id * 10, "questions": [
            {"id": pack_id * 1000 + n, "number": n,
             "text": f"{QUESTION} {pack_id}-{n}{suffix}", "answer": "Стакан"}
            for n in range(questions)
        ]}],
    }
    stream = "prefix" + json.dumps({"pack": pack}, ensure_ascii=False,
                                   separators=(",", ":"))
    return (f"<html><script>self.__next_f.push"
            f"([1,{json.dumps(stream, ensure_ascii=False)}])</script></html>")


def _index_html(pack_ids: list[int]) -> str:
    links = "".join(f'<a href="/pack/{i}">p{i}</a>' for i in pack_ids)
    return f"<html><body>{links}</body></html>"


class FakeHttp:
    """Serves canned pages and records every request, so we can assert on traffic."""

    def __init__(self, pages: dict[str, str], fail: set[str] | None = None):
        self.pages = pages
        self.fail = fail or set()
        self.requests: list[str] = []

    def get(self, url: str):
        self.requests.append(url)
        if url in self.fail:
            raise RuntimeError("connection reset")
        if url not in self.pages:
            if "?page=" in url:  # past the end of the index: a page with no links
                return SimpleNamespace(text=_index_html([]))
            raise RuntimeError(f"404 {url}")
        return SimpleNamespace(text=self.pages[url])

    def pack_requests(self) -> list[str]:
        return [u for u in self.requests if "/pack/" in u]


def _site(pack_ids: list[int], *, played_at: str = OLD, **kw) -> FakeHttp:
    pages = {BASE_URL: _index_html(pack_ids)}
    for i in pack_ids:
        pages[PACK_URL.format(i)] = _pack_html(i, played_at=played_at, **kw)
    return FakeHttp(pages)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "shgk.sqlite3"
    db.initialize(path)
    return path


def _counts(path, http, **kw):
    with db.connect(path) as connection:
        return ingest(connection, http, workers=2, **kw)


def _statuses(path) -> dict[int, str]:
    with db.connect(path, read_only=True) as connection:
        return {r["id"]: r["status"] for r in connection.execute(
            "SELECT id, status FROM packages")}


# --- discovery -------------------------------------------------------------

def test_discover_stops_at_the_first_page_with_nothing_unseen() -> None:
    http = FakeHttp({
        BASE_URL: _index_html([3, 2]),
        f"{BASE_URL}/?page=2": _index_html([1]),
    })
    assert discover(http, settled=frozenset({3, 2})) == []
    # Page 2 is never requested: the index is newest first, so there is nothing
    # older worth walking to.
    assert http.requests == [BASE_URL]


def test_discover_keeps_walking_while_pages_yield_unseen_packages() -> None:
    http = FakeHttp({
        BASE_URL: _index_html([4, 3]),
        f"{BASE_URL}/?page=2": _index_html([2, 1]),
        f"{BASE_URL}/?page=3": _index_html([]),
    })
    assert discover(http) == [4, 3, 2, 1]


def test_discover_respects_an_explicit_page_limit() -> None:
    http = FakeHttp({
        BASE_URL: _index_html([4, 3]),
        f"{BASE_URL}/?page=2": _index_html([2, 1]),
    })
    assert discover(http, pages=1) == [4, 3]


# --- idempotency -----------------------------------------------------------

def test_first_run_stores_packages_and_questions(database) -> None:
    counts = _counts(database, _site([1, 2]))
    assert counts["new"] == 2 and counts["questions"] == 2
    with db.connect(database, read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 2
        assert c.execute("SELECT COUNT(*) FROM packages").fetchone()[0] == 2


def test_second_run_fetches_nothing_at_all(database) -> None:
    _counts(database, _site([1, 2]))
    http = _site([1, 2])
    counts = _counts(database, http)
    assert counts["new"] == 0 and counts["updated"] == 0
    assert http.pack_requests() == []          # not even a conditional request
    assert http.requests == [BASE_URL]         # one index page, then stop


def test_repeated_runs_do_not_change_the_corpus(database) -> None:
    def snapshot():
        with db.connect(database, read_only=True) as c:
            return c.execute(
                "SELECT id, content_hash FROM questions ORDER BY id").fetchall()
    _counts(database, _site([1, 2], questions=3))
    first = snapshot()
    for _ in range(3):
        _counts(database, _site([1, 2], questions=3))
    assert snapshot() == first


def test_changed_page_is_reparsed_and_unchanged_page_is_not(database) -> None:
    """A restless package is refetched every run, but only reparsed when it moved."""
    _counts(database, _site([1], played_at=RECENT))
    counts = _counts(database, _site([1], played_at=RECENT))
    assert counts["unchanged"] == 1
    counts = _counts(database, _site([1], played_at=RECENT, suffix=" переработано"))
    assert counts["updated"] == 1


def test_refresh_refetches_settled_packages(database) -> None:
    _counts(database, _site([1, 2]))
    http = _site([1, 2])
    counts = _counts(database, http, refresh=True)
    assert len(http.pack_requests()) == 2
    assert counts["updated"] == 2


# --- recency ---------------------------------------------------------------

def test_recently_played_packages_are_refetched(database) -> None:
    """Solve statistics arrive late, so new packages must not settle."""
    recent = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
    _counts(database, _site([1], played_at=recent))
    http = _site([1], played_at=recent)
    _counts(database, http)
    assert http.pack_requests() == [PACK_URL.format(1)]


def test_old_packages_settle(database) -> None:
    old = (datetime.now(UTC) - timedelta(days=RESTLESS_DAYS + 30)).strftime("%Y-%m-%d")
    _counts(database, _site([1], played_at=old))
    http = _site([1], played_at=old)
    _counts(database, http)
    assert http.pack_requests() == []


# --- failures --------------------------------------------------------------

def test_fetch_failure_is_recorded_and_retried(database) -> None:
    broken = FakeHttp({BASE_URL: _index_html([1])}, fail={PACK_URL.format(1)})
    counts = _counts(database, broken)
    assert counts["fetch_error"] == 1
    assert _statuses(database) == {1: "http_error"}

    # A failure is not settled, so the next run tries again -- and succeeds.
    http = _site([1])
    counts = _counts(database, http)
    assert counts["new"] == 1
    assert _statuses(database) == {1: "ok"}


def test_unparseable_page_is_recorded(database) -> None:
    http = FakeHttp({BASE_URL: _index_html([1]),
                     PACK_URL.format(1): "<html>no rsc stream</html>"})
    counts = _counts(database, http)
    assert counts["parse_error"] == 1
    assert _statuses(database) == {1: "parse_error"}


def test_a_failure_does_not_discard_stored_questions(database) -> None:
    _counts(database, _site([1], questions=2))
    broken = FakeHttp({BASE_URL: _index_html([1])}, fail={PACK_URL.format(1)})
    _counts(database, broken, pack_ids=[1], refresh=True)
    with db.connect(database, read_only=True) as c:
        assert c.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 2


def test_a_package_that_failed_then_returns_unchanged_recovers(database) -> None:
    """A transient failure must not strand a healthy package as broken forever."""
    _counts(database, _site([1], played_at=RECENT))
    assert _statuses(database) == {1: "ok"}

    broken = FakeHttp({BASE_URL: _index_html([1])}, fail={PACK_URL.format(1)})
    _counts(database, broken)
    assert _statuses(database) == {1: "http_error"}

    # The page is byte-identical to the one already stored, so nothing needs
    # reparsing -- but the package is healthy again and must say so, or it is
    # refetched on every run for the rest of time.
    counts = _counts(database, _site([1], played_at=RECENT))
    assert counts["unchanged"] == 1 and counts["recovered"] == 1
    assert _statuses(database) == {1: "ok"}


def test_an_index_with_no_packages_is_an_error_not_an_empty_result() -> None:
    """A maintenance stub and a corpus with nothing new look identical otherwise."""
    http = FakeHttp({BASE_URL: "<html><body>Технические работы</body></html>"})
    with pytest.raises(IndexUnavailable):
        discover(http)


def test_a_later_empty_page_just_ends_the_crawl() -> None:
    http = FakeHttp({
        BASE_URL: _index_html([2, 1]),
        f"{BASE_URL}/?page=2": _index_html([]),
    })
    assert discover(http) == [2, 1]
