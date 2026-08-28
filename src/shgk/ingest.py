"""Stage 1: fetch GotQuestions packages into the corpus.

Idempotent through the packages table: a package whose page hashes the same as
last time is not re-parsed, and a package that failed is recorded as such so it
is distinguishable from one that was never attempted.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from .http import Fetcher
from .progress import Reporter
from .sources.gotquestions import (
    BASE_URL,
    PACK_URL,
    GotQuestionsParseError,
    ParsedPack,
    discover_pack_ids,
    parse_pack,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _index_url(page: int) -> str:
    return BASE_URL if page == 1 else f"{BASE_URL}/?page={page}"


# A package is settled once it has been fetched and parsed. Failures are not,
# so they are retried on the next run.
SETTLED = ("ok", "empty")

# Solve statistics land after a tournament is played, so a package fetched soon
# after publication has questions with no difficulty data yet. Recently played
# packages are therefore never treated as settled: without this the newest
# questions would keep their empty statistics permanently.
RESTLESS_DAYS = 180


class IndexUnavailable(RuntimeError):
    """The package index returned no packages.

    Raised rather than reported as "nothing new", because the two are
    indistinguishable from the outside and only one of them is fine. The site
    serves a maintenance stub with no package links, and treating that as an
    up-to-date corpus hides an outage for as long as it lasts.
    """


def discover(
    client: Fetcher,
    *,
    settled: frozenset[int] = frozenset(),
    pages: int | None = None,
) -> list[int]:
    """Walk the package index, newest first, collecting packages to fetch.

    The server offers no ETag or Last-Modified, so a package cannot be checked
    for changes without downloading it in full. The index is ordered newest
    first, so once a page contains nothing but packages already settled there is
    nothing older worth walking to, and crawling stops. That makes the ordinary
    incremental run cost one index page rather than several hundred.
    """
    found: list[int] = []
    seen: set[int] = set()
    page = 1
    while pages is None or page <= pages:
        ids = discover_pack_ids(client.get(_index_url(page)).text)
        if not ids:
            if page == 1:
                raise IndexUnavailable(
                    f"{_index_url(page)} listed no packages; the site may be "
                    "down for maintenance"
                )
            break
        fresh = [pack_id for pack_id in ids if pack_id not in seen]
        seen.update(fresh)
        wanted = [pack_id for pack_id in fresh if pack_id not in settled]
        found.extend(wanted)
        if not wanted and pages is None:
            break
        page += 1
    return found


def _store(connection: sqlite3.Connection, pack: ParsedPack, page_hash: str) -> None:
    now = _now()
    connection.execute(
        """
        INSERT INTO packages (id, title, slug, played_at_start, played_at_end,
            editor_ids, editor_names, url, status, http_status, page_hash,
            questions_found, first_seen_at, fetched_at, error)
        VALUES (?,?,?,?,?,?,?,?,'ok',200,?,?,?,?,'')
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title, slug = excluded.slug,
            played_at_start = excluded.played_at_start,
            played_at_end = excluded.played_at_end,
            editor_ids = excluded.editor_ids, editor_names = excluded.editor_names,
            status = 'ok', http_status = 200, page_hash = excluded.page_hash,
            questions_found = excluded.questions_found,
            fetched_at = excluded.fetched_at, error = ''
        """,
        (pack.id, pack.title, pack.slug, pack.played_at_start, pack.played_at_end,
         pack.editor_ids, pack.editor_names, PACK_URL.format(pack.id), page_hash,
         len(pack.questions), now, now),
    )
    connection.executemany(
        """
        INSERT INTO questions (id, package_id, question_number, question, answer,
            explanation, acceptance_criteria, handout_text, host_note, kind,
            has_media, media_urls, author_ids, author_names, tournament_ids,
            source_references, taken_down, solve_percentages, correct_answers,
            content_hash, normalized_hash)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            package_id = excluded.package_id,
            question_number = excluded.question_number,
            question = excluded.question, answer = excluded.answer,
            explanation = excluded.explanation,
            acceptance_criteria = excluded.acceptance_criteria,
            handout_text = excluded.handout_text, host_note = excluded.host_note,
            kind = excluded.kind, has_media = excluded.has_media,
            media_urls = excluded.media_urls, author_ids = excluded.author_ids,
            author_names = excluded.author_names,
            tournament_ids = excluded.tournament_ids,
            source_references = excluded.source_references,
            taken_down = excluded.taken_down,
            solve_percentages = excluded.solve_percentages,
            correct_answers = excluded.correct_answers,
            content_hash = excluded.content_hash,
            normalized_hash = excluded.normalized_hash
        """,
        [
            (q.id, pack.id, q.question_number, q.question, q.answer, q.explanation,
             q.acceptance_criteria, q.handout_text, q.host_note, q.kind, q.has_media,
             q.media_urls, q.author_ids, q.author_names, q.tournament_ids,
             q.source_references, q.taken_down, q.solve_percentages,
             q.correct_answers, q.content_hash, q.normalized_hash)
            for q in pack.questions
        ],
    )


def _mark_ok(connection: sqlite3.Connection, pack_id: int) -> None:
    """Clear a failure without reparsing.

    Only a package that was once stored successfully has a page_hash, so a
    matching hash proves this exact page parsed before and the stored questions
    are still good. Without this a package that failed once and then came back
    unchanged would keep its failed status forever, and be refetched on every
    run for the rest of time.
    """
    connection.execute(
        "UPDATE packages SET status = 'ok', http_status = 200, error = '', "
        "fetched_at = ? WHERE id = ?",
        (_now(), pack_id),
    )


def _record_failure(
    connection: sqlite3.Connection, pack_id: int, status: str, error: str
) -> None:
    now = _now()
    connection.execute(
        """
        INSERT INTO packages (id, url, status, questions_found,
            first_seen_at, fetched_at, error)
        VALUES (?,?,?,0,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status, fetched_at = excluded.fetched_at,
            error = excluded.error
        """,
        (pack_id, PACK_URL.format(pack_id), status, now, now, error[:500]),
    )


def ingest(
    connection: sqlite3.Connection,
    client: Fetcher,
    *,
    pack_ids: list[int] | None = None,
    pages: int | None = None,
    refresh: bool = False,
    workers: int = 8,
    progress: Reporter | None = None,
) -> Counter[str]:
    """Fetch and store packages.

    Pages take seconds each and run to a megabyte, so fetches overlap; parsing
    and every write stay on the calling thread, since the SQLite connection
    belongs to it.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=RESTLESS_DAYS)).strftime("%Y-%m-%d")
    known = {}
    settled_ids = set()
    for row in connection.execute(
        "SELECT id, status, page_hash, played_at_start FROM packages"
    ):
        known[row["id"]] = (row["status"], row["page_hash"])
        recent = (row["played_at_start"] or "") >= cutoff
        if row["status"] in SETTLED and not recent:
            settled_ids.add(row["id"])
    settled = frozenset(settled_ids)
    if pack_ids is None:
        pack_ids = discover(
            client, settled=frozenset() if refresh else settled, pages=pages
        )
    targets = [p for p in pack_ids if refresh or p not in settled]
    counts: Counter[str] = Counter(skipped=len(pack_ids) - len(targets))
    if not targets:
        return counts

    def fetch(pack_id: int) -> tuple[int, str | None, Exception | None]:
        try:
            return pack_id, client.get(PACK_URL.format(pack_id)).text, None
        except Exception as error:  # keep a long crawl moving
            return pack_id, None, error

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for pack_id, html, error in pool.map(fetch, targets):
            done += 1
            if html is None:
                counts["fetch_error"] += 1
                _record_failure(connection, pack_id, "http_error", str(error))
                if progress:
                    progress(done, len(targets), "fetch failed")
                continue

            previous_status, previous_hash = known.get(pack_id, (None, None))
            page_hash = sha256(html.encode("utf-8")).hexdigest()[:16]
            if not refresh and previous_hash == page_hash:
                counts["unchanged"] += 1
                if previous_status != "ok":
                    _mark_ok(connection, pack_id)
                    counts["recovered"] += 1
                if progress:
                    progress(done, len(targets))
                continue

            try:
                pack = parse_pack(html)
            except GotQuestionsParseError as error:
                counts["parse_error"] += 1
                _record_failure(connection, pack_id, "parse_error", str(error))
                continue

            if not pack.questions:
                counts["empty"] += 1
                _record_failure(connection, pack_id, "empty", "")
                continue

            _store(connection, pack, page_hash)
            # A package whose only previous record was a failure has never had
            # its questions stored, so this is the first real ingest, not an update.
            counts["updated" if previous_status == "ok" else "new"] += 1
            counts["questions"] += len(pack.questions)
            if progress:
                progress(done, len(targets))
    connection.commit()
    return counts
