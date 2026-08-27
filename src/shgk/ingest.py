"""Stage 1: fetch GotQuestions packages into the corpus.

Idempotent through the packages table: a package whose page hashes the same as
last time is not re-parsed, and a package that failed is recorded as such so it
is distinguishable from one that was never attempted.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from typing import Callable

from .http import HttpClient
from .sources.gotquestions import (
    BASE_URL,
    PACK_URL,
    GotQuestionsParseError,
    ParsedPack,
    discover_pack_ids,
    parse_pack,
)

SOURCE = "gotquestions"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _index_url(page: int) -> str:
    return BASE_URL if page == 1 else f"{BASE_URL}/?page={page}"


def discover(
    client: HttpClient, *, pages: int | None = None
) -> list[int]:
    """Walk the package index until a page introduces nothing new."""
    found: list[int] = []
    seen: set[int] = set()
    page = 1
    while pages is None or page <= pages:
        html = client.get(_index_url(page)).text
        new = [pack_id for pack_id in discover_pack_ids(html) if pack_id not in seen]
        if not new:
            break
        found.extend(new)
        seen.update(new)
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
    client: HttpClient,
    *,
    pack_ids: list[int] | None = None,
    pages: int | None = None,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Counter[str]:
    """Fetch and store packages.

    A package whose page hashes the same as last time is not re-parsed, which
    avoids rewriting rows that have not changed. It is still fetched: the hash
    is only knowable after the request.
    """
    if pack_ids is None:
        pack_ids = discover(client, pages=pages)
    known = {
        row["id"]: row["page_hash"]
        for row in connection.execute("SELECT id, page_hash FROM packages")
    }
    counts: Counter[str] = Counter()

    for index, pack_id in enumerate(pack_ids, start=1):
        try:
            html = client.get(PACK_URL.format(pack_id)).text
        except Exception as error:
            counts["fetch_error"] += 1
            _record_failure(connection, pack_id, "http_error", str(error))
            continue

        page_hash = sha256(html.encode("utf-8")).hexdigest()[:16]
        if known.get(pack_id) == page_hash and not refresh:
            counts["unchanged"] += 1
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
        counts["new" if pack_id not in known else "updated"] += 1
        counts["questions"] += len(pack.questions)
        if progress and index % 500 == 0:
            progress(f"  {index:,}/{len(pack_ids):,} packages")
    connection.commit()
    return counts
