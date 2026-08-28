"""Build the corpus: fetch packages, then rebuild the curated stages.

Stages 2 and 3 are pure functions of stored text and rebuild in seconds, so
they run unconditionally. Translation is not here: it costs money per question
and is driven separately.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .curation import rebuild_duplicates, rebuild_exclusions
from .http import Fetcher
from .ingest import ingest

PENDING_TRANSLATION = """
SELECT COUNT(*) FROM questions_canonical AS q
WHERE NOT EXISTS (
    SELECT 1 FROM translations AS t
    WHERE t.question_id = q.id AND t.content_hash = q.content_hash
)
"""


@dataclass(slots=True)
class CorpusStats:
    """How many questions survive each stage."""

    questions: int = 0
    clean: int = 0
    canonical: int = 0
    translated: int = 0
    awaiting_translation: int = 0


@dataclass(slots=True)
class BuildReport:
    fetched: Counter[str] = field(default_factory=Counter)
    exclusions: dict[str, int] = field(default_factory=dict)
    duplicate_groups: int = 0
    duplicate_rows: int = 0
    stats: CorpusStats = field(default_factory=CorpusStats)


def stats(connection: sqlite3.Connection) -> CorpusStats:
    def count(source: str) -> int:
        return connection.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]

    return CorpusStats(
        questions=count("questions"),
        clean=count("questions_clean"),
        canonical=count("questions_canonical"),
        translated=count("questions_translated"),
        awaiting_translation=connection.execute(PENDING_TRANSLATION).fetchone()[0],
    )


def build(
    client: Fetcher,
    *,
    database: str | Path = db.DEFAULT_PATH,
    pages: int | None = None,
    refresh: bool = False,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    db.initialize(database)
    with db.connect(database) as connection:
        report = BuildReport(
            fetched=ingest(
                connection,
                client,
                pages=pages,
                refresh=refresh,
                workers=workers,
                progress=progress,
            )
        )
        report.exclusions = rebuild_exclusions(connection)
        duplicates = rebuild_duplicates(connection)
        report.duplicate_groups = duplicates["groups"]
        report.duplicate_rows = duplicates["duplicates"]
        connection.commit()
        report.stats = stats(connection)
    return report
