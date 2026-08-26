from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator

from .models import QuestionRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id                  INTEGER PRIMARY KEY,
    source              TEXT NOT NULL,
    source_question_id  TEXT NOT NULL,
    source_url          TEXT NOT NULL,
    game_kind           TEXT NOT NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    explanation         TEXT,
    acceptance_criteria TEXT,
    handout_text        TEXT,
    media_urls_json     TEXT NOT NULL DEFAULT '[]',
    package_title       TEXT,
    played_at           TEXT,
    extra_json          TEXT NOT NULL DEFAULT '{}',
    content_hash        TEXT NOT NULL,
    fetched_at          TEXT NOT NULL,
    UNIQUE(source, source_question_id)
);

CREATE INDEX IF NOT EXISTS questions_game_kind_idx ON questions(game_kind);
CREATE INDEX IF NOT EXISTS questions_package_title_idx ON questions(package_title);
"""


class QuestionDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert(self, records: Iterable[QuestionRecord]) -> Counter[str]:
        self.initialize()
        records = [record.finalize() for record in records]
        counts: Counter[str] = Counter()
        column_names = [field.name for field in fields(QuestionRecord)]
        insert_columns = ", ".join(column_names)
        placeholders = ", ".join("?" for _ in column_names)
        updates = ", ".join(
            f"{name}=excluded.{name}"
            for name in column_names
            if name not in {"source", "source_question_id"}
        )
        statement = f"""
            INSERT INTO questions ({insert_columns})
            VALUES ({placeholders})
            ON CONFLICT(source, source_question_id) DO UPDATE SET {updates}
        """

        with self.connect() as connection:
            for record in records:
                previous = connection.execute(
                    """
                    SELECT content_hash FROM questions
                    WHERE source = ? AND source_question_id = ?
                    """,
                    (record.source, record.source_question_id),
                ).fetchone()
                if previous is None:
                    counts["inserted"] += 1
                elif previous["content_hash"] == record.content_hash:
                    counts["unchanged"] += 1
                else:
                    counts["updated"] += 1
                connection.execute(
                    statement,
                    tuple(getattr(record, name) for name in column_names),
                )
        return counts

    def stats(self) -> list[sqlite3.Row]:
        self.initialize()
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT source, game_kind, COUNT(*) AS questions,
                           SUM(CASE WHEN explanation <> '' THEN 1 ELSE 0 END)
                               AS with_explanation,
                           SUM(CASE WHEN media_urls_json <> '[]' THEN 1 ELSE 0 END)
                               AS with_media
                    FROM questions
                    GROUP BY source, game_kind
                    ORDER BY source, game_kind
                    """
                )
            )

