from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3


BASIC_FILTER_VERSION = 1

BASIC_FILTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS basic_filter_results (
    source                  TEXT NOT NULL,
    source_question_id      TEXT NOT NULL,
    source_content_hash     TEXT NOT NULL,
    eligible                INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    rejection_reasons_json  TEXT NOT NULL CHECK (json_valid(rejection_reasons_json)),
    filter_version          INTEGER NOT NULL,
    evaluated_at            TEXT NOT NULL,
    PRIMARY KEY (source, source_question_id)
);

CREATE INDEX IF NOT EXISTS basic_filter_eligible_idx
    ON basic_filter_results(eligible, source);
"""

_HAS_QUESTION = "question <> '' AND question NOT LIKE '[Media question;%'"
_IS_ELIGIBLE = f"""
    ({_HAS_QUESTION})
    AND answer <> ''
    AND explanation <> ''
    AND json_array_length(media_urls_json) = 0
"""

_REJECTION_REASONS = f"""
    '[' || rtrim(
        CASE WHEN NOT ({_HAS_QUESTION}) THEN '"missing_question",' ELSE '' END ||
        CASE WHEN answer = '' THEN '"missing_answer",' ELSE '' END ||
        CASE WHEN explanation = '' THEN '"missing_explanation",' ELSE '' END ||
        CASE WHEN json_array_length(media_urls_json) > 0 THEN '"has_media",' ELSE '' END,
        ','
    ) || ']'
"""


class BasicFilterPipeline:
    """Materialize deterministic source eligibility without copying source text."""

    def __init__(self, source_db: str | Path, pipeline_db: str | Path):
        self.source_db = Path(source_db)
        self.pipeline_db = Path(pipeline_db)
        if self.source_db.resolve() == self.pipeline_db.resolve():
            raise ValueError("source and pipeline databases must be separate files")

    def run(self) -> dict[str, int]:
        if not self.source_db.is_file():
            raise FileNotFoundError(f"Source database not found: {self.source_db}")
        self.pipeline_db.parent.mkdir(parents=True, exist_ok=True)
        evaluated_at = datetime.now(UTC).isoformat()
        source_uri = f"file:{self.source_db.resolve().as_posix()}?mode=ro"

        with sqlite3.connect(self.pipeline_db, uri=True) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(BASIC_FILTER_SCHEMA)
            connection.execute(f"PRAGMA user_version = {BASIC_FILTER_VERSION}")
            connection.execute("ATTACH DATABASE ? AS raw", (source_uri,))
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    f"""
                    INSERT INTO basic_filter_results (
                        source, source_question_id, source_content_hash,
                        eligible, rejection_reasons_json, filter_version,
                        evaluated_at
                    )
                    SELECT source, source_question_id, content_hash,
                           CASE WHEN {_IS_ELIGIBLE} THEN 1 ELSE 0 END,
                           {_REJECTION_REASONS},
                           ?, ?
                    FROM raw.questions
                    WHERE 1
                    ON CONFLICT(source, source_question_id) DO UPDATE SET
                        source_content_hash = excluded.source_content_hash,
                        eligible = excluded.eligible,
                        rejection_reasons_json = excluded.rejection_reasons_json,
                        filter_version = excluded.filter_version,
                        evaluated_at = excluded.evaluated_at
                    """,
                    (BASIC_FILTER_VERSION, evaluated_at),
                )
                connection.execute(
                    """
                    DELETE FROM basic_filter_results AS result
                    WHERE NOT EXISTS (
                        SELECT 1 FROM raw.questions AS question
                        WHERE question.source = result.source
                          AND question.source_question_id = result.source_question_id
                    )
                    """
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

            total, eligible = connection.execute(
                """
                SELECT COUNT(*), SUM(eligible)
                FROM basic_filter_results
                WHERE filter_version = ?
                """,
                (BASIC_FILTER_VERSION,),
            ).fetchone()
            connection.execute("DETACH DATABASE raw")
        return {
            "total": int(total or 0),
            "eligible": int(eligible or 0),
            "rejected": int((total or 0) - (eligible or 0)),
        }

    def stats(self) -> dict[str, object]:
        if not self.pipeline_db.is_file():
            raise FileNotFoundError(f"Pipeline database not found: {self.pipeline_db}")
        with sqlite3.connect(self.pipeline_db) as connection:
            connection.row_factory = sqlite3.Row
            total = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(eligible) AS eligible,
                       SUM(NOT eligible) AS rejected
                FROM basic_filter_results
                """
            ).fetchone()
            by_source = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT source, COUNT(*) AS total,
                           SUM(eligible) AS eligible,
                           SUM(NOT eligible) AS rejected
                    FROM basic_filter_results
                    GROUP BY source ORDER BY source
                    """
                )
            ]
            reasons = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT reason.value AS reason, COUNT(*) AS questions
                    FROM basic_filter_results AS result,
                         json_each(result.rejection_reasons_json) AS reason
                    GROUP BY reason.value ORDER BY questions DESC, reason.value
                    """
                )
            ]
        return {
            "total": int(total["total"] or 0),
            "eligible": int(total["eligible"] or 0),
            "rejected": int(total["rejected"] or 0),
            "by_source": by_source,
            "rejection_reasons": reasons,
        }
