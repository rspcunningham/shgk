from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


def _with_explanation(answer: str, explanation: str) -> str:
    return f"{answer}\n\n{explanation}" if explanation else answer


@dataclass(frozen=True, slots=True)
class Quad:
    id: int
    english_question: str | None
    russian_question: str
    english_answer: str | None
    russian_answer: str


class CorpusReader:
    """Read English/Russian question-answer quads by source question id."""

    def __init__(self, source_db: str | Path, pipeline_db: str | Path):
        source_path = Path(source_db)
        pipeline_path = Path(pipeline_db)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source database not found: {source_path}")
        if not pipeline_path.is_file():
            raise FileNotFoundError(f"Pipeline database not found: {pipeline_path}")
        source_uri = f"file:{source_path.resolve().as_posix()}?mode=ro"
        pipeline_uri = f"file:{pipeline_path.resolve().as_posix()}?mode=ro"
        self._connection = sqlite3.connect(source_uri, uri=True)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("ATTACH DATABASE ? AS pipeline", (pipeline_uri,))

    def read(self, id: int) -> Quad:
        row = self._connection.execute(
            """
            SELECT question.id, question.question, question.answer,
                   question.explanation,
                   translation.status, translation.question_en,
                   translation.answer_en, translation.explanation_en
            FROM questions AS question
            LEFT JOIN pipeline.translations AS translation
              ON translation.source = question.source
             AND translation.source_question_id = question.source_question_id
            WHERE question.id = ?
            """,
            (id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No question with id {id}")

        translated = row["status"] in ("translated", "adapted")
        return Quad(
            id=row["id"],
            english_question=row["question_en"] if translated else None,
            russian_question=row["question"],
            english_answer=(
                _with_explanation(row["answer_en"], row["explanation_en"])
                if translated
                else None
            ),
            russian_answer=_with_explanation(
                row["answer"], row["explanation"] or ""
            ),
        )

    def random_quad(self) -> Quad:
        """A random question that has usable English text."""
        row = self._connection.execute(
            """
            SELECT question.id
            FROM pipeline.translations AS translation
            JOIN questions AS question
              ON question.source = translation.source
             AND question.source_question_id = translation.source_question_id
            WHERE translation.status IN ('translated', 'adapted')
            ORDER BY RANDOM()
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise LookupError("No translated questions available")
        return self.read(row["id"])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CorpusReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
