"""Read English/Russian question pairs out of the curated views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .db import DEFAULT_PATH, connect


def _with_explanation(answer: str, explanation: str) -> str:
    return f"{answer}\n\n{explanation}" if explanation else answer


@dataclass(frozen=True, slots=True)
class Quad:
    id: int
    english_question: str
    russian_question: str
    english_answer: str
    russian_answer: str


_SELECT = """
    SELECT id, question, answer, explanation,
           question_en, answer_en, explanation_en
    FROM questions_translated
"""


class CorpusReader:
    """Reads the questions_translated view, which is already the quad join."""

    def __init__(self, database: str | Path = DEFAULT_PATH):
        self._connection = connect(database, read_only=True)

    def _quad(self, row) -> Quad:
        return Quad(
            id=row["id"],
            english_question=row["question_en"],
            russian_question=row["question"],
            english_answer=_with_explanation(row["answer_en"], row["explanation_en"]),
            russian_answer=_with_explanation(row["answer"], row["explanation"]),
        )

    def read(self, question_id: int) -> Quad:
        row = self._connection.execute(
            f"{_SELECT} WHERE id = ?", (question_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No translated question with id {question_id}")
        return self._quad(row)

    def random(self) -> Quad:
        row = self._connection.execute(
            f"{_SELECT} ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if row is None:
            raise LookupError("No translated questions available")
        return self._quad(row)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CorpusReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
