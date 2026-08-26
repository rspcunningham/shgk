from __future__ import annotations

import sqlite3

from shgk.database import QuestionDatabase
from shgk.models import QuestionRecord


def _record(question: str = "Вопрос") -> QuestionRecord:
    return QuestionRecord(
        source="test",
        source_question_id="1",
        source_url="https://example.test/1",
        game_kind="sport_chgk",
        question=question,
        answer="Ответ",
        explanation="Объяснение",
        fetched_at="2025-01-01T00:00:00+00:00",
    )


def test_database_is_one_table_and_upserts(tmp_path) -> None:
    path = tmp_path / "questions.sqlite3"
    database = QuestionDatabase(path)

    assert database.upsert([_record()]) == {"inserted": 1}
    assert database.upsert([_record()]) == {"unchanged": 1}
    assert database.upsert([_record("Новый вопрос")]) == {"updated": 1}

    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        row = connection.execute(
            "SELECT question, answer, explanation FROM questions"
        ).fetchone()
    assert tables == [("questions",)]
    assert row == ("Новый вопрос", "Ответ", "Объяснение")
