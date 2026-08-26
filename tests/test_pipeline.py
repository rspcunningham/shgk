from __future__ import annotations

import sqlite3

from shgk.database import QuestionDatabase
from shgk.models import QuestionRecord
from shgk.pipeline import BasicFilterPipeline


def _record(
    identifier: str,
    *,
    question: str = "Вопрос",
    answer: str = "Ответ",
    explanation: str = "Объяснение",
    media: str = "[]",
) -> QuestionRecord:
    return QuestionRecord(
        source="test",
        source_question_id=identifier,
        source_url=f"https://example.test/{identifier}",
        game_kind="sport_chgk",
        question=question,
        answer=answer,
        explanation=explanation,
        media_urls_json=media,
        fetched_at="2025-01-01T00:00:00+00:00",
    )


def test_basic_filter_materializes_results_and_reasons(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert(
        [
            _record("eligible"),
            _record("no-answer", answer=""),
            _record("no-explanation", explanation=""),
            _record("media", media='[{"url":"https://example.test/image.jpg"}]'),
            _record("multiple", answer="", explanation=""),
        ]
    )

    pipeline = BasicFilterPipeline(source_path, pipeline_path)
    assert pipeline.run() == {"total": 5, "eligible": 1, "rejected": 4}
    stats = pipeline.stats()

    assert stats["by_source"] == [
        {"source": "test", "total": 5, "eligible": 1, "rejected": 4}
    ]
    assert {row["reason"]: row["questions"] for row in stats["rejection_reasons"]} == {
        "missing_answer": 2,
        "missing_explanation": 2,
        "has_media": 1,
    }

    with sqlite3.connect(pipeline_path) as connection:
        columns = connection.execute(
            "SELECT source_question_id, eligible FROM basic_filter_results ORDER BY 1"
        ).fetchall()
    assert columns == [
        ("eligible", 1),
        ("media", 0),
        ("multiple", 0),
        ("no-answer", 0),
        ("no-explanation", 0),
    ]


def test_basic_filter_updates_when_source_content_changes(tmp_path) -> None:
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    source = QuestionDatabase(source_path)
    source.upsert([_record("one", explanation="")])
    pipeline = BasicFilterPipeline(source_path, pipeline_path)
    assert pipeline.run()["eligible"] == 0

    source.upsert([_record("one", explanation="Теперь есть")])

    assert pipeline.run() == {"total": 1, "eligible": 1, "rejected": 0}
